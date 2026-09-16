"""
nlp_parser.py
=============
Task 1: NLP User Understanding Module (Clean, Robust & Concise)
- Multilingual intent classification & food NER (Arabic, English, Code-switching)
- Deterministic language & gender fallback guards
- Selective onboarding state machine for Task 3 & Task 4
"""

import json
import re
from datetime import datetime
from typing import List, Optional, Literal, Tuple
from pydantic import BaseModel, Field
import ollama

# ==========================================
# 1. Output Schema (Pydantic)
# ==========================================

DEFAULT_MODEL: str = "qwen2.5:3b"

INTENT_TYPES = Literal[
    "calculate_calories", "calculate_macros", "food_nutrition",
    "meal_recommendation", "daily_plan", "food_substitution",
    "fitness_nutrition_question", "update_profile", "general_chat"
]

class FoodItem(BaseModel):
    food_name_original: str = Field(..., description="Original food name from input")
    food_name_en: str       = Field(..., description="Standard English USDA-compatible food name")
    quantity: float         = Field(..., description="Numeric quantity e.g. 200.0, 1.0")
    unit: str               = Field(..., description="Unit: gram, cup, piece, slice, bowl, etc.")

class UserProfileExtract(BaseModel):
    birth_date:           Optional[str]                                                  = Field(..., description="YYYY-MM-DD or null")
    gender:               Optional[Literal["male", "female"]]                            = Field(..., description="male, female, or null")
    weight:               Optional[float]                                                = Field(..., description="Weight in kg or null")
    height:               Optional[float]                                                = Field(..., description="Height in cm or null")
    activity_level:       Optional[Literal["sedentary", "light", "moderate", "active", "very_active"]] = Field(..., description="Activity level or null")
    goal:                 Optional[Literal["weight_loss", "maintenance", "weight_gain", "muscle_gain"]] = Field(..., description="Goal or null")
    dietary_restrictions: List[str]                                                      = Field(default=[], description="e.g. ['keto']")
    forbidden_foods:      List[str]                                                      = Field(default=[], description="e.g. ['eggs']")
    preferred_foods:      List[str]                                                      = Field(default=[], description="e.g. ['chicken']")
    allergies:            List[str]                                                      = Field(default=[], description="e.g. ['dairy']")

class NLPOutput(BaseModel):
    intent: INTENT_TYPES = Field(..., description="User intent according to taxonomy")
    language_detected: Literal["arabic", "english", "mixed"] = Field(..., description="Language detected")
    raw_text: str = Field(..., description="Original user input echoed verbatim")
    foods: List[FoodItem] = Field(..., description="Extracted foods list ([] if none)")
    meal_type: Optional[Literal["breakfast", "lunch", "dinner", "snack", "all_day"]] = Field(..., description="Meal type or null")
    target_calories: Optional[float] = Field(..., description="Target calories or null")
    requested_substitution: Optional[str] = Field(..., description="Substituted food or null")
    profile_data: Optional[UserProfileExtract] = Field(..., description="Profile object or null")

# ==========================================
# 2. Deterministic Helpers & Normalization
# ==========================================

UNIT_MAP = {
    "g": "gram", "grams": "gram", "غم": "gram", "غرام": "gram", "جرام": "gram",
    "كوب": "cup", "أكواب": "cup", "cups": "cup", "قطعة": "piece", "قطع": "piece",
    "شريحة": "slice", "شرايح": "slice", "طبق": "bowl", "ml": "milliliter",
    "tbsp": "tablespoon", "tsp": "teaspoon", "can": "can", "bottle": "bottle",
    "loaf": "loaf", "رغيف": "loaf", "handful": "handful"
}

EMPTY_FOODS_INTENTS = {
    "meal_recommendation", "fitness_nutrition_question", "daily_plan",
    "food_substitution", "update_profile", "general_chat"
}

NON_FOOD_STOPWORDS = {
    "before workout", "after workout", "without eggs", "breakfast meal", "lunch meal",
    "dinner meal", "snack", "item", "meal", "substitute", "food", "diet",
    "وجبة فطور", "وجبة غداء", "وجبة عشاء", "بدون بيض", "قبل التمرين", "بعد التمرين"
}

def detect_language(text: str) -> Literal["arabic", "english", "mixed"]:
    """Classify language by Arabic Unicode vs Latin characters."""
    ar = len(re.findall(r'[\u0600-\u06FF]', text))
    en = len(re.findall(r'[a-zA-Z]', text))
    if ar == 0: return "english"
    if en == 0: return "arabic"
    if en <= 2 and ar >= 6: return "arabic"
    if ar <= 1 and en >= 10: return "english"
    return "mixed"

def _extract_gender_from_text(text: str) -> Optional[str]:
    """Deterministic gender keyword extractor fallback."""
    t = f" {re.sub(r'[^a-zA-Z\u0600-\u06FF]', ' ', text.lower())} "
    for w in [" female ", " woman ", " girl ", "بنت", "أنثى", "انثى", "ست", "امرأة"]:
        if w.strip() in t.split() or w in t: return "female"
    for w in [" male ", " man ", " boy ", "ولد", "ذكر", "راجل", "رجل"]:
        if w.strip() in t.split() or w in t: return "male"
    return None

def sanitize_extracted_values(p: dict) -> dict:
    """Sanity checks for height (mm auto-fix), weight, and target calories."""
    prof = p.get("profile_data")
    if prof:
        if prof.get("height") is not None:
            try:
                h = float(prof["height"])
                if h >= 500: h /= 10.0
                prof["height"] = round(h, 1) if 50.0 <= h <= 260.0 else None
            except (ValueError, TypeError): prof["height"] = None
        if prof.get("weight") is not None:
            try:
                w = float(prof["weight"])
                prof["weight"] = round(w, 1) if 20.0 <= w <= 350.0 else None
            except (ValueError, TypeError): prof["weight"] = None

    if p.get("target_calories") is not None:
        try:
            tc = float(p["target_calories"])
            p["target_calories"] = round(tc, 1) if 50.0 <= tc <= 15000.0 else None
        except (ValueError, TypeError): p["target_calories"] = None
    return p

def normalize_output(p: dict) -> dict:
    """Clean units, remove non-food artifacts, and enforce safe defaults."""
    lang = detect_language(p.get("raw_text", ""))
    intent = p.get("intent")

    if intent in EMPTY_FOODS_INTENTS:
        p["foods"] = []
    else:
        clean = []
        seen = set()
        for f in p.get("foods", []):
            name_en = f.get("food_name_en", "").strip().lower()
            name_orig = f.get("food_name_original", "").strip()

            if name_en in NON_FOOD_STOPWORDS or name_orig.lower() in NON_FOOD_STOPWORDS:
                continue
            if lang == "english" and re.search(r'[\u0600-\u06FF]', name_orig):
                name_orig = name_en

            u = UNIT_MAP.get(str(f.get("unit", "")).lower(), f.get("unit", "gram"))
            try:
                q = float(f.get("quantity", 1.0))
                if q <= 0: q = 1.0
            except (ValueError, TypeError):
                q = 1.0
            key = (name_en, q, u)

            if key not in seen and name_en:
                seen.add(key)
                clean.append({"food_name_original": name_orig, "food_name_en": name_en, "quantity": q, "unit": u})
        p["foods"] = clean

    prof = p.get("profile_data")
    if prof and not prof.get("gender"):
        prof["gender"] = _extract_gender_from_text(p.get("raw_text", ""))

    return sanitize_extracted_values(p)

# ==========================================
# 3. Prompt & Few-Shot Examples
# ==========================================

FEW_SHOTS = [
    # 1. calculate_calories (Arabic)
    {"role": "user", "content": "كلت 200g فراخ مشوية مع طبق رز وسلطة"},
    {"role": "assistant", "content": json.dumps({
        "intent": "calculate_calories", "language_detected": "arabic",
        "raw_text": "كلت 200g فراخ مشوية مع طبق رز وسلطة",
        "foods": [
            {"food_name_original": "فراخ مشوية", "food_name_en": "grilled chicken breast", "quantity": 200.0, "unit": "gram"},
            {"food_name_original": "طبق رز", "food_name_en": "cooked white rice", "quantity": 1.0, "unit": "bowl"},
            {"food_name_original": "سلطة", "food_name_en": "salad", "quantity": 1.0, "unit": "bowl"}
        ],
        "meal_type": None, "target_calories": None, "requested_substitution": None, "profile_data": None
    }, ensure_ascii=False)},
    
    # 2. calculate_calories (English)
    {"role": "user", "content": "I had 2 boiled eggs and a cup of oats for breakfast"},
    {"role": "assistant", "content": json.dumps({
        "intent": "calculate_calories", "language_detected": "english",
        "raw_text": "I had 2 boiled eggs and a cup of oats for breakfast",
        "foods": [
            {"food_name_original": "boiled eggs", "food_name_en": "boiled egg", "quantity": 2.0, "unit": "piece"},
            {"food_name_original": "oats", "food_name_en": "oats", "quantity": 1.0, "unit": "cup"}
        ],
        "meal_type": "breakfast", "target_calories": None, "requested_substitution": None, "profile_data": None
    }, ensure_ascii=False)},

    # 3. food_nutrition (Single food query)
    {"role": "user", "content": "How many calories are in 150g sweet potato?"},
    {"role": "assistant", "content": json.dumps({
        "intent": "food_nutrition", "language_detected": "english",
        "raw_text": "How many calories are in 150g sweet potato?",
        "foods": [{"food_name_original": "sweet potato", "food_name_en": "sweet potato", "quantity": 150.0, "unit": "gram"}],
        "meal_type": None, "target_calories": None, "requested_substitution": None, "profile_data": None
    }, ensure_ascii=False)},

    # 4. meal_recommendation
    {"role": "user", "content": "Make me a 400-calorie breakfast without eggs"},
    {"role": "assistant", "content": json.dumps({
        "intent": "meal_recommendation", "language_detected": "english",
        "raw_text": "Make me a 400-calorie breakfast without eggs",
        "foods": [], "meal_type": "breakfast", "target_calories": 400.0, "requested_substitution": "eggs", "profile_data": None
    }, ensure_ascii=False)},

    # 5. daily_plan
    {"role": "user", "content": "ممكن تعملي خطة أكل ليوم كامل؟"},
    {"role": "assistant", "content": json.dumps({
        "intent": "daily_plan", "language_detected": "arabic",
        "raw_text": "ممكن تعملي خطة أكل ليوم كامل؟",
        "foods": [], "meal_type": "all_day", "target_calories": None, "requested_substitution": None, "profile_data": None
    }, ensure_ascii=False)},

    # 6. fitness_nutrition_question
    {"role": "user", "content": "أكل إيه قبل التمرين عشان يديني طاقة؟"},
    {"role": "assistant", "content": json.dumps({
        "intent": "fitness_nutrition_question", "language_detected": "arabic",
        "raw_text": "أكل إيه قبل التمرين عشان يديني طاقة؟",
        "foods": [], "meal_type": "snack", "target_calories": None, "requested_substitution": None, "profile_data": None
    }, ensure_ascii=False)},

    # 7. food_substitution
    {"role": "user", "content": "What can I substitute for cheese in my diet?"},
    {"role": "assistant", "content": json.dumps({
        "intent": "food_substitution", "language_detected": "english",
        "raw_text": "What can I substitute for cheese in my diet?",
        "foods": [], "meal_type": None, "target_calories": None, "requested_substitution": "cheese", "profile_data": None
    }, ensure_ascii=False)},

    # 8. update_profile
    {"role": "user", "content": "أنا ولد اتولدت 2004-05-10، وزني 90 وطولي 175، بتمرن 4 أيام وعايز أخس، مش باكل بيض وعندي حساسية ألبان"},
    {"role": "assistant", "content": json.dumps({
        "intent": "update_profile", "language_detected": "arabic",
        "raw_text": "أنا ولد اتولدت 2004-05-10، وزني 90 وطولي 175، بتمرن 4 أيام وعايز أخس، مش باكل بيض وعندي حساسية ألبان",
        "foods": [], "meal_type": None, "target_calories": None, "requested_substitution": None,
        "profile_data": {
            "birth_date": "2004-05-10", "gender": "male", "weight": 90.0, "height": 175.0,
            "activity_level": "moderate", "goal": "weight_loss", "dietary_restrictions": [],
            "forbidden_foods": ["eggs"], "preferred_foods": [], "allergies": ["dairy"]
        }
    }, ensure_ascii=False)}
]

def parse_user_input(msg: str, history: Optional[List[dict]] = None, model: str = DEFAULT_MODEL) -> dict:
    """Unified single/multi-turn parser with retry and fallback."""
    curr_yr = datetime.now().year
    system_prompt = (
        f"You are a precise multilingual fitness NLP parser. Date: {datetime.now().strftime('%Y-%m-%d')} (Year: {curr_yr}).\n"
        "INTENT TAXONOMY:\n"
        "1. calculate_calories         → user is logging food eaten, meals, or ingredients consumed (e.g. 'أكلت نص كيلو لحمة', 'I used 0.5 cup of olive oil', 'أكلت طبق كشري وعصير قصب')\n"
        "2. calculate_macros           → user asks explicitly for macronutrient breakdown (e.g. 'how much protein and carbs in this?')\n"
        "3. food_nutrition             → user asks calories/nutrition of a specific item ('How many calories in 100g avocado?')\n"
        "4. meal_recommendation        → user requests a single meal idea with calorie budget ('400-calorie breakfast')\n"
        "5. daily_plan                 → user requests a full day or multi-day diet plan ('خطة أكل ليوم كامل')\n"
        "6. food_substitution          → user asks what to substitute for a food ('what to substitute for cheese?')\n"
        "7. fitness_nutrition_question → user asks fitness/exercise workout nutrition advice ('أكل إيه قبل التمرين؟')\n"
        "8. update_profile             → user provides biometrics (age/weight/height/goal/activity)\n"
        "9. general_chat               → greetings, thank you, chit-chat\n"
        "RULES:\n"
        "- Extract foods list ONLY for calculate_calories, calculate_macros, and food_nutrition. For any meal/food eaten, extract all mentioned foods/drinks (e.g. كشري, عصير قصب, olive oil, lemon juice). For other intents, foods MUST be [].\n"
        "- If age is given (e.g. 22 yrs), calculate birth_date as (Year - age)-01-01.\n"
        "- Return valid JSON matching schema."
    )

    messages = [{"role": "system", "content": system_prompt}] + FEW_SHOTS + (history or [])[-6:] + [{"role": "user", "content": msg}]

    for attempt in range(2):
        try:
            res = ollama.chat(model=model, messages=messages, format=NLPOutput.model_json_schema(), options={"temperature": 0.0})
            p = json.loads(res["message"]["content"])
            p["language_detected"] = detect_language(msg)
            p["raw_text"] = msg
            return normalize_output(p)
        except Exception:
            if attempt == 1:
                return normalize_output({
                    "intent": "general_chat", "language_detected": detect_language(msg),
                    "raw_text": msg, "foods": [], "meal_type": None,
                    "target_calories": None, "requested_substitution": None, "profile_data": None
                })

def parse_with_context(msg: str, history: Optional[List[dict]] = None, model: str = DEFAULT_MODEL) -> dict:
    return parse_user_input(msg, history=history, model=model)

# ==========================================
# 4. Standalone Split Helpers for Teammates
# ==========================================

def get_food_data(msg: str, model: str = DEFAULT_MODEL) -> dict:
    """Extracts meal data with metadata for Task 2 (USDA API) & Task 4 (Agent)."""
    p = parse_user_input(msg, model=model)
    return {
        "intent": p.get("intent"), "language_detected": p.get("language_detected"),
        "raw_text": p.get("raw_text", msg), "meal_type": p.get("meal_type"),
        "target_calories": p.get("target_calories"), "requested_substitution": p.get("requested_substitution"),
        "foods": p.get("foods", [])
    }

def get_user_profile(msg: str, model: str = DEFAULT_MODEL) -> Optional[dict]:
    """Extracts 10-field profile for Task 3 (BMR/TDEE calculation)."""
    return parse_user_input(msg, model=model).get("profile_data")

# ==========================================
# 5. Selective Onboarding Session Manager
# ==========================================

MANDATORY_FIELDS = ["birth_date", "gender", "weight", "height", "activity_level", "goal"]
PROFILE_DEPENDENT_INTENTS = {"daily_plan", "update_profile"}

def validate_mandatory_profile_fields(prof: dict) -> Tuple[bool, List[str]]:
    if not prof: return False, MANDATORY_FIELDS[:]
    missing = [k for k in MANDATORY_FIELDS if prof.get(k) is None]
    return (len(missing) == 0), missing

def merge_profile(existing: dict, new_data: dict, raw_text: Optional[str] = None) -> dict:
    merged = existing.copy()
    for k, v in new_data.items():
        if k in {"dietary_restrictions", "forbidden_foods", "preferred_foods", "allergies"}:
            if v: merged[k] = list(set(merged.get(k, []) + v))
        elif k == "gender":
            if existing.get("gender") is None and v is not None: merged["gender"] = v
            elif raw_text and _extract_gender_from_text(raw_text): merged["gender"] = _extract_gender_from_text(raw_text)
        elif v is not None:
            merged[k] = v
    return merged

class NLPSession:
    """Selective onboarding state machine: unblocks simple queries while enforcing profile on daily_plan."""
    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self.profile = {k: None for k in MANDATORY_FIELDS}
        self.profile.update({"dietary_restrictions": [], "forbidden_foods": [], "preferred_foods": [], "allergies": []})
        self.onboarding_complete = False
        self.history = []

    def process(self, msg: str) -> dict:
        parsed = parse_with_context(msg, self.history, model=self.model)
        self.history.extend([{"role": "user", "content": msg}, {"role": "assistant", "content": json.dumps(parsed, ensure_ascii=False)}])

        if parsed.get("profile_data"):
            self.profile = merge_profile(self.profile, parsed["profile_data"], msg)

        is_complete, missing = validate_mandatory_profile_fields(self.profile)
        if is_complete: self.onboarding_complete = True

        intent = parsed.get("intent")
        lang = detect_language(msg)

        # Unblock simple queries; only enforce onboarding for profile-dependent flows
        if intent in PROFILE_DEPENDENT_INTENTS and not self.onboarding_complete:
            prompt = (
                "مرحباً! 👋 عشان أقدر أحسب سعراتك وخطة أكلك بدقة، محتاج باقي بياناتك (الطول، الوزن، الهدف، مستوى النشاط)."
                if lang == "arabic" else
                "👋 Welcome! To calculate your personalized plan accurately, please provide your missing details (height, weight, goal, activity level)."
            )
            return {"parsed": parsed, "status": "needs_profile", "prompt_for_user": prompt, "profile": self.profile, "missing_fields": missing}

        parsed["profile_data"] = self.profile

        if intent == "update_profile" and self.onboarding_complete:
            confirm = "✅ ممتاز! تم حفظ بياناتك بنجاح. تقدر تسجل وجباتك أو تطلب خطة أكل." if lang == "arabic" else "✅ Profile complete! You can now log meals or ask for meal plans."
            return {"parsed": parsed, "status": "onboarding_complete", "prompt_for_user": confirm, "profile": self.profile, "missing_fields": []}

        return {"parsed": parsed, "status": "ready", "prompt_for_user": None, "profile": self.profile, "missing_fields": []}