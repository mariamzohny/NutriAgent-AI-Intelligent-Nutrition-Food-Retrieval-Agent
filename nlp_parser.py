import json
import re
from datetime import datetime
from typing import List, Optional, Literal, Tuple
from pydantic import BaseModel, Field
import ollama


# ==========================================
# 0. Global Configuration
# ==========================================

DEFAULT_MODEL: str = "qwen2.5:3b"

# Intents supported across the project
INTENT_TYPES = Literal[
    "calculate_calories",          # Log meal or calculate meal calories
    "calculate_macros",            # Specific macro breakdown request
    "food_nutrition",              # Nutrition query about a single food/portion
    "meal_recommendation",         # Single meal idea/recipe (e.g. "400 cal breakfast")
    "daily_plan",                  # Full day/multi-day diet plan
    "food_substitution",           # What can I substitute for X?
    "fitness_nutrition_question",  # Workout/sports nutrition advice (FAISS RAG trigger)
    "update_profile",              # Profile metrics & onboarding
    "general_chat"                 # Greetings / chit-chat
]


# ==========================================
# 1. Output Schema (Pydantic)
# ==========================================

class FoodItem(BaseModel):
    food_name_original: str = Field(..., description="Original food name as written by the user")
    food_name_en: str       = Field(..., description="Standard English USDA-compatible food name")
    quantity: float          = Field(..., description="Numeric quantity e.g. 0.5, 1.0, 200.0")
    unit: str                = Field(..., description="Unit: gram, cup, piece, slice, can, bottle, tablespoon, bowl, loaf, milliliter")


class UserProfileExtract(BaseModel):
    birth_date:           Optional[str]                                                                    = Field(..., description="YYYY-MM-DD or null")
    gender:               Optional[Literal["male", "female"]]                                              = Field(..., description="male, female, or null")
    weight:               Optional[float]                                                                  = Field(..., description="Weight in kg or null")
    height:               Optional[float]                                                                  = Field(..., description="Height in cm or null")
    activity_level:       Optional[Literal["sedentary", "light", "moderate", "active", "very_active"]]    = Field(..., description="Activity level or null")
    goal:                 Optional[Literal["weight_loss", "maintenance", "weight_gain", "muscle_gain"]]   = Field(..., description="User goal or null")
    dietary_restrictions: List[str]                                                                        = Field(default=[], description="e.g. ['keto', 'vegan'] or []")
    forbidden_foods:      List[str]                                                                        = Field(default=[], description="Foods user avoids e.g. ['eggs'] or []")
    preferred_foods:      List[str]                                                                        = Field(default=[], description="Foods user prefers or []")
    allergies:            List[str]                                                                        = Field(default=[], description="Medical allergies e.g. ['dairy'] or []")


class NLPOutput(BaseModel):
    intent: INTENT_TYPES = Field(
        ..., description="User intent according to project specifications"
    )
    language_detected: Literal["arabic", "english", "mixed"] = Field(
        ..., description="Detected input language"
    )
    raw_text: str = Field(..., description="Original user input, echoed verbatim")
    foods: List[FoodItem] = Field(..., description="Extracted food items ([] if none)")
    
    # Specific fields for meal planning and recommendation requests (Critical Gap 1 & 2)
    meal_type: Optional[Literal["breakfast", "lunch", "dinner", "snack", "all_day"]] = Field(
        ..., description="Specific meal requested: breakfast, lunch, dinner, snack, all_day, or null"
    )
    target_calories: Optional[float] = Field(
        ..., description="Explicit target calories requested (e.g. 400.0 for '400-calorie breakfast') or null"
    )
    requested_substitution: Optional[str] = Field(
        ..., description="Food to be substituted when intent is food_substitution (e.g. 'eggs', 'cheese') or null"
    )

    profile_data: Optional[UserProfileExtract] = Field(
        ...,
        description="Full profile object when intent is update_profile, otherwise null"
    )


# ==========================================
# 2. Language Detection (Deterministic — no LLM)
# ==========================================

def detect_language(text: str) -> Literal["arabic", "english", "mixed"]:
    """
    Classify input language by Unicode character counts.
    Arabic block: U+0600–U+06FF
    Returns 'arabic', 'english', or 'mixed'.
    """
    arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
    latin_chars  = len(re.findall(r'[a-zA-Z]', text))
    total = arabic_chars + latin_chars
    if total == 0:
        return "english"
    if arabic_chars == 0:
        return "english"
    if latin_chars == 0:
        return "arabic"
    
    # Common short Latin unit abbreviations in Arabic (e.g. "200g", "175cm", "90kg")
    if latin_chars <= 2 and arabic_chars >= 6:
        return "arabic"
    if arabic_chars <= 1 and latin_chars >= 10:
        return "english"
        
    return "mixed"


# ==========================================
# 3. Normalization & Sanity Checks
# ==========================================

UNIT_ALIASES: dict = {
    # English plurals / abbreviations
    "grams": "gram", "g": "gram",
    "cups": "cup",
    "pieces": "piece",
    "slices": "slice",
    "tablespoons": "tablespoon", "tbsp": "tablespoon",
    "teaspoons": "teaspoon", "tsp": "teaspoon",
    "bowls": "bowl",
    "cans": "can",
    "bottles": "bottle",
    "loaves": "loaf",
    "ml": "milliliter", "mls": "milliliter",
    "liters": "liter", "l": "liter",
    # Arabic units
    "غم": "gram", "غرام": "gram", "جرام": "gram",
    "كوب": "cup", "أكواب": "cup",
    "قطعة": "piece", "قطع": "piece",
    "شريحة": "slice", "شرايح": "slice",
    "ملعقة كبيرة": "tablespoon",
    "ملعقة صغيرة": "teaspoon",
    "طبق": "bowl", "أطباق": "bowl",
    "علبة": "can",
    "زجاجة": "bottle",
    "رغيف": "loaf",
}

# ── Deterministic gender keywords (checked after LLM output as a fallback) ──
_GENDER_FEMALE_KEYWORDS: List[str] = [
    # Arabic
    "أنثى", "انثى", "بنت", "ست ", "ستة", "امرأة", "فتاة",
    "أنا بنت", "أنا أنثى",
    # English whole words (checked with surrounding whitespace)
    " female ", " woman ", " girl ", " female,", " female.",
]

_GENDER_MALE_KEYWORDS: List[str] = [
    # Arabic
    "ذكر", "ولد", "راجل", "رجل", "صبي", "أنا ذكر", "أنا ولد",
    # English whole words (checked with surrounding whitespace)
    " male ", " man ", " boy ", " male,", " male.",
]


def _extract_gender_from_text(text: str) -> Optional[str]:
    """
    Deterministic gender extraction from well-known Arabic/English keywords.
    Checks female keywords first to avoid 'female' containing 'male' substring.
    """
    cleaned = re.sub(r'[^a-zA-Z\u0600-\u06FF\s]', ' ', text.lower())
    padded = f" {cleaned} "
    
    for kw in _GENDER_FEMALE_KEYWORDS:
        if kw.strip() in padded.split() or kw in padded:
            return "female"
    for kw in _GENDER_MALE_KEYWORDS:
        if kw.strip() in padded.split() or kw in padded:
            return "male"
    return None


def sanitize_extracted_values(parsed: dict) -> dict:
    """
    Sanity checks for extracted biometric & metric numbers (Critical Gap 7):
    - Height: Fix mm (1750 -> 175) or clamp between 50cm and 260cm.
    - Weight: Clamp/validate between 20kg and 350kg.
    - Target calories: Clamp between 50 and 10000 kcal.
    """
    profile = parsed.get("profile_data")
    if profile:
        # Height sanity
        if profile.get("height") is not None:
            try:
                h = float(profile["height"])
                if h >= 500:   # e.g. 1750 mm -> 175.0 cm
                    h = h / 10.0
                if 50.0 <= h <= 260.0:
                    profile["height"] = round(h, 1)
                else:
                    profile["height"] = None
            except (ValueError, TypeError):
                profile["height"] = None

        # Weight sanity
        if profile.get("weight") is not None:
            try:
                w = float(profile["weight"])
                if 20.0 <= w <= 350.0:
                    profile["weight"] = round(w, 1)
                else:
                    profile["weight"] = None
            except (ValueError, TypeError):
                profile["weight"] = None

    # Target calories sanity
    if parsed.get("target_calories") is not None:
        try:
            tc = float(parsed["target_calories"])
            if 50.0 <= tc <= 15000.0:
                parsed["target_calories"] = round(tc, 1)
            else:
                parsed["target_calories"] = None
        except (ValueError, TypeError):
            parsed["target_calories"] = None

    return parsed


NON_FOOD_STOP_WORDS = {
    "before workout", "after workout", "without eggs", "breakfast meal", "lunch meal",
    "dinner meal", "snack", "item", "meal", "substitute", "food", "diet",
    "وجبة فطور", "وجبة غداء", "وجبة عشاء", "بدون بيض", "قبل التمرين", "بعد التمرين"
}

EMPTY_FOODS_INTENTS = {
    "meal_recommendation",
    "fitness_nutrition_question",
    "daily_plan",
    "food_substitution",
    "update_profile",
    "general_chat"
}


def normalize_output(parsed: dict) -> dict:
    """
    Post-process LLM output:
    - Normalize unit strings to canonical form.
    - Clamp non-positive quantities to 1.0.
    - Lowercase food_name_en for USDA search compatibility.
    - Remove duplicate food entries and non-food descriptor artifacts.
    - Ensure foods is empty for recommendation/question/profile intents.
    - Ensure food_name_original matches the language of raw_text.
    - Deterministic gender fallback if profile_data.gender is null.
    - Run sanity checks on biometric values.
    """
    intent = parsed.get("intent")
    raw_text = parsed.get("raw_text", "")
    lang = detect_language(raw_text)

    # ── If intent does NOT log an eaten meal, foods MUST be [] ───────────────
    if intent in EMPTY_FOODS_INTENTS:
        parsed["foods"] = []
    else:
        # ── Food normalization & artifact removal ─────────────────────────────
        seen: set = set()
        clean_foods: list = []
        for food in parsed.get("foods", []):
            name_en = food.get("food_name_en", "").strip().lower()
            name_orig = food.get("food_name_original", "").strip()

            # Filter out non-food stopwords
            if name_en in NON_FOOD_STOP_WORDS or name_orig.lower() in NON_FOOD_STOP_WORDS:
                continue

            # Fix food_name_original: on English inputs, never keep an Arabic translation hallucination
            if lang == "english" and re.search(r'[\u0600-\u06FF]', name_orig):
                name_orig = name_en
            food["food_name_original"] = name_orig

            raw_unit = str(food.get("unit", "")).strip().lower()
            food["unit"] = UNIT_ALIASES.get(raw_unit, raw_unit)
            if food.get("quantity", 0) <= 0:
                food["quantity"] = 1.0
            food["food_name_en"] = name_en
            
            key = (food["food_name_en"], food["quantity"], food["unit"])
            if key not in seen and name_en:
                seen.add(key)
                clean_foods.append(food)
        parsed["foods"] = clean_foods

    # ── Deterministic gender fallback ─────────────────────────────────────────
    profile = parsed.get("profile_data")
    if profile and profile.get("gender") is None:
        detected_gender = _extract_gender_from_text(raw_text)
        if detected_gender:
            profile["gender"] = detected_gender

    # ── Sanity checks ─────────────────────────────────────────────────────────
    return sanitize_extracted_values(parsed)


# ==========================================
# 4. System Prompt & Dynamic Date Injection
# ==========================================

def get_system_prompt() -> str:
    """
    Build system prompt dynamically injecting current date & year (Critical Gap 4).
    """
    current_date = datetime.now().strftime("%Y-%m-%d")
    current_year = datetime.now().year
    
    return f"""You are a precise multilingual NLP parser for a fitness calorie-tracking and nutrition assistant.
You understand Arabic (including Egyptian slang), English, and Arabic-English mixed (code-switched) text.

CURRENT RUNTIME DATE: {current_date} (Current Year: {current_year})

INTENT TAXONOMY (Strictly map to one of these):
1. calculate_calories         → user is logging/recording food they ate or will eat (e.g. "كلت 200g فراخ", "I had 2 eggs and toast")
2. calculate_macros           → user specifically asks for macro breakdown (protein/carbs/fats) of their meal
3. food_nutrition             → user asks for calories/macros of a specific item (e.g. "How many calories in 100g avocado?", "كم سعرة في الموز؟")
4. meal_recommendation        → user asks for a single meal suggestion with specific constraints/calories (e.g. "Make me a 400-calorie breakfast without eggs", "عايز فطار 400 سعرة بدون بيض")
5. daily_plan                 → user asks for a full day or multi-day meal plan (e.g. "خطة أكل ليوم كامل", "give me a full day meal plan")
6. food_substitution          → user asks what to substitute for a specific food (e.g. "what can I substitute for cheese?", "بديل البيض في الدايت")
7. fitness_nutrition_question → user asks fitness/exercise nutrition advice (e.g. "what should I eat before workout?", "أكل إيه قبل التمرين؟", "best post workout protein")
8. update_profile             → user provides personal metrics (age, date of birth, weight, height, goal, allergies, activity level)
9. general_chat               → greetings, thank you, off-topic, chit-chat

OUTPUT RULES:
1. Always return valid JSON matching the schema.
2. raw_text: copy the user's exact input.
3. language_detected: set to "arabic", "english", or "mixed".
4. meal_type: set to "breakfast", "lunch", "dinner", "snack", "all_day", or null.
5. target_calories: extract numeric calories if user specified a calorie budget (e.g. 400.0 for "400-calorie breakfast").
6. requested_substitution: extract the food name being substituted if intent is food_substitution (e.g. "eggs" or "cheese").
7. foods: list of food items. MUST be [] when intent is update_profile, fitness_nutrition_question, or general_chat.
8. profile_data: full object when intent is update_profile, otherwise MUST be null.

RULES FOR profile_data (when intent = update_profile):
Fill EVERY field — do NOT omit any key:
- birth_date:           "YYYY-MM-DD" (if user gives age e.g. 22 years, calculate as "{current_year - 22}-01-01") or null
- gender:               "male" | "female" | null
- weight:               number in kg (e.g. 80.0) | null
- height:               number in cm (e.g. 175.0) | null
- activity_level:       "sedentary" | "light" | "moderate" | "active" | "very_active" | null
- goal:                 "weight_loss" | "maintenance" | "weight_gain" | "muscle_gain" | null
- dietary_restrictions: [] or list (e.g. ["keto", "vegan"])
- forbidden_foods:      [] or list (e.g. ["eggs"])
- preferred_foods:      [] or list
- allergies:            [] or list (e.g. ["dairy", "peanuts"])

ACTIVITY LEVEL MAPPING:
- No exercise / desk job / مش بتمرن / مفيش رياضة        → sedentary
- Light exercise 1-3 days/week / تمرين خفيف / بسيط      → light
- Moderate exercise 3-5 days/week / تمرين متوسط         → moderate
- Hard exercise 6-7 days/week / جيم كل يوم / تدريب عنيف → active
- Very intense / twice daily / مرتين في اليوم           → very_active
"""


# ==========================================
# 4.1 Few-Shot Examples
# ==========================================

FEW_SHOT_EXAMPLES: List[dict] = [

    # ── 1. Meal Recommendation with Target Calories & Meal Type (Critical Gap 1)
    {"role": "user", "content": "Make me a 400-calorie breakfast without eggs"},
    {"role": "assistant", "content": json.dumps({
        "intent": "meal_recommendation", "language_detected": "english",
        "raw_text": "Make me a 400-calorie breakfast without eggs",
        "meal_type": "breakfast", "target_calories": 400.0, "requested_substitution": None,
        "foods": [], "profile_data": None
    }, ensure_ascii=False)},

    # ── 2. Arabic Meal Recommendation with Target Calories
    {"role": "user", "content": "عايز وجبة غداء في حدود 600 سعرة بدون سمك"},
    {"role": "assistant", "content": json.dumps({
        "intent": "meal_recommendation", "language_detected": "arabic",
        "raw_text": "عايز وجبة غداء في حدود 600 سعرة بدون سمك",
        "meal_type": "lunch", "target_calories": 600.0, "requested_substitution": None,
        "foods": [], "profile_data": None
    }, ensure_ascii=False)},

    # ── 3. Food Substitution (Critical Gap 2)
    {"role": "user", "content": "What can I substitute for cheese in my diet?"},
    {"role": "assistant", "content": json.dumps({
        "intent": "food_substitution", "language_detected": "english",
        "raw_text": "What can I substitute for cheese in my diet?",
        "meal_type": None, "target_calories": None, "requested_substitution": "cheese",
        "foods": [{"food_name_original": "cheese", "food_name_en": "cheese", "quantity": 1.0, "unit": "piece"}],
        "profile_data": None
    }, ensure_ascii=False)},

    # ── 4. Fitness Nutrition Question (FAISS RAG Trigger)
    {"role": "user", "content": "أكل إيه قبل التمرين عشان يديني طاقة؟"},
    {"role": "assistant", "content": json.dumps({
        "intent": "fitness_nutrition_question", "language_detected": "arabic",
        "raw_text": "أكل إيه قبل التمرين عشان يديني طاقة؟",
        "meal_type": "snack", "target_calories": None, "requested_substitution": None,
        "foods": [], "profile_data": None
    }, ensure_ascii=False)},

    # ── 5. Full Profile Onboarding (Arabic)
    {"role": "user", "content": "أنا ولد اتولدت 2004-05-10، وزني 90 وطولي 175، بتمرن 4 أيام في الأسبوع وعايز أخس، بس مش باكل بيض وعندي حساسية ألبان"},
    {"role": "assistant", "content": json.dumps({
        "intent": "update_profile", "language_detected": "arabic",
        "raw_text": "أنا ولد اتولدت 2004-05-10، وزني 90 وطولي 175، بتمرن 4 أيام في الأسبوع وعايز أخس، بس مش باكل بيض وعندي حساسية ألبان",
        "meal_type": None, "target_calories": None, "requested_substitution": None,
        "foods": [],
        "profile_data": {
            "birth_date": "2004-05-10", "gender": "male",
            "weight": 90.0, "height": 175.0,
            "activity_level": "moderate", "goal": "weight_loss",
            "dietary_restrictions": [], "forbidden_foods": ["eggs"],
            "preferred_foods": [], "allergies": ["dairy"]
        }
    }, ensure_ascii=False)},

    # ── 6. Log Meal / Calculate Calories (Arabic)
    {"role": "user", "content": "كلت 200g فراخ مشوية مع طبق رز"},
    {"role": "assistant", "content": json.dumps({
        "intent": "calculate_calories", "language_detected": "arabic",
        "raw_text": "كلت 200g فراخ مشوية مع طبق رز",
        "meal_type": "lunch", "target_calories": None, "requested_substitution": None,
        "foods": [
            {"food_name_original": "فراخ مشوية", "food_name_en": "grilled chicken breast", "quantity": 200.0, "unit": "gram"},
            {"food_name_original": "رز",          "food_name_en": "cooked white rice",       "quantity": 1.0,   "unit": "bowl"}
        ],
        "profile_data": None
    }, ensure_ascii=False)},

    # ── 7. Food Nutrition Query (Single item)
    {"role": "user", "content": "How many calories are in 200g chicken breast?"},
    {"role": "assistant", "content": json.dumps({
        "intent": "food_nutrition", "language_detected": "english",
        "raw_text": "How many calories are in 200g chicken breast?",
        "meal_type": None, "target_calories": None, "requested_substitution": None,
        "foods": [{"food_name_original": "chicken breast", "food_name_en": "chicken breast", "quantity": 200.0, "unit": "gram"}],
        "profile_data": None
    }, ensure_ascii=False)},

    # ── 8. Daily Plan Request
    {"role": "user", "content": "ممكن تعملي خطة أكل ليوم كامل؟"},
    {"role": "assistant", "content": json.dumps({
        "intent": "daily_plan", "language_detected": "arabic",
        "raw_text": "ممكن تعملي خطة أكل ليوم كامل؟",
        "meal_type": "all_day", "target_calories": None, "requested_substitution": None,
        "foods": [], "profile_data": None
    }, ensure_ascii=False)},

    # ── 9. General Chat
    {"role": "user", "content": "صباح الخير، إيه اللي تقدر تعمله؟"},
    {"role": "assistant", "content": json.dumps({
        "intent": "general_chat", "language_detected": "arabic",
        "raw_text": "صباح الخير، إيه اللي تقدر تعمله؟",
        "meal_type": None, "target_calories": None, "requested_substitution": None,
        "foods": [], "profile_data": None
    }, ensure_ascii=False)}
]


# ==========================================
# 5. Onboarding: Mandatory Fields & Prompts
# ==========================================

MANDATORY_FIELDS: List[str] = [
    "birth_date", "gender", "weight", "height", "activity_level", "goal"
]

# Only intents that calculate personalized BMR/TDEE targets require full onboarding (Critical Gap 3)
PROFILE_DEPENDENT_INTENTS = {"daily_plan", "update_profile"}

_FIELD_PROMPTS: dict = {
    "birth_date": {
        "ar": "📅 ما هو تاريخ ميلادك أو عمرك؟ (مثلاً: 2000-03-15 أو عندي 24 سنة)",
        "en": "📅 What is your date of birth or age? (e.g. 2000-03-15 or 24 years old)"
    },
    "gender": {
        "ar": "⚧  ما هو جنسك؟ (ذكر = male / أنثى = female)",
        "en": "⚧  What is your gender? (male / female)"
    },
    "weight": {
        "ar": "⚖️  كم وزنك بالكيلوجرام؟",
        "en": "⚖️  What is your weight in kilograms?"
    },
    "height": {
        "ar": "📏  كم طولك بالسنتيمتر؟",
        "en": "📏  What is your height in centimeters?"
    },
    "activity_level": {
        "ar": (
            "🏃  ما هو مستوى نشاطك البدني؟\n"
            "  • sedentary  — لا تمارس رياضة\n"
            "  • light      — ١-٣ أيام في الأسبوع\n"
            "  • moderate   — ٣-٥ أيام في الأسبوع\n"
            "  • active     — ٦-٧ أيام في الأسبوع\n"
            "  • very_active — تدريب مكثف يومياً"
        ),
        "en": (
            "🏃  What is your activity level?\n"
            "  • sedentary   — no exercise\n"
            "  • light       — 1-3 days/week\n"
            "  • moderate    — 3-5 days/week\n"
            "  • active      — 6-7 days/week\n"
            "  • very_active — intense daily training"
        )
    },
    "goal": {
        "ar": (
            "🎯  ما هو هدفك؟\n"
            "  • weight_loss  — إنقاص الوزن\n"
            "  • maintenance  — الحفاظ على الوزن\n"
            "  • weight_gain  — زيادة الوزن\n"
            "  • muscle_gain  — بناء العضلات"
        ),
        "en": (
            "🎯  What is your goal?\n"
            "  • weight_loss\n"
            "  • maintenance\n"
            "  • weight_gain\n"
            "  • muscle_gain"
        )
    }
}


def get_onboarding_prompt(missing_fields: List[str], lang: str = "english") -> str:
    key = "ar" if lang == "arabic" else "en"
    if lang == "arabic":
        header = "مرحباً! 👋 عشان أقدر أحسب سعراتك وخطة أكلك بدقة، محتاج بعض المعلومات الأساسية:\n\n"
    else:
        header = "👋 Welcome! To calculate your personalized calories and meal plan accurately, I need a few details:\n\n"

    lines = [_FIELD_PROMPTS[f][key] for f in missing_fields if f in _FIELD_PROMPTS]
    return header + "\n\n".join(lines)


# ==========================================
# 6. Profile Validation & Merging
# ==========================================

def validate_mandatory_profile_fields(profile: dict) -> Tuple[bool, List[str]]:
    if not profile:
        return False, MANDATORY_FIELDS[:]
    missing = [k for k in MANDATORY_FIELDS if profile.get(k) is None]
    return (len(missing) == 0), missing


def merge_profile(existing: dict, new_data: dict, raw_text: Optional[str] = None) -> dict:
    """
    Merge a newly parsed profile_data into a stored profile.
    - Scalar fields: only overwrite if new value is non-null.
    - If gender is already set, do not overwrite unless raw_text explicitly has a gender keyword.
    - List fields: union (no duplicates).
    """
    merged = existing.copy()
    list_fields = {"dietary_restrictions", "forbidden_foods", "preferred_foods", "allergies"}
    for key, value in new_data.items():
        if key in list_fields:
            if value:  # non-empty list → union
                merged[key] = list(set(merged.get(key, []) + value))
        elif key == "gender":
            if existing.get("gender") is None and value is not None:
                merged["gender"] = value
            elif raw_text and _extract_gender_from_text(raw_text) is not None:
                merged["gender"] = _extract_gender_from_text(raw_text)
        else:
            if value is not None:
                merged[key] = value
    return merged


# ==========================================
# 7. Core Parser Functions (with Retry & Fallback)
# ==========================================

def parse_user_input(user_message: str, model: str = DEFAULT_MODEL) -> dict:
    """
    Parse a single user message into a structured NLPOutput dict with retry & error handling (Critical Gap 5 & 6).
    """
    system_prompt = get_system_prompt()
    messages = [
        {"role": "system", "content": system_prompt},
        *FEW_SHOT_EXAMPLES,
        {"role": "user", "content": user_message}
    ]

    # Retry logic (1 retry if JSON decode or connection fails)
    for attempt in range(2):
        try:
            response = ollama.chat(
                model=model,
                messages=messages,
                format=NLPOutput.model_json_schema(),
                options={"temperature": 0.0}
            )
            parsed: dict = json.loads(response["message"]["content"])
            
            # Deterministic overrides
            parsed["language_detected"] = detect_language(user_message)
            parsed["raw_text"] = user_message

            return normalize_output(parsed)
        except Exception as e:
            if attempt == 1:
                # Safe fallback response (never crash)
                fallback = {
                    "intent": "general_chat",
                    "language_detected": detect_language(user_message),
                    "raw_text": user_message,
                    "meal_type": None,
                    "target_calories": None,
                    "requested_substitution": None,
                    "foods": [],
                    "profile_data": None
                }
                return normalize_output(fallback)


def parse_with_context(user_message: str, history: Optional[List[dict]] = None, model: str = DEFAULT_MODEL) -> dict:
    """
    Parse with conversation history for pronoun and context resolution.
    """
    history_turns = (history or [])[-6:]
    system_prompt = get_system_prompt()

    messages = [
        {"role": "system", "content": system_prompt},
        *FEW_SHOT_EXAMPLES,
        *history_turns,
        {"role": "user", "content": user_message}
    ]

    for attempt in range(2):
        try:
            response = ollama.chat(
                model=model,
                messages=messages,
                format=NLPOutput.model_json_schema(),
                options={"temperature": 0.0}
            )
            parsed: dict = json.loads(response["message"]["content"])
            parsed["language_detected"] = detect_language(user_message)
            parsed["raw_text"] = user_message

            return normalize_output(parsed)
        except Exception as e:
            if attempt == 1:
                fallback = {
                    "intent": "general_chat",
                    "language_detected": detect_language(user_message),
                    "raw_text": user_message,
                    "meal_type": None,
                    "target_calories": None,
                    "requested_substitution": None,
                    "foods": [],
                    "profile_data": None
                }
                return normalize_output(fallback)


# ==========================================
# 7.1 Dedicated Split Functions for Teammates
# ==========================================

def get_food_data(user_message: str, model: str = DEFAULT_MODEL) -> dict:
    """
    Extract the Meal/Food data along with message metadata (intent, language, raw_text, meal_type, target_calories).
    Perfect for Task 2 (USDA API search), Task 3 (food nutrition calculation), and Task 4 (Agent).

    Example:
        >>> get_food_data("Make me a 400-calorie breakfast without eggs")
        {
            "intent": "meal_recommendation",
            "language_detected": "english",
            "raw_text": "Make me a 400-calorie breakfast without eggs",
            "meal_type": "breakfast",
            "target_calories": 400.0,
            "requested_substitution": None,
            "foods": []
        }
    """
    parsed = parse_user_input(user_message, model=model)
    return {
        "intent":                 parsed.get("intent"),
        "language_detected":      parsed.get("language_detected"),
        "raw_text":               parsed.get("raw_text", user_message),
        "meal_type":              parsed.get("meal_type"),
        "target_calories":        parsed.get("target_calories"),
        "requested_substitution": parsed.get("requested_substitution"),
        "foods":                  parsed.get("foods", [])
    }


def get_user_profile(user_message: str, model: str = DEFAULT_MODEL) -> Optional[dict]:
    """
    Extract ONLY the 10-field User Profile from user message.
    Perfect for Task 3 (BMR, TDEE, target calories engine).
    """
    parsed = parse_user_input(user_message, model=model)
    return parsed.get("profile_data")


# ==========================================
# 8. NLPSession — Selective Onboarding & State Manager
# ==========================================

class NLPSession:
    """
    Stateful session manager for conversational interaction.

    Selective Onboarding (Critical Gap 3):
    - Simple queries (food_nutrition, fitness_nutrition_question, food_substitution, calculate_calories)
      are NEVER blocked, enabling simplified usage.
    - Profile-dependent intents (daily_plan, update_profile) check for the 6 mandatory fields.
    """

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model: str = model
        self.profile: dict = {
            "birth_date":           None,
            "gender":               None,
            "weight":               None,
            "height":               None,
            "activity_level":       None,
            "goal":                 None,
            "dietary_restrictions": [],
            "forbidden_foods":      [],
            "preferred_foods":      [],
            "allergies":            []
        }
        self.onboarding_complete: bool = False
        self.history: List[dict] = []
        self.language: str = "english"

    def process(self, user_message: str) -> dict:
        """
        Process one user turn.
        """
        self.language = detect_language(user_message)
        parsed = parse_with_context(user_message, self.history, model=self.model)

        # Append to history
        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": json.dumps(parsed, ensure_ascii=False)})

        # Merge extracted profile data if present
        if parsed.get("profile_data"):
            self.profile = merge_profile(self.profile, parsed["profile_data"], user_message)

        # Check mandatory fields
        is_complete, missing = validate_mandatory_profile_fields(self.profile)
        if is_complete:
            self.onboarding_complete = True

        intent = parsed.get("intent")

        # ── Case 1: Intent strictly requires profile AND profile is incomplete ──
        if intent in PROFILE_DEPENDENT_INTENTS and not self.onboarding_complete:
            prompt = get_onboarding_prompt(missing, self.language)
            return {
                "parsed":          parsed,
                "status":          "needs_profile",
                "prompt_for_user": prompt,
                "profile":         self.profile,
                "missing_fields":  missing
            }

        # ── Attach complete user profile so Task 3 engine always has access to it ──
        parsed["profile_data"] = self.profile

        # ── Case 2: Just completed onboarding this turn ────────────────────────
        if intent == "update_profile" and self.onboarding_complete:
            if self.language == "arabic":
                confirm = (
                    "✅ ممتاز! استلمت بياناتك بنجاح.\n"
                    "دلوقتي تقدر:\n"
                    "  • تسجل وجباتك (مثلاً: 'كلت 200g فراخ مشوية')\n"
                    "  • تطلب خطة أكل ('ممكن خطة أكل ليوم؟')\n"
                    "  • تسأل عن سعرات أي أكلة ('كم سعرة في الأفوكادو؟')\n"
                    "  • تسأل أسئلة تمارين وتغذية ('أكل إيه قبل التمرين؟')"
                )
            else:
                confirm = (
                    "✅ Profile complete! Here's what you can do now:\n"
                    "  • Log a meal (e.g. 'I had 200g grilled chicken')\n"
                    "  • Request a meal plan ('Give me a full day meal plan')\n"
                    "  • Ask about food calories ('How many calories in avocado?')\n"
                    "  • Ask fitness nutrition advice ('What to eat before workout?')"
                )
            return {
                "parsed":          parsed,
                "status":          "onboarding_complete",
                "prompt_for_user": confirm,
                "profile":         self.profile,
                "missing_fields":  []
            }

        # ── Case 3: Ready state (simplified query or complete profile) ─────────
        return {
            "parsed":          parsed,
            "status":          "ready",
            "prompt_for_user": None,
            "profile":         self.profile,
            "missing_fields":  []
        }


# ==========================================
# 9. Quick Test Execution
# ==========================================

if __name__ == "__main__":
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    def safe_print(*args, **kwargs):
        try:
            print(*args, **kwargs)
        except UnicodeEncodeError:
            safe_str = " ".join(str(a).encode("ascii", "replace").decode() for a in args)
            print(safe_str)

    safe_print("=" * 65)
    safe_print("NLP Parser — Task 1 Upgraded Demonstration")
    safe_print("=" * 65)

    session = NLPSession()

    test_queries = [
        # 1. Simplified usage (Should NEVER be blocked by onboarding!)
        "How many calories are in 200g chicken breast?",
        # 2. Meal recommendation with target calories & meal type (Gap 1)
        "Make me a 400-calorie breakfast without eggs",
        # 3. Fitness nutrition question (FAISS RAG trigger - Gap 2)
        "أكل إيه قبل التمرين عشان يديني طاقة؟",
        # 4. Food substitution (Gap 2)
        "What can I substitute for cheese in my diet?",
        # 5. Onboarding profile
        "أنا ولد اتولدت 2004-05-10، وزني 90 وطولي 175، بتمرن 4 أيام وعايز أخس"
    ]

    for i, q in enumerate(test_queries, 1):
        safe_print(f"\n--- Query {i} ---")
        safe_print(f"User: {q}")
        res = session.process(q)
        safe_print(f"Status : {res['status']}")
        safe_print(f"Intent : {res['parsed']['intent']}")
        if res['parsed'].get('meal_type'):
            safe_print(f"MealType: {res['parsed']['meal_type']}")
        if res['parsed'].get('target_calories'):
            safe_print(f"TargetCal: {res['parsed']['target_calories']}")
        if res['parsed'].get('requested_substitution'):
            safe_print(f"SubFood : {res['parsed']['requested_substitution']}")
        if res['parsed']['foods']:
            safe_print(f"Foods  : {json.dumps(res['parsed']['foods'], ensure_ascii=False, indent=2)}")
        if res['prompt_for_user']:
            safe_print(f"Bot    : {res['prompt_for_user']}")