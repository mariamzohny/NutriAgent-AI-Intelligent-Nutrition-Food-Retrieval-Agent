import json
import re
import ollama
from typing import List, Optional, Literal, Tuple
from pydantic import BaseModel, Field


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
    dietary_restrictions: List[str]                                                                        = Field(..., description="e.g. ['keto', 'vegan'] or []")
    forbidden_foods:      List[str]                                                                        = Field(..., description="Foods user avoids e.g. ['eggs'] or []")
    preferred_foods:      List[str]                                                                        = Field(..., description="Foods user prefers or []")
    allergies:            List[str]                                                                        = Field(..., description="Medical allergies e.g. ['dairy'] or []")


class NLPOutput(BaseModel):
    intent: Literal["log_meal", "create_meal_plan", "ask_nutrition", "update_profile", "general_chat"] = Field(
        ..., description="User intent"
    )
    language_detected: Literal["arabic", "english", "mixed"] = Field(
        ..., description="Detected input language"
    )
    raw_text: str = Field(..., description="Original user input, echoed verbatim")
    foods: List[FoodItem] = Field(..., description="Extracted food items ([] for update_profile / general_chat)")
    profile_data: Optional[UserProfileExtract] = Field(
        default=None,
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
# 3. Unit Normalization
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
    # Clean and pad text with spaces for whole-word matching
    cleaned = re.sub(r'[^a-zA-Z\u0600-\u06FF\s]', ' ', text.lower())
    padded = f" {cleaned} "
    
    for kw in _GENDER_FEMALE_KEYWORDS:
        if kw.strip() in padded.split() or kw in padded:
            return "female"
    for kw in _GENDER_MALE_KEYWORDS:
        if kw.strip() in padded.split() or kw in padded:
            return "male"
    return None


def normalize_output(parsed: dict) -> dict:
    """
    Post-process LLM output:
    - Normalize unit strings to canonical form.
    - Clamp non-positive quantities to 1.0.
    - Lowercase food_name_en for USDA search compatibility.
    - Remove duplicate food entries.
    - Deterministic gender fallback: if profile_data.gender is null,
      try to extract it from the raw text via keyword matching.
    """
    # ── Food normalization ────────────────────────────────────────────────────
    seen: set = set()
    clean_foods: list = []
    for food in parsed.get("foods", []):
        raw_unit = str(food.get("unit", "")).strip().lower()
        food["unit"] = UNIT_ALIASES.get(raw_unit, raw_unit)
        if food.get("quantity", 0) <= 0:
            food["quantity"] = 1.0
        food["food_name_en"] = food.get("food_name_en", "").strip().lower()
        key = (food["food_name_en"], food["quantity"], food["unit"])
        if key not in seen:
            seen.add(key)
            clean_foods.append(food)
    parsed["foods"] = clean_foods

    # ── Deterministic gender fallback ─────────────────────────────────────────
    profile = parsed.get("profile_data")
    if profile and profile.get("gender") is None:
        raw = parsed.get("raw_text", "")
        detected_gender = _extract_gender_from_text(raw)
        if detected_gender:
            profile["gender"] = detected_gender

    return parsed


# ==========================================
# 4. System Prompt & Few-Shot Examples
# ==========================================

SYSTEM_PROMPT = """
You are a precise multilingual NLP parser for a fitness calorie-tracking assistant.
You understand Arabic, English, and Arabic-English mixed (code-switched) text.

INTENT DEFINITIONS:
- log_meal         → user is recording food they ate or will eat
- create_meal_plan → user wants a full day/week meal plan suggestion
- ask_nutrition     → user is asking about calories/macros of a specific food
- update_profile   → user is providing personal info (age, weight, height, goal, restrictions, etc.)
- general_chat     → greetings, off-topic, unclear intent

OUTPUT RULES:
1. Always return valid JSON matching the schema exactly.
2. raw_text: copy the user's exact input.
3. language_detected: set to "arabic", "english", or "mixed".
4. foods: list of food items. MUST be [] when intent is update_profile or general_chat.
5. profile_data: full object when intent is update_profile, otherwise MUST be null.

RULES FOR profile_data (when intent = update_profile):
Fill EVERY field — do NOT omit any key:
- birth_date:           "YYYY-MM-DD" or null
- gender:               "male" | "female" | null
- weight:               number in kg | null
- height:               number in cm | null
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

AGE HINT: If user gives age instead of birth_date, estimate birth_date by subtracting from current year.
"""

# fmt: off
FEW_SHOT_EXAMPLES: List[dict] = [

    # ── 1. Arabic full profile ─────────────────────────────────────────────────
    {"role": "user", "content": "أنا ولد اتولدت 2004-05-10، وزني 90 وطولي 175، بتمرن 4 أيام في الأسبوع وعايز أخس، بس مش باكل بيض وعندي حساسية ألبان"},
    {"role": "assistant", "content": json.dumps({
        "intent": "update_profile", "language_detected": "arabic",
        "raw_text": "أنا ولد اتولدت 2004-05-10، وزني 90 وطولي 175، بتمرن 4 أيام في الأسبوع وعايز أخس، بس مش باكل بيض وعندي حساسية ألبان",
        "foods": [],
        "profile_data": {
            "birth_date": "2004-05-10", "gender": "male",
            "weight": 90.0, "height": 175.0,
            "activity_level": "moderate", "goal": "weight_loss",
            "dietary_restrictions": [], "forbidden_foods": ["eggs"],
            "preferred_foods": [], "allergies": ["dairy"]
        }
    }, ensure_ascii=False)},

    # ── 2. English full profile ────────────────────────────────────────────────
    {"role": "user", "content": "I'm a female born on 1999-03-15, weight 65kg height 160cm, gym 3 times a week, goal is weight loss, I'm vegan and allergic to peanuts"},
    {"role": "assistant", "content": json.dumps({
        "intent": "update_profile", "language_detected": "english",
        "raw_text": "I'm a female born on 1999-03-15, weight 65kg height 160cm, gym 3 times a week, goal is weight loss, I'm vegan and allergic to peanuts",
        "foods": [],
        "profile_data": {
            "birth_date": "1999-03-15", "gender": "female",
            "weight": 65.0, "height": 160.0,
            "activity_level": "moderate", "goal": "weight_loss",
            "dietary_restrictions": ["vegan"], "forbidden_foods": [],
            "preferred_foods": [], "allergies": ["peanuts"]
        }
    }, ensure_ascii=False)},

    # ── 3. Code-switched profile ───────────────────────────────────────────────
    {"role": "user", "content": "أنا female اتولدت 2001-08-20 وزني 55 kg وطولي 162 cm، بعمل light exercise وعايزة muscle gain"},
    {"role": "assistant", "content": json.dumps({
        "intent": "update_profile", "language_detected": "mixed",
        "raw_text": "أنا female اتولدت 2001-08-20 وزني 55 kg وطولي 162 cm، بعمل light exercise وعايزة muscle gain",
        "foods": [],
        "profile_data": {
            "birth_date": "2001-08-20", "gender": "female",
            "weight": 55.0, "height": 162.0,
            "activity_level": "light", "goal": "muscle_gain",
            "dietary_restrictions": [], "forbidden_foods": [],
            "preferred_foods": [], "allergies": []
        }
    }, ensure_ascii=False)},

    # ── 4. Arabic meal log ─────────────────────────────────────────────────────
    {"role": "user", "content": "كلت 200g فراخ مشوية مع طبق رز"},
    {"role": "assistant", "content": json.dumps({
        "intent": "log_meal", "language_detected": "arabic",
        "raw_text": "كلت 200g فراخ مشوية مع طبق رز",
        "foods": [
            {"food_name_original": "فراخ مشوية", "food_name_en": "grilled chicken breast", "quantity": 200.0, "unit": "gram"},
            {"food_name_original": "رز",          "food_name_en": "cooked white rice",       "quantity": 1.0,   "unit": "bowl"}
        ],
        "profile_data": None
    }, ensure_ascii=False)},

    # ── 5. English meal log ────────────────────────────────────────────────────
    {"role": "user", "content": "I had 2 boiled eggs and a cup of oats with milk for breakfast"},
    {"role": "assistant", "content": json.dumps({
        "intent": "log_meal", "language_detected": "english",
        "raw_text": "I had 2 boiled eggs and a cup of oats with milk for breakfast",
        "foods": [
            {"food_name_original": "boiled eggs", "food_name_en": "boiled egg",  "quantity": 2.0, "unit": "piece"},
            {"food_name_original": "oats",         "food_name_en": "rolled oats", "quantity": 1.0, "unit": "cup"},
            {"food_name_original": "milk",          "food_name_en": "whole milk",  "quantity": 1.0, "unit": "cup"}
        ],
        "profile_data": None
    }, ensure_ascii=False)},

    # ── 6. Code-switched meal log ──────────────────────────────────────────────
    {"role": "user", "content": "أكلت grilled salmon مع 1 cup برية وسلطة"},
    {"role": "assistant", "content": json.dumps({
        "intent": "log_meal", "language_detected": "mixed",
        "raw_text": "أكلت grilled salmon مع 1 cup برية وسلطة",
        "foods": [
            {"food_name_original": "grilled salmon", "food_name_en": "grilled salmon", "quantity": 1.0, "unit": "piece"},
            {"food_name_original": "برية",            "food_name_en": "brown rice",     "quantity": 1.0, "unit": "cup"},
            {"food_name_original": "سلطة",            "food_name_en": "green salad",    "quantity": 1.0, "unit": "bowl"}
        ],
        "profile_data": None
    }, ensure_ascii=False)},

    # ── 7. Meal plan request ───────────────────────────────────────────────────
    {"role": "user", "content": "ممكن تعملي خطة أكل ليوم كامل؟"},
    {"role": "assistant", "content": json.dumps({
        "intent": "create_meal_plan", "language_detected": "arabic",
        "raw_text": "ممكن تعملي خطة أكل ليوم كامل؟",
        "foods": [], "profile_data": None
    }, ensure_ascii=False)},

    # ── 8. Nutrition question ──────────────────────────────────────────────────
    {"role": "user", "content": "How many calories are in 100g of avocado?"},
    {"role": "assistant", "content": json.dumps({
        "intent": "ask_nutrition", "language_detected": "english",
        "raw_text": "How many calories are in 100g of avocado?",
        "foods": [{"food_name_original": "avocado", "food_name_en": "avocado", "quantity": 100.0, "unit": "gram"}],
        "profile_data": None
    }, ensure_ascii=False)},

    # ── 9. General chat / greeting ─────────────────────────────────────────────
    {"role": "user", "content": "أهلاً، إيه اللي تقدر تعمله؟"},
    {"role": "assistant", "content": json.dumps({
        "intent": "general_chat", "language_detected": "arabic",
        "raw_text": "أهلاً، إيه اللي تقدر تعمله؟",
        "foods": [], "profile_data": None
    }, ensure_ascii=False)},
]
# fmt: on


# ==========================================
# 5. Onboarding: Mandatory Fields & Prompts
# ==========================================

# The 6 fields Task 3 MUST have before calculating BMR / TDEE
MANDATORY_FIELDS: List[str] = [
    "birth_date", "gender", "weight", "height", "activity_level", "goal"
]

# Bilingual follow-up prompts for each missing field
_FIELD_PROMPTS: dict = {
    "birth_date": {
        "ar": "📅 ما هو تاريخ ميلادك؟ (بصيغة YYYY-MM-DD مثلاً: 2000-03-15)",
        "en": "📅 What is your date of birth? (format: YYYY-MM-DD, e.g. 2000-03-15)"
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
    """
    Build a bilingual conversational prompt asking for missing mandatory fields.

    Args:
        missing_fields: list of field names that are still None.
        lang: "arabic" | "english" | "mixed"  (from detect_language)

    Returns:
        A formatted string to show the user.
    """
    key = "ar" if lang == "arabic" else "en"
    if lang == "arabic":
        header = "مرحباً! 👋 قبل ما نبدأ، محتاج بعض المعلومات الأساسية عشان أحسب السعرات الحرارية بدقة:\n\n"
    else:
        header = "👋 Welcome! Before we start, I need a few details to calculate your calories accurately:\n\n"

    lines = [_FIELD_PROMPTS[f][key] for f in missing_fields if f in _FIELD_PROMPTS]
    return header + "\n\n".join(lines)


# ==========================================
# 6. Profile Validation & Merging
# ==========================================

def validate_mandatory_profile_fields(profile: dict) -> Tuple[bool, List[str]]:
    """
    Check whether all 6 mandatory Task-3 fields are non-null.

    Returns:
        (is_complete: bool, missing_fields: List[str])
    """
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
# 7. Core Parser Functions
# ==========================================

def parse_user_input(user_message: str) -> dict:
    """
    Parse a single user message into a structured NLPOutput dict.

    - Uses Ollama (qwen2.5:3b) with structured JSON output.
    - Overrides language_detected deterministically via detect_language().
    - Runs normalize_output() post-processing.

    Returns:
        dict matching NLPOutput schema.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *FEW_SHOT_EXAMPLES,
        {"role": "user", "content": user_message}
    ]

    response = ollama.chat(
        model="qwen2.5:3b",
        messages=messages,
        format=NLPOutput.model_json_schema(),
        options={"temperature": 0.0}
    )

    parsed: dict = json.loads(response["message"]["content"])

    # Deterministic overrides — never trust the LLM for these
    parsed["language_detected"] = detect_language(user_message)
    parsed["raw_text"] = user_message

    return normalize_output(parsed)


def parse_with_context(user_message: str, history: Optional[List[dict]] = None) -> dict:
    """
    Parse with optional conversation history for reference/pronoun resolution.

    Args:
        user_message: The new user utterance.
        history: List of {"role": "user"/"assistant", "content": str} dicts.
                 Only the last 3 full exchanges (6 messages) are included.

    Returns:
        dict matching NLPOutput schema.
    """
    history_turns = (history or [])[-6:]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *FEW_SHOT_EXAMPLES,
        *history_turns,
        {"role": "user", "content": user_message}
    ]

    response = ollama.chat(
        model="qwen2.5:3b",
        messages=messages,
        format=NLPOutput.model_json_schema(),
        options={"temperature": 0.0}
    )

    parsed: dict = json.loads(response["message"]["content"])
    parsed["language_detected"] = detect_language(user_message)
    parsed["raw_text"] = user_message

    return normalize_output(parsed)


# ==========================================
# 7.1 Dedicated Split Functions for Teammates
# ==========================================

def get_food_data(user_message: str) -> dict:
    """
    Extract the Meal/Food data along with message metadata (intent, language, raw_text).
    Perfect for Task 2 (USDA API search), Task 3 (food nutrition calculation), and Task 4.

    Example:
        >>> get_food_data("كلت 200g فراخ مشوية مع طبق رز")
        {
            "intent": "log_meal",
            "language_detected": "arabic",
            "raw_text": "كلت 200g فراخ مشوية مع طبق رز",
            "foods": [
                {"food_name_original": "فراخ مشوية", "food_name_en": "grilled chicken breast", "quantity": 200.0, "unit": "gram"},
                {"food_name_original": "طبق رز", "food_name_en": "cooked white rice", "quantity": 1.0, "unit": "bowl"}
            ]
        }
    """
    parsed = parse_user_input(user_message)
    return {
        "intent":            parsed.get("intent"),
        "language_detected": parsed.get("language_detected"),
        "raw_text":          parsed.get("raw_text", user_message),
        "foods":             parsed.get("foods", [])
    }


def get_user_profile(user_message: str) -> Optional[dict]:
    """
    Extract ONLY the 10-field User Profile from user message.
    Perfect for Task 3 (BMR, TDEE, target calories engine).

    Example:
        >>> get_user_profile("أنا ولد اتولدت 2004-05-10، وزني 90 وطولي 175، بتمرن 4 أيام وعايز أخس، مش باكل بيض")
        {
            "birth_date": "2004-05-10",
            "gender": "male",
            "weight": 90.0,
            "height": 175.0,
            "activity_level": "moderate",
            "goal": "weight_loss",
            "dietary_restrictions": [],
            "forbidden_foods": ["eggs"],
            "preferred_foods": [],
            "allergies": []
        }
    """
    parsed = parse_user_input(user_message)
    return parsed.get("profile_data")


# ==========================================
# 8. NLPSession — Onboarding + State Manager
# ==========================================

class NLPSession:
    """
    Stateful session manager for a single user conversation.

    Enforces onboarding: the user MUST supply all 6 mandatory profile fields
    (birth_date, gender, weight, height, activity_level, goal) before any
    other intents (log_meal, create_meal_plan, etc.) are processed by Task 3/4.

    Typical usage
    -------------
        session = NLPSession()

        while True:
            user_input = input("You: ")
            result = session.process(user_input)

            if result["status"] == "needs_profile":
                print("Bot:", result["prompt_for_user"])   # ask for missing fields
            elif result["status"] == "onboarding_complete":
                print("Bot:", result["prompt_for_user"])   # confirmation message
                # pass result["profile"] to Task 3 to calculate BMR/TDEE
            else:  # "ready"
                # pass result["parsed"] to Task 4 agent
                ...
    """

    def __init__(self):
        # Stored user profile — starts fully null / empty
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
        self.history: List[dict] = []        # raw conversation history
        self.language: str = "english"       # updated on each turn

    # ------------------------------------------------------------------
    def process(self, user_message: str) -> dict:
        """
        Process one user turn.

        Returns a dict with keys:
        ┌─────────────────┬──────────────────────────────────────────────────────┐
        │ key             │ description                                          │
        ├─────────────────┼──────────────────────────────────────────────────────┤
        │ parsed          │ Full NLPOutput dict from the parser                  │
        │ status          │ "needs_profile" | "onboarding_complete" | "ready"    │
        │ prompt_for_user │ String to display to user (or None when ready)       │
        │ profile         │ Current accumulated profile dict                     │
        │ missing_fields  │ List of still-missing mandatory field names (or [])  │
        └─────────────────┴──────────────────────────────────────────────────────┘
        """
        self.language = detect_language(user_message)

        # Parse with conversation history for context awareness
        parsed = parse_with_context(user_message, self.history)

        # Append to history
        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": json.dumps(parsed, ensure_ascii=False)})

        # Merge any extracted profile data
        if parsed.get("profile_data"):
            self.profile = merge_profile(self.profile, parsed["profile_data"], user_message)

        # Check mandatory fields
        is_complete, missing = validate_mandatory_profile_fields(self.profile)

        # ── Case 1: still missing mandatory fields ──────────────────────────
        if not is_complete:
            prompt = get_onboarding_prompt(missing, self.language)
            return {
                "parsed":          parsed,
                "status":          "needs_profile",
                "prompt_for_user": prompt,
                "profile":         self.profile,
                "missing_fields":  missing
            }

        # ── Attach complete user profile so Task 3 engine always has it ──
        parsed["profile_data"] = self.profile

        # ── Case 2: just completed onboarding this turn ────────────────────
        if not self.onboarding_complete:
            self.onboarding_complete = True
            if self.language == "arabic":
                confirm = (
                    "✅ ممتاز! استلمت بياناتك بنجاح.\n"
                    "دلوقتي تقدر:\n"
                    "  • تسجل وجباتك (مثلاً: 'كلت 200g فراخ مشوية')\n"
                    "  • تطلب خطة أكل ('ممكن خطة أكل ليوم؟')\n"
                    "  • تسأل عن سعرات أي أكلة ('كم سعرة في الأفوكادو؟')"
                )
            else:
                confirm = (
                    "✅ Profile complete! Here's what you can do now:\n"
                    "  • Log a meal (e.g. 'I had 200g grilled chicken')\n"
                    "  • Request a meal plan ('Give me a full day meal plan')\n"
                    "  • Ask about food calories ('How many calories in avocado?')"
                )
            return {
                "parsed":          parsed,
                "status":          "onboarding_complete",
                "prompt_for_user": confirm,
                "profile":         self.profile,
                "missing_fields":  []
            }

        # ── Case 3: normal flow — profile already complete ─────────────────
        return {
            "parsed":          parsed,
            "status":          "ready",
            "prompt_for_user": None,
            "profile":         self.profile,
            "missing_fields":  []
        }


# ==========================================
# 9. Quick Test — run with: python nlp_parser.py
# ==========================================

if __name__ == "__main__":
    import sys, io
    # Force UTF-8 output so Arabic characters print correctly on Windows
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    def safe_print(*args, **kwargs):
        try:
            print(*args, **kwargs)
        except UnicodeEncodeError:
            safe_str = " ".join(str(a).encode("ascii", "replace").decode() for a in args)
            print(safe_str)

    safe_print("=" * 60)
    safe_print("NLP Parser — Task 1 Quick Test")
    safe_print("=" * 60)

    session = NLPSession()

    test_turns = [
        # Turn 1: partial profile (missing height, activity, goal)
        "أنا ولد اتولدت 2004-05-10، وزني 90، مش باكل بيض",
        # Turn 2: fill in the rest
        "طولي 175، بتمرن 4 أيام في الأسبوع وعايز أخس",
        # Turn 3: log a meal (should now be in "ready" state)
        "كلت 200g فراخ مشوية مع طبق رز",
    ]

    for i, msg in enumerate(test_turns, 1):
        safe_print(f"\n--- Turn {i} ---")
        safe_print(f"User: {msg}")
        result = session.process(msg)
        safe_print(f"Status : {result['status']}")
        safe_print(f"Intent : {result['parsed']['intent']}")
        safe_print(f"Lang   : {result['parsed']['language_detected']}")
        if result["prompt_for_user"]:
            safe_print(f"Bot    : {result['prompt_for_user']}")
        if result["missing_fields"]:
            safe_print(f"Missing: {result['missing_fields']}")
        if result["parsed"]["foods"]:
            safe_print(f"Foods  : {json.dumps(result['parsed']['foods'], ensure_ascii=False, indent=2)}")
        safe_print(f"Profile: {json.dumps(result['profile'], ensure_ascii=False, indent=2)}")