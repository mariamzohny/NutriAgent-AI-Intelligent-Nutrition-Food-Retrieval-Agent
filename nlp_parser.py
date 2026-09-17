"""
nlp_parser.py — Task 1: NLP User Understanding Module

Responsibilities:
- Multilingual intent classification (Arabic / English / code-switched)
- Food entity extraction (NER) with USDA-compatible English names
- User profile extraction & deterministic sanity checks
- Selective onboarding state machine (NLPSession) for Tasks 3 & 4

Model: Groq — qwen/qwen3.8-27b (strict JSON schema mode)

Public API:
    parse_user_input(msg)                 → full NLPOutput dict
    parse_with_context(msg, history)      → same + conversation context
    get_food_data(msg)                    → intent + foods (for Task 2/4)
    get_user_profile(msg)                 → 10-field profile (for Task 3)
    NLPSession()                          → stateful multi-turn manager

Environment:
    GROQ_API_KEY must be set (via .env file or shell environment).
"""
import json
import os
import re
from datetime import datetime
from typing import List, Optional, Literal, Tuple

from pydantic import BaseModel, ConfigDict, Field
from groq import Groq
from dotenv import load_dotenv

load_dotenv() 


# ─── Groq API Client ────────────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY environment variable is not set. "
        "Get a key from https://console.groq.com/keys and set it via "
        "`export GROQ_API_KEY=...` — never hardcode it in this file."
    )

# qwen/qwen3.8-27b: stronger Arabic/multilingual coverage — good default here.
# openai/gpt-oss-120b: larger, supports strict schema mode too — worth A/B testing.
GROQ_MODEL = "qwen/qwen3.8-27b"

groq_client = Groq(api_key=GROQ_API_KEY)


# ─── 0.1 Intent Taxonomy (Extended) ────────────────────────────────────────────────────────
INTENT_TYPES = Literal[
    "calculate_calories",           # log eaten meal
    "calculate_macros",             # macro breakdown request
    "food_nutrition",               # nutrition query about a single food
    "meal_recommendation",          # single meal suggestion with constraints
    "daily_plan",                   # full day/multi-day plan
    "food_substitution",            # substitutes for X
    "fitness_nutrition_question",   # workout/sports nutrition (RAG trigger)
    "update_profile",               # profile metrics & onboarding
    "general_chat",                 # greetings / chit-chat
]


# ─── 1. Output Schema (Pydantic) ────────────────────────────────────────────────────────
# NOTE: All Pydantic models below use extra="forbid" — required for Groq's strict JSON-schema mode.

class FoodItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    food_name_original: str = Field(..., description="Original food name as written by the user")
    food_name_en: str       = Field(..., description="Standard English USDA-compatible food name")
    quantity: float          = Field(..., description="Numeric quantity e.g. 0.5, 1.0, 200.0")
    unit: str                = Field(..., description="Unit: gram, cup, piece, slice, can, bottle, tablespoon, bowl, loaf, milliliter")


class UserProfileExtract(BaseModel):
    model_config = ConfigDict(extra="forbid")

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
    model_config = ConfigDict(extra="forbid")

    intent: INTENT_TYPES = Field(..., description="User intent according to project specifications")
    language_detected: Literal["arabic", "english", "mixed"] = Field(
        ..., description="Detected input language"
    )
    raw_text: str = Field(..., description="Original user input, echoed verbatim")

    # ── Extended fields for meal-planning / recommendation ──
    meal_type: Optional[Literal["breakfast", "lunch", "dinner", "snack", "all_day"]] = Field(
        ..., description="Meal requested: breakfast, lunch, dinner, snack, all_day, or null"
    )
    target_calories: Optional[float] = Field(
        ..., description="Explicit target calories (e.g. 400.0 for '400-calorie breakfast') or null"
    )
    requested_substitution: Optional[str] = Field(
        ..., description="Food to substitute when intent is food_substitution (e.g. 'eggs') or null"
    )

    foods: List[FoodItem] = Field(..., description="Extracted food items ([] if none)")

    # No `default=None` here — Groq's strict mode requires every field to be `required` in the schema.
    # "Optional" is expressed by the TYPE being nullable (Optional[...]), not by a Python-side default value.
    profile_data: Optional[UserProfileExtract] = Field(
        ..., description="Full profile object whenever ANY profile info is present, otherwise null"
    )

# ─── 2. Language Detection (Deterministic — no LLM) ────────────────────────────────────────────────────────
def detect_language(text: str) -> Literal["arabic", "english", "mixed"]:
    """
    Classify input language by Unicode character counts (Arabic U+0600–U+06FF).

    Uses a RATIO-based check so long messages with a few borrowed/foreign
    words (e.g. a long Arabic paragraph containing one English brand name)
    aren't misclassified as "mixed".

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
    # and the symmetric case (a couple of Arabic words dropped into an
    # otherwise English sentence) shouldn't flip the whole message to "mixed".
    if latin_chars <= 2 and arabic_chars >= 6:
        return "arabic"
    if arabic_chars <= 2 and latin_chars >= 6:
        return "english"

    arabic_ratio = arabic_chars / total
    if arabic_ratio >= 0.85:
        return "arabic"
    if arabic_ratio <= 0.15:
        return "english"

    return "mixed"


# ─── 2.5 Arabic Text Normalization (Deterministic) ────────────────────────────────────────────────────────
_ARABIC_DIACRITICS_RE = re.compile(r'[\u064B-\u065F\u0670\u06D6-\u06ED]')
_ARABIC_ALEF_RE       = re.compile(r'[إأآٱ]')

def normalize_arabic_text(text: str) -> str:
    """
    Deterministic Arabic normalization applied to the WORKING COPY sent
    to the LLM for parsing. Does NOT touch `raw_text` in the final output
    — raw_text must stay a verbatim copy of what the user typed (Rule #2).

    Mirrors the rules used in Task 2's `_normalize_lookup_key`, so both
    modules treat Arabic text the same way and don't drift apart:
    - Strips tashkeel (diacritics).
    - Strips tatweel (ـ).
    - Normalizes أ / إ / آ / ٱ → ا
    - Normalizes ى → ي
    - Collapses repeated whitespace.
    """
    if not text:
        return text

    normalized = _ARABIC_DIACRITICS_RE.sub('', text)
    normalized = normalized.replace("ـ", "")
    normalized = _ARABIC_ALEF_RE.sub('ا', normalized)
    normalized = normalized.replace('ى', 'ي')
    normalized = re.sub(r'\s+', ' ', normalized).strip()

    return normalized


# ─── 3. Unit Normalization ────────────────────────────────────────────────────────
UNIT_ALIASES: dict = {
    # English plurals / abbreviations
    "kg": "kilogram", "kilograms": "kilogram", "kgs": "kilogram",
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
    "أنثى", "انثى", "بنت", "امرأة", "فتاة", "ست",
    "أنا بنت", "أنا أنثى",
    # English
    "female", "woman", "girl",
]

_GENDER_MALE_KEYWORDS: List[str] = [
    # Arabic
    "ذكر", "ولد", "راجل", "رجل", "صبي", "أنا ذكر", "أنا ولد",
    # English whole words (checked with surrounding whitespace)
    "male", "man", "boy",
]


def _extract_gender_from_text(text: str) -> Optional[str]:
    """
    Deterministic gender extraction from Arabic/English keywords.

    Uses TOKEN matching only — no substring matching — so "مولود" does NOT
    match "ولد" and "بنطلون" does NOT match "بنت".
    """
    cleaned = re.sub(r'[^a-zA-Z\u0600-\u06FF\s]', ' ', text.lower())
    tokens = set(cleaned.split())
    padded = f" {cleaned} "

    for kw in _GENDER_FEMALE_KEYWORDS:
        kw_norm = kw.strip().lower()
        if " " in kw_norm:
            if kw_norm in padded:
                return "female"
        elif kw_norm in tokens:
            return "female"

    for kw in _GENDER_MALE_KEYWORDS:
        kw_norm = kw.strip().lower()
        if " " in kw_norm:
            if kw_norm in padded:
                return "male"
        elif kw_norm in tokens:
            return "male"

    return None


# ─── 3.5 Sanity Checks (Hallucination Protection) ────────────────────────────────────────────────────────
def sanitize_extracted_values(parsed: dict) -> dict:
    """
    Sanity checks for biometric & metric values:
    - Height: fix mm (1750 → 175), clamp 50–260 cm
    - Weight: clamp 20–350 kg
    - Target calories: clamp 50–15000 kcal
    Anything out of range is set to None.
    """
    profile = parsed.get("profile_data")
    if profile:
        if profile.get("height") is not None:
            try:
                h = float(profile["height"])
                if h >= 500:
                    h = h / 10.0
                profile["height"] = round(h, 1) if 50.0 <= h <= 260.0 else None
            except (ValueError, TypeError):
                profile["height"] = None

        if profile.get("weight") is not None:
            try:
                w = float(profile["weight"])
                profile["weight"] = round(w, 1) if 20.0 <= w <= 350.0 else None
            except (ValueError, TypeError):
                profile["weight"] = None

    if parsed.get("target_calories") is not None:
        try:
            tc = float(parsed["target_calories"])
            parsed["target_calories"] = round(tc, 1) if 50.0 <= tc <= 15000.0 else None
        except (ValueError, TypeError):
            parsed["target_calories"] = None

    return parsed


NON_FOOD_STOP_WORDS = {
    "before workout", "after workout", "without eggs", "breakfast meal", "lunch meal",
    "dinner meal", "snack", "item", "meal", "substitute", "food", "diet",
    "وجبة فطور", "وجبة غداء", "وجبة عشاء", "بدون بيض", "قبل التمرين", "بعد التمرين",
}

EMPTY_FOODS_INTENTS = {
    "meal_recommendation",
    "fitness_nutrition_question",
    "daily_plan",
    "food_substitution",
    "update_profile",
    "general_chat",
}


def normalize_output(parsed: dict) -> dict:
    """
    Post-process LLM output:
    - Force foods=[] for non-logging intents.
    - Remove non-food artifacts and duplicates.
    - Normalize units + lowercase food_name_en.
    - Fix food_name_original if it hallucinated Arabic on English input.
    - Deterministic gender fallback.
    - Sanity checks on biometric & metric values.
    """
    intent = parsed.get("intent")
    raw_text = parsed.get("raw_text", "")
    lang = detect_language(raw_text)

    # ── Foods: force empty for non-logging intents ──────────────────────────
    if intent in EMPTY_FOODS_INTENTS:
        parsed["foods"] = []
    else:
        seen: set = set()
        clean_foods: list = []
        for food in parsed.get("foods", []):
            name_en = food.get("food_name_en", "").strip().lower()
            name_orig = food.get("food_name_original", "").strip()

            # Filter non-food artifacts
            if name_en in NON_FOOD_STOP_WORDS or name_orig.lower() in NON_FOOD_STOP_WORDS:
                continue

            # If input is English, don't keep an Arabic hallucination in original
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

    # ── Deterministic gender fallback ───────────────────────────────────────
    profile = parsed.get("profile_data")
    if profile and profile.get("gender") is None:
        detected_gender = _extract_gender_from_text(raw_text)
        if detected_gender:
            profile["gender"] = detected_gender

    # ── Sanity checks ──────────────────────────────────────────────────────
    return sanitize_extracted_values(parsed)


# ─── 4. System Prompt & Few-Shot Examples ────────────────────────────────────────────────────────
def get_system_prompt() -> str:
    """
    Build the system prompt dynamically, injecting the current date & year so
    the LLM can convert "I'm 22 years old" into a plausible birth_date.
    """
    current_date = datetime.now().strftime("%Y-%m-%d")
    current_year = datetime.now().year

    return f"""You are a precise multilingual NLP parser for a fitness calorie-tracking and nutrition assistant.
You understand Arabic (including Egyptian slang), English, and Arabic-English mixed (code-switched) text.

CURRENT RUNTIME DATE: {current_date} (Current Year: {current_year})

INTENT TAXONOMY (choose exactly one):
1. calculate_calories         → user logs/records food they ate or will eat (e.g. "كلت 200g فراخ", "I had 2 eggs and toast")
2. calculate_macros           → user asks for macro breakdown (protein/carbs/fats) of their meal
3. food_nutrition             → user asks calories/macros of a specific item ("How many calories in 100g avocado?", "كم سعرة في الموز؟")
4. meal_recommendation        → user asks for a SINGLE meal suggestion with constraints ("Make me a 400-calorie breakfast without eggs", "عايز فطار 400 سعرة بدون بيض")
5. daily_plan                 → user asks for a FULL day or multi-day meal plan ("خطة أكل ليوم كامل", "give me a full day meal plan")
6. food_substitution          → user asks what to substitute for a specific food ("what can I substitute for cheese?", "بديل البيض في الدايت")
7. fitness_nutrition_question → user asks fitness/exercise nutrition advice ("what should I eat before workout?", "أكل إيه قبل التمرين؟")
8. update_profile             → user provides personal metrics (age, DOB, weight, height, goal, allergies, activity level)
9. general_chat               → greetings, thank you, off-topic, chit-chat

OUTPUT RULES:
1. Always return valid JSON matching the schema exactly.
2. raw_text: copy the user's exact input verbatim.
3. language_detected: "arabic", "english", or "mixed".
4. meal_type: "breakfast" | "lunch" | "dinner" | "snack" | "all_day" | null
5. target_calories: extract numeric calories if user specified a budget (e.g. 400.0 for "400-calorie breakfast").
6. requested_substitution: extract the food being substituted when intent is food_substitution (e.g. "eggs").
7. foods: list of food items. MUST be [] for update_profile, fitness_nutrition_question, general_chat, daily_plan, meal_recommendation, and food_substitution.
8. profile_data: return the FULL object whenever the message contains ANY profile information —
   goal, restriction, allergy, preferred food, weight, height, birth_date, gender, or activity level —
   REGARDLESS of which intent you picked. Only return null when the message truly contains
   ZERO profile information.
   Example: "Suggest a meal without eggs" → intent = meal_recommendation,
   but MUST still return profile_data with forbidden_foods = ["eggs"].

RULES FOR profile_data (whenever ANY profile info is present):
Fill EVERY field — do NOT omit any key. Fields not mentioned in THIS message stay null / []:
- birth_date:           "YYYY-MM-DD" or null. If user gives age (e.g. 22 years), compute as "{current_year - 22}-01-01".
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

CODE-SWITCHING HANDLING:
The user may freely mix Arabic and English within the same sentence, or even
within the same phrase (e.g. "grilled دجاج", "كوب rice", "عايز muscle gain").
- Extract every entity regardless of which language it appears in.
- Always normalize food_name_en to English, even when food_name_original is Arabic/mixed.
- A single food item's name may itself be mixed — do NOT create two separate
  food entries for the same physical food item.
"""


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
        "intent": "calculate_calories", "language_detected": "arabic",
        "raw_text": "كلت 200g فراخ مشوية مع طبق رز",
        "meal_type": "lunch", "target_calories": None, "requested_substitution": None,
        "foods": [
            {"food_name_original": "فراخ مشوية", "food_name_en": "grilled chicken breast", "quantity": 200.0, "unit": "gram"},
            {"food_name_original": "رز",          "food_name_en": "cooked white rice",       "quantity": 1.0,   "unit": "bowl"}
        ],
        "profile_data": None
    }, ensure_ascii=False)},

    # ── 5. English meal log ────────────────────────────────────────────────────
    {"role": "user", "content": "I had 2 boiled eggs and a cup of oats with milk for breakfast"},
    {"role": "assistant", "content": json.dumps({
        "intent": "calculate_calories", "language_detected": "english",
        "raw_text": "I had 2 boiled eggs and a cup of oats with milk for breakfast",
        "meal_type": "breakfast", "target_calories": None, "requested_substitution": None,
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
        "intent": "calculate_calories", "language_detected": "mixed",
        "raw_text": "أكلت grilled salmon مع 1 cup برية وسلطة",
        "meal_type": "lunch", "target_calories": None, "requested_substitution": None,
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
        "intent": "daily_plan", "language_detected": "arabic",
        "raw_text": "ممكن تعملي خطة أكل ليوم كامل؟",
        "meal_type": "all_day", "target_calories": None, "requested_substitution": None,
        "foods": [], "profile_data": None
    }, ensure_ascii=False)},

    # ── 8. Nutrition question ──────────────────────────────────────────────────
    {"role": "user", "content": "How many calories are in 100g of avocado?"},
    {"role": "assistant", "content": json.dumps({
        "intent": "food_nutrition", "language_detected": "english",
        "raw_text": "How many calories are in 100g of avocado?",
        "meal_type": None, "target_calories": None, "requested_substitution": None,
        "foods": [{"food_name_original": "avocado", "food_name_en": "avocado", "quantity": 100.0, "unit": "gram"}],
        "profile_data": None
    }, ensure_ascii=False)},

    # ── 9. General chat / greeting ─────────────────────────────────────────────
    {"role": "user", "content": "أهلاً، إيه اللي تقدر تعمله؟"},
    {"role": "assistant", "content": json.dumps({
        "intent": "general_chat", "language_detected": "arabic",
        "raw_text": "أهلاً، إيه اللي تقدر تعمله؟",
        "meal_type": None, "target_calories": None, "requested_substitution": None,
        "foods": [], "profile_data": None
    }, ensure_ascii=False)},

    # ── 10. Restriction embedded inside a non-update_profile intent ────────────
    {"role": "user", "content": "Suggest a meal without eggs."},
    {"role": "assistant", "content": json.dumps({
        "intent": "meal_recommendation", "language_detected": "english",
        "raw_text": "Suggest a meal without eggs.",
        "meal_type": None, "target_calories": None, "requested_substitution": None,
        "foods": [],
        "profile_data": {
            "birth_date": None, "gender": None,
            "weight": None, "height": None,
            "activity_level": None, "goal": None,
            "dietary_restrictions": [], "forbidden_foods": ["eggs"],
            "preferred_foods": [], "allergies": []
        }
    }, ensure_ascii=False)},

    # ── 11. Goal embedded inside a non-update_profile intent (Arabic) ──────────
    {"role": "user", "content": "عايز أخس، ممكن أكل إيه النهاردة؟"},
    {"role": "assistant", "content": json.dumps({
        "intent": "daily_plan", "language_detected": "arabic",
        "raw_text": "عايز أخس، ممكن أكل إيه النهاردة؟",
        "meal_type": "all_day", "target_calories": None, "requested_substitution": None,
        "foods": [],
        "profile_data": {
            "birth_date": None, "gender": None,
            "weight": None, "height": None,
            "activity_level": None, "goal": "weight_loss",
            "dietary_restrictions": [], "forbidden_foods": [],
            "preferred_foods": [], "allergies": []
        }
    }, ensure_ascii=False)},

    # ── 12. Mixed food/quantity/unit within the same food item ─────────────────
    {"role": "user", "content": "كلت three eggs وكوب milk على الفطار"},
    {"role": "assistant", "content": json.dumps({
        "intent": "calculate_calories", "language_detected": "mixed",
        "raw_text": "كلت three eggs وكوب milk على الفطار",
        "meal_type": "breakfast", "target_calories": None, "requested_substitution": None,
        "foods": [
            {"food_name_original": "three eggs", "food_name_en": "egg",  "quantity": 3.0, "unit": "piece"},
            {"food_name_original": "كوب milk",     "food_name_en": "milk", "quantity": 1.0, "unit": "cup"}
        ],
        "profile_data": None
    }, ensure_ascii=False)},

    # ── 13. Mixed allergy/restriction extraction ────────────────────────────────
    {"role": "user", "content": "I want خطة أكل بس بدون dairy عشان عندي حساسية"},
    {"role": "assistant", "content": json.dumps({
        "intent": "daily_plan", "language_detected": "mixed",
        "raw_text": "I want خطة أكل بس بدون dairy عشان عندي حساسية",
        "meal_type": "all_day", "target_calories": None, "requested_substitution": None,
        "foods": [],
        "profile_data": {
            "birth_date": None, "gender": None,
            "weight": None, "height": None,
            "activity_level": None, "goal": None,
            "dietary_restrictions": [], "forbidden_foods": [],
            "preferred_foods": [], "allergies": ["dairy"]
        }
    }, ensure_ascii=False)},

    # ── 14. Meal recommendation (EN) with target calories + restriction ────
    {"role": "user", "content": "Make me a 400-calorie breakfast without eggs"},
    {"role": "assistant", "content": json.dumps({
        "intent": "meal_recommendation", "language_detected": "english",
        "raw_text": "Make me a 400-calorie breakfast without eggs",
        "meal_type": "breakfast", "target_calories": 400.0, "requested_substitution": None,
        "foods": [],
        "profile_data": {
            "birth_date": None, "gender": None, "weight": None, "height": None,
            "activity_level": None, "goal": None,
            "dietary_restrictions": [], "forbidden_foods": ["eggs"],
            "preferred_foods": [], "allergies": []
        }
    }, ensure_ascii=False)},

    # ── 15. Meal recommendation (AR) with target calories + restriction ────
    {"role": "user", "content": "عايز وجبة غداء في حدود 600 سعرة بدون سمك"},
    {"role": "assistant", "content": json.dumps({
        "intent": "meal_recommendation", "language_detected": "arabic",
        "raw_text": "عايز وجبة غداء في حدود 600 سعرة بدون سمك",
        "meal_type": "lunch", "target_calories": 600.0, "requested_substitution": None,
        "foods": [],
        "profile_data": {
            "birth_date": None, "gender": None, "weight": None, "height": None,
            "activity_level": None, "goal": None,
            "dietary_restrictions": [], "forbidden_foods": ["fish"],
            "preferred_foods": [], "allergies": []
        }
    }, ensure_ascii=False)},

    # ── 16. Food substitution (EN) ────────────────────────────────────────
    {"role": "user", "content": "What can I substitute for cheese in my diet?"},
    {"role": "assistant", "content": json.dumps({
        "intent": "food_substitution", "language_detected": "english",
        "raw_text": "What can I substitute for cheese in my diet?",
        "meal_type": None, "target_calories": None, "requested_substitution": "cheese",
        "foods": [], "profile_data": None
    }, ensure_ascii=False)},

    # ── 17. Fitness nutrition question (RAG trigger) ───────────────────────
    {"role": "user", "content": "أكل إيه قبل التمرين عشان يديني طاقة؟"},
    {"role": "assistant", "content": json.dumps({
        "intent": "fitness_nutrition_question", "language_detected": "arabic",
        "raw_text": "أكل إيه قبل التمرين عشان يديني طاقة؟",
        "meal_type": "snack", "target_calories": None, "requested_substitution": None,
        "foods": [], "profile_data": None
    }, ensure_ascii=False)},
]


# ─── 5. Onboarding: Mandatory Fields & Prompts ────────────────────────────────────────────────────────
# The 6 fields Task 3 MUST have before calculating BMR / TDEE
MANDATORY_FIELDS: List[str] = [
    "birth_date", "gender", "weight", "height", "activity_level", "goal"
]


# Only these intents require full onboarding before processing.
# Simple queries (food_nutrition, calculate_calories, etc.) are never blocked.
PROFILE_DEPENDENT_INTENTS = {"daily_plan", "update_profile"}

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


# ─── 6. Profile Validation & Merging ────────────────────────────────────────────────────────
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


# ─── 7. Core Parser Functions ────────────────────────────────────────────────────────
def _build_messages(user_message: str, history: Optional[List[dict]] = None) -> List[dict]:
    """Build the Groq message list with system prompt + few-shot + optional history."""
    normalized = normalize_arabic_text(user_message)
    messages = [
        {"role": "system", "content": get_system_prompt()},
        *FEW_SHOT_EXAMPLES,
    ]
    if history:
        messages.extend(history[-6:])  # last 3 exchanges
    messages.append({"role": "user", "content": normalized})
    return messages


def _groq_call(messages: List[dict]) -> dict:
    """Execute one Groq chat completion with strict JSON schema."""
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.0,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "nlp_output",
                "strict": True,
                "schema": NLPOutput.model_json_schema(),
            },
        },
    )
    return json.loads(response.choices[0].message.content)


def _safe_fallback(user_message: str) -> dict:
    """Never crash: return a safe general_chat when all retries fail."""
    return {
        "intent": "general_chat",
        "language_detected": detect_language(user_message),
        "raw_text": user_message,
        "meal_type": None,
        "target_calories": None,
        "requested_substitution": None,
        "foods": [],
        "profile_data": None,
    }


def parse_user_input(user_message: str) -> dict:
    """
    Parse a single user message into a structured NLPOutput dict.
    Retries once on failure; falls back to general_chat on double failure.
    """
    messages = _build_messages(user_message)

    for attempt in range(2):
        try:
            parsed = _groq_call(messages)
            parsed["language_detected"] = detect_language(user_message)
            parsed["raw_text"] = user_message
            return normalize_output(parsed)
        except Exception:
            if attempt == 1:
                return normalize_output(_safe_fallback(user_message))

    return normalize_output(_safe_fallback(user_message))


def parse_with_context(user_message: str, history: Optional[List[dict]] = None) -> dict:
    """
    Parse with optional conversation history for reference/pronoun resolution.
    Only the last 6 messages (3 exchanges) are included.
    Retries once on failure; falls back to general_chat on double failure.
    """
    messages = _build_messages(user_message, history=history)

    for attempt in range(2):
        try:
            parsed = _groq_call(messages)
            parsed["language_detected"] = detect_language(user_message)
            parsed["raw_text"] = user_message
            return normalize_output(parsed)
        except Exception:
            if attempt == 1:
                return normalize_output(_safe_fallback(user_message))

    return normalize_output(_safe_fallback(user_message))


# ─── 7.1 Dedicated Split Functions for Teammates ────────────────────────────────────────────────────────
def get_food_data(user_message: str) -> dict:
    """
    Extract the Meal/Food data along with message metadata (intent, language, raw_text).
    Perfect for Task 2 (USDA API search), Task 3 (food nutrition calculation), and Task 4.

    Example:
        >>> get_food_data("كلت 200g فراخ مشوية مع طبق رز")
        {
            "intent": "calculate_calories",
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
        "intent":                 parsed.get("intent"),
        "language_detected":      parsed.get("language_detected"),
        "raw_text":               parsed.get("raw_text", user_message),
        "meal_type":              parsed.get("meal_type"),
        "target_calories":        parsed.get("target_calories"),
        "requested_substitution": parsed.get("requested_substitution"),
        "foods":                  parsed.get("foods", []),
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


# ─── 8. NLPSession — Onboarding + State Manager ────────────────────────────────────────────────────────
class NLPSession:
    """
    Stateful session manager for a single user conversation.

    Selective Onboarding:
    - Simple queries (calculate_calories, food_nutrition, meal_recommendation,
      food_substitution, fitness_nutrition_question, general_chat) are NEVER
      blocked by missing profile fields.
    - Profile-dependent intents (daily_plan, update_profile) require all 6
      mandatory fields (birth_date, gender, weight, height, activity_level, goal)
      before they are processed by Task 3.

    Typical usage
    -------------
        session = NLPSession()

        while True:
            user_input = input("You: ")
            result = session.process(user_input)

            if result["status"] == "needs_profile":
                print("Bot:", result["prompt_for_user"])
            elif result["status"] == "onboarding_complete":
                print("Bot:", result["prompt_for_user"])
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
        self.history: List[dict] = []
        self.language: str = "english"

    # ------------------------------------------------------------------
    def process(self, user_message: str) -> dict:
        """
        Process one user turn.

        Selective onboarding:
        - Simple queries are NEVER blocked by missing profile fields.
        - Only profile-dependent intents (daily_plan, update_profile) trigger onboarding.
        """
        self.language = detect_language(user_message)
        parsed = parse_with_context(user_message, self.history)

        # Append to history
        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": json.dumps(parsed, ensure_ascii=False)})

        # Merge any extracted profile info
        if parsed.get("profile_data"):
            self.profile = merge_profile(self.profile, parsed["profile_data"], user_message)

        # Check mandatory fields
        is_complete, missing = validate_mandatory_profile_fields(self.profile)
        if is_complete:
            self.onboarding_complete = True

        intent = parsed.get("intent")

        # ── Case 1: profile-dependent intent, profile incomplete ───────────
        if intent in PROFILE_DEPENDENT_INTENTS and not self.onboarding_complete:
            prompt = get_onboarding_prompt(missing, self.language)
            return {
                "parsed":          parsed,
                "status":          "needs_profile",
                "prompt_for_user": prompt,
                "profile":         self.profile,
                "missing_fields":  missing,
            }

        # Attach the full accumulated profile so Task 3 always has it
        parsed["profile_data"] = self.profile

        # ── Case 2: just finished onboarding via update_profile ────────────
        if intent == "update_profile" and self.onboarding_complete:
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


# ─── 9. Quick Test — run with: python nlp_parser.py ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, io
    # Force UTF-8 output so Arabic characters print correctly on Windows
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer,
        encoding="utf-8",
        errors="replace",
        line_buffering=True,
        write_through=True,
    )

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

    test_queries = [
        # 1. Simple nutrition query — must NOT be blocked by onboarding
        "How many calories are in 200g chicken breast?",
        # 2. Meal recommendation with target cal + restriction
        "Make me a 400-calorie breakfast without eggs",
        # 3. Fitness nutrition question (RAG trigger)
        "أكل إيه قبل التمرين عشان يديني طاقة؟",
        # 4. Food substitution
        "What can I substitute for cheese in my diet?",
        # 5. Daily plan (should trigger onboarding)
        "ممكن تعملي خطة أكل ليوم كامل؟",
        # 6. Full profile (completes onboarding)
        "أنا ولد اتولدت 2004-05-10، وزني 90 وطولي 175، بتمرن 4 أيام وعايز أخس",
        # 7. Daily plan again — should now be ready
        "عايز خطة أكل ليوم كامل",
    ]

    for i, q in enumerate(test_queries, 1):
        safe_print(f"\n--- Query {i} ---")
        safe_print(f"User: {q}")
        res = session.process(q)
        safe_print(f"Status : {res['status']}")
        safe_print(f"Intent : {res['parsed']['intent']}")
        safe_print(f"Lang   : {res['parsed']['language_detected']}")
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
        if res['missing_fields']:
            safe_print(f"Missing: {res['missing_fields']}")
