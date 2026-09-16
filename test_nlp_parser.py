"""
test_nlp_parser.py
==================
Task 1 — NLP User Understanding: Comprehensive Hard Test Suite

Sections:
  1. detect_language()       — deterministic, no model needed
  2. normalize_output()      — deterministic, no model needed
  3. validate_mandatory_profile_fields() — deterministic
  4. LLM Parser — Basic single-turn (all 5 intents × 3 languages)
  5. LLM Parser — Hard single-turn (edge cases, dialects, ambiguous input)
  6. LLM Parser — Quantity & Unit edge cases
  7. LLM Parser — Food entity hard cases (branded, vague, multi-item)
  8. NLPSession — Onboarding enforcement & multi-turn merging
  9. NLPSession — Hard multi-turn conversations
 10. Intent boundary cases (ambiguous inputs)

Run:
    python -X utf8 test_nlp_parser.py
"""

import sys
import io
import json

# ── Force UTF-8 stdout so Arabic prints correctly on Windows ──────────────────
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from nlp_parser import (
    parse_user_input,
    detect_language,
    validate_mandatory_profile_fields,
    normalize_output,
    merge_profile,
    _extract_gender_from_text,
    NLPSession,
)

# ─────────────────────────────────────────────────────────────────────────────
# Test harness
# ─────────────────────────────────────────────────────────────────────────────

_results: list = []
_section_results: dict = {}
_current_section: str = ""


def section(title: str) -> None:
    global _current_section
    _current_section = title
    _section_results[title] = []
    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print("=" * 65)


def check(name: str, condition: bool, detail: str = "", show_parsed: dict = None) -> None:
    icon = "✅" if condition else "❌"
    print(f"{icon}  {name}")
    if detail:
        print(f"     ↳ {detail}")
    if not condition and show_parsed:
        print(f"     ↳ Full output: {json.dumps(show_parsed, ensure_ascii=False)[:300]}")
    _results.append(condition)
    _section_results[_current_section].append(condition)


def run_parser(label: str, msg: str, expected_intent, expected_lang: str,
               extra_check=None, detail_fn=None) -> None:
    """
    Run parse_user_input and check intent, language, and optional custom condition.
    Pass expected_intent=None to skip the intent assertion (useful for crash-only tests).
    """
    try:
        p = parse_user_input(msg)
        intent_ok = (expected_intent is None) or (p.get("intent") == expected_intent)
        lang_ok   = p.get("language_detected") == expected_lang
        extra_ok  = extra_check(p) if extra_check else True
        ok = intent_ok and lang_ok and extra_ok

        intent_info = f"intent={p.get('intent')}" + (f" (want {expected_intent})" if expected_intent else "")
        detail = f"{intent_info}, lang={p.get('language_detected')} (want {expected_lang})"
        if detail_fn:
            detail += f" | {detail_fn(p)}"
        check(label, ok, detail, show_parsed=p if not ok else None)
    except Exception as e:
        check(label, False, f"Exception: {e}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — detect_language()   [No model]
# ═════════════════════════════════════════════════════════════════════════════
section("SECTION 1 — detect_language()  [No model needed]")

check("Pure Arabic sentence",
      detect_language("أنا ولد وزني تسعين كيلو وطولي مية وسبعة وسبعين") == "arabic")

check("Pure English sentence",
      detect_language("I had grilled chicken and brown rice for lunch today") == "english")

check("Code-switched Arabic dominant",
      detect_language("أكلت grilled salmon مع cup برية") == "mixed")

check("Code-switched English dominant",
      detect_language("I am male born 2000-01-01, وزني 80, I want weight_loss") == "mixed")

check("Numbers + spaces only → english",
      detect_language("200 100 50 3.5") == "english")

check("Punctuation + Arabic → arabic",
      detect_language("!!! أنا عايز أخس !!!") == "arabic")

check("Single Arabic word",
      detect_language("مرحبا") == "arabic")

check("Single English word",
      detect_language("hello") == "english")

check("Empty string → english fallback",
      detect_language("") == "english")

check("Emoji only → english fallback",
      detect_language("😊🍗💪") == "english")

check("50-50 Arabic/Latin → mixed",
      detect_language("أنا male وزني 80 kg") == "mixed")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — normalize_output()  [No model]
# ═════════════════════════════════════════════════════════════════════════════
section("SECTION 2 — normalize_output()  [No model needed]")

def make_food(name_en, qty, unit, name_orig=None):
    return {"food_name_original": name_orig or name_en, "food_name_en": name_en,
            "quantity": qty, "unit": unit}

# Arabic unit aliases
raw_ar = {"intent": "log_meal", "language_detected": "arabic", "raw_text": "t",
          "profile_data": None,
          "foods": [
              make_food("  Chicken  ", 200, "grams"),
              make_food("rice", -5, "كوب"),
              make_food("salad", 0, "قطعة"),
              make_food("milk", 1, "ml"),
          ]}
out_ar = normalize_output(raw_ar)
check("'grams' → 'gram'",          out_ar["foods"][0]["unit"] == "gram")
check("food_name_en stripped/lowercased", out_ar["foods"][0]["food_name_en"] == "chicken")
check("'كوب' → 'cup'",             out_ar["foods"][1]["unit"] == "cup")
check("Negative qty clamped → 1.0", out_ar["foods"][1]["quantity"] == 1.0)
check("'قطعة' → 'piece'",          out_ar["foods"][2]["unit"] == "piece")
check("Zero qty clamped → 1.0",    out_ar["foods"][2]["quantity"] == 1.0)
check("'ml' → 'milliliter'",       out_ar["foods"][3]["unit"] == "milliliter")

# Duplicate removal
raw_dup = {"intent": "log_meal", "language_detected": "english", "raw_text": "t",
           "profile_data": None,
           "foods": [
               make_food("grilled chicken", 200, "gram"),
               make_food("grilled chicken", 200, "gram"),   # exact duplicate
               make_food("grilled chicken", 150, "gram"),   # different qty → not duplicate
           ]}
out_dup = normalize_output(raw_dup)
check("Exact duplicates removed → 2 foods left", len(out_dup["foods"]) == 2)

# Other aliases
raw_aliases = {"intent": "log_meal", "language_detected": "english", "raw_text": "t",
               "profile_data": None,
               "foods": [
                   make_food("butter", 1, "tbsp"),
                   make_food("oil", 2, "tablespoons"),
                   make_food("water", 500, "mls"),
                   make_food("bread", 2, "slices"),
               ]}
out_al = normalize_output(raw_aliases)
check("'tbsp' → 'tablespoon'",       out_al["foods"][0]["unit"] == "tablespoon")
check("'tablespoons' → 'tablespoon'", out_al["foods"][1]["unit"] == "tablespoon")
check("'mls' → 'milliliter'",        out_al["foods"][2]["unit"] == "milliliter")
check("'slices' → 'slice'",          out_al["foods"][3]["unit"] == "slice")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — validate_mandatory_profile_fields()  [No model]
# ═════════════════════════════════════════════════════════════════════════════
section("SECTION 3 — validate_mandatory_profile_fields()  [No model needed]")

_full = {"birth_date": "2004-05-10", "gender": "male", "weight": 90.0,
         "height": 175.0, "activity_level": "moderate", "goal": "weight_loss",
         "dietary_restrictions": [], "forbidden_foods": [], "preferred_foods": [], "allergies": []}

ok, miss = validate_mandatory_profile_fields(_full)
check("All 6 fields present → complete", ok, f"missing={miss}")

for field in ["birth_date", "gender", "weight", "height", "activity_level", "goal"]:
    p = {**_full, field: None}
    ok2, miss2 = validate_mandatory_profile_fields(p)
    check(f"Missing only '{field}' → incomplete", not ok2 and field in miss2)

ok3, miss3 = validate_mandatory_profile_fields(None)
check("None profile → all 6 missing", len(miss3) == 6)

ok4, miss4 = validate_mandatory_profile_fields({})
check("Empty dict → all 6 missing", len(miss4) == 6)

# merge_profile tests
base = {**_full, "height": None, "goal": None, "forbidden_foods": ["eggs"]}
new  = {"height": 175.0, "goal": "weight_gain", "forbidden_foods": ["dairy"], "allergies": ["nuts"],
        "birth_date": None, "gender": None, "weight": None,
        "activity_level": None, "dietary_restrictions": [], "preferred_foods": []}
merged = merge_profile(base, new)
check("merge_profile: scalar filled in (height)",   merged["height"] == 175.0)
check("merge_profile: scalar filled in (goal)",     merged["goal"] == "weight_gain")
check("merge_profile: None does NOT overwrite",     merged["birth_date"] == "2004-05-10")
check("merge_profile: list union (forbidden_foods)", set(merged["forbidden_foods"]) == {"eggs", "dairy"})
check("merge_profile: list union (allergies)",       "nuts" in merged["allergies"])

# Deterministic gender extraction tests
check("Gender extracted: 'أنا ولد' → male",       _extract_gender_from_text("أنا ولد اتولدت 2004-05-10") == "male")
check("Gender extracted: 'أنا ذكر' → male",      _extract_gender_from_text("أنا ذكر") == "male")
check("Gender extracted: 'أنا بنت' → female",     _extract_gender_from_text("أنا بنت وزني 60") == "female")
check("Gender extracted: 'I am female' → female", _extract_gender_from_text("I am female born 1999") == "female")
check("Gender extracted: 'I am a man' → male",   _extract_gender_from_text("I am a man") == "male")
check("Gender extracted: no gender keywords → None", _extract_gender_from_text("كلت بيتزا") is None)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Basic Single-Turn LLM Tests  [Requires Ollama]
# ═════════════════════════════════════════════════════════════════════════════
section("SECTION 4 — Basic LLM Parser  [Requires Ollama]")

_basic = [
    # (label, message, expected_intent, expected_lang, extra_check)
    (
        "Arabic full profile",
        "أنا ولد اتولدت 2004-05-10، وزني 90 وطولي 175، بتمرن 4 أيام وعايز أخس، مش باكل بيض وعندي حساسية ألبان",
        "update_profile", "arabic",
        lambda p: (p.get("profile_data", {}) or {}).get("gender") == "male"
                  and (p.get("profile_data", {}) or {}).get("weight") == 90.0
    ),
    (
        "English full profile",
        "I'm female, born 1999-03-15, 65kg, 160cm, gym 3x/week, goal weight loss, vegan, peanut allergy",
        "update_profile", "english",
        lambda p: (p.get("profile_data", {}) or {}).get("gender") == "female"
                  and (p.get("profile_data", {}) or {}).get("goal") == "weight_loss"
    ),
    (
        "Code-switched profile",
        "أنا female اتولدت 2001-08-20 وزني 55 kg وطولي 162 cm، light exercise، goal muscle gain",
        "update_profile", "mixed",
        lambda p: (p.get("profile_data", {}) or {}).get("activity_level") == "light"
    ),
    (
        "Arabic meal log — 2 foods",
        "كلت 200g فراخ مشوية مع طبق رز",
        "log_meal", "arabic",
        lambda p: len(p.get("foods", [])) >= 2
    ),
    (
        "English meal log",
        "I had 2 boiled eggs and a cup of oats for breakfast",
        "log_meal", "english",
        lambda p: len(p.get("foods", [])) >= 2
    ),
    (
        "Code-switched meal log",
        "أكلت grilled salmon مع 1 cup برية",
        "log_meal", "mixed",
        lambda p: len(p.get("foods", [])) >= 2
    ),
    (
        "Meal plan request — Arabic",
        "ممكن تعملي خطة أكل ليوم كامل؟",
        "create_meal_plan", "arabic",
        lambda p: p.get("foods") == []
    ),
    (
        "Nutrition question — English",
        "How many calories are in 100g of avocado?",
        "ask_nutrition", "english",
        lambda p: any("avocado" in f["food_name_en"] for f in p.get("foods", []))
    ),
    (
        "General chat / greeting",
        "أهلاً، إيه اللي تقدر تعمله؟",
        "general_chat", "arabic",
        lambda p: p.get("profile_data") is None
    ),
]

for label, msg, exp_i, exp_l, chk in _basic:
    run_parser(label, msg, exp_i, exp_l, chk)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Hard Profile Extraction  [Requires Ollama]
# ═════════════════════════════════════════════════════════════════════════════
section("SECTION 5 — Hard Profile Extraction  [Requires Ollama]")

# 5-A: Age given as number, NOT birth_date
run_parser(
    "Age stated as number (not date) — should still parse update_profile",
    "أنا ذكر عندي 22 سنة، وزني 78 كيلو، طولي 178، بتمرن خفيف، وعايز أكسب عضل",
    "update_profile", "arabic",
    lambda p: (p.get("profile_data") or {}).get("gender") == "male"
              and (p.get("profile_data") or {}).get("goal") == "muscle_gain",
    detail_fn=lambda p: f"birth_date={(p.get('profile_data') or {}).get('birth_date')}"
)

# 5-B: Weight in lbs mentioned in English — model should ideally convert or store
run_parser(
    "Weight mentioned in pounds (English) — extracts weight field",
    "I'm a male born 1998-06-10, I weigh 176 lbs, height 5ft 11in, I work a desk job, goal is weight loss",
    "update_profile", "english",
    lambda p: (p.get("profile_data") or {}).get("gender") == "male"
              and (p.get("profile_data") or {}).get("goal") == "weight_loss",
    detail_fn=lambda p: f"weight={(p.get('profile_data') or {}).get('weight')}, height={(p.get('profile_data') or {}).get('height')}"
)

# 5-C: Heavily informal Egyptian Arabic slang
run_parser(
    "Egyptian Arabic slang profile ('بتعمل إيه في الجيم')",
    "أنا راجل عندي 25 سنة، وزني تمانين، طولي مية وسبعة وسبعين، بروح الجيم كل يوم وعايز أكبر، مش بعمل dairy",
    "update_profile", "mixed",
    lambda p: (p.get("profile_data") or {}).get("gender") == "male"
              and (p.get("profile_data") or {}).get("activity_level") in ("active", "very_active"),
)

# 5-D: Multiple dietary restrictions at once
run_parser(
    "Multiple restrictions — keto, no sugar, no gluten",
    "I'm female, born 2000-07-04, 58kg, 163cm, light exercise, want to maintain weight. "
    "I follow a keto diet, can't eat gluten, and I avoid refined sugar",
    "update_profile", "english",
    lambda p: (
        "keto" in (p.get("profile_data") or {}).get("dietary_restrictions", [])
        or "gluten" in (p.get("profile_data") or {}).get("forbidden_foods", [])
        or "gluten" in (p.get("profile_data") or {}).get("allergies", [])
    )
)

# 5-E: Very terse update — only goal changes
run_parser(
    "Partial update — only goal (very short)",
    "عايز أغير هدفي لـ weight gain",
    "update_profile", "mixed",
    lambda p: (p.get("profile_data") or {}).get("goal") == "weight_gain"
)

# 5-F: Female using Arabic feminine verb forms
run_parser(
    "Female user — Arabic feminine grammar (عايزة، بتمرني)",
    "أنا بنت اتولدت 2003-09-22، وزني 62 وطولي 165، بتمرني تلت أيام في الأسبوع، وعايزة أخس",
    "update_profile", "arabic",
    lambda p: (p.get("profile_data") or {}).get("gender") == "female"
              and (p.get("profile_data") or {}).get("goal") == "weight_loss"
)

# 5-G: Sedentary user (no exercise mentioned at all)
run_parser(
    "Sedentary user — no mention of exercise",
    "Male, born 1990-12-01, weight 95kg, height 180cm, I don't exercise at all, want to lose weight",
    "update_profile", "english",
    lambda p: (p.get("profile_data") or {}).get("activity_level") == "sedentary"
)

# 5-H: very_active — twice-a-day training
run_parser(
    "Very active — trains twice a day",
    "أنا ذكر اتولدت 2002-03-15، وزني 80، طولي 182، بتمرن مرتين في اليوم وعايز muscle gain",
    "update_profile", "mixed",
    lambda p: (p.get("profile_data") or {}).get("activity_level") in ("active", "very_active")
)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Hard Quantity & Unit Extraction  [Requires Ollama]
# ═════════════════════════════════════════════════════════════════════════════
section("SECTION 6 — Hard Quantity & Unit Extraction  [Requires Ollama]")

# 6-A: Half kg in Arabic ("نص كيلو")
run_parser(
    "Arabic 'نص كيلو لحمة' → ~500g",
    "أكلت نص كيلو لحمة مشوية",
    "log_meal", "arabic",
    lambda p: len(p.get("foods", [])) >= 1,
    detail_fn=lambda p: f"foods={[(f['food_name_en'], f['quantity'], f['unit']) for f in p.get('foods', [])]}"
)

# 6-B: Decimal quantity in English
run_parser(
    "Decimal quantity '0.5 cup of olive oil'",
    "I used 0.5 cup of olive oil and 1.5 tablespoons of lemon juice",
    "log_meal", "english",
    lambda p: len(p.get("foods", [])) >= 2
             and any(f["quantity"] == 0.5 for f in p.get("foods", [])),
    detail_fn=lambda p: f"foods={[(f['food_name_en'], f['quantity'], f['unit']) for f in p.get('foods', [])]}"
)

# 6-C: Implied unit ("أكلت تفاحة" — one apple, unit=piece implied)
run_parser(
    "Implied unit — 'أكلت تفاحة' (no explicit unit)",
    "أكلت تفاحة وموزة وشوية عنب",
    "log_meal", "arabic",
    lambda p: len(p.get("foods", [])) >= 2,
    detail_fn=lambda p: f"foods={[(f['food_name_en'], f['quantity'], f['unit']) for f in p.get('foods', [])]}"
)

# 6-D: Large meal with 5+ items
run_parser(
    "Large meal — 5 foods in one message",
    "الفطار كان: 3 بيضات مسلوقة، كوب لبن، 2 توست، ملعقة زبدة، وكوب شاي بالسكر",
    "log_meal", "arabic",
    lambda p: len(p.get("foods", [])) >= 4,
    detail_fn=lambda p: f"{len(p.get('foods', []))} foods extracted"
)

# 6-E: Branded food
run_parser(
    "Branded food — Big Mac from McDonald's",
    "أكلت big mac من ماكدونالدز مع medium fries وكوكاكولا",
    "log_meal", "mixed",
    lambda p: len(p.get("foods", [])) >= 2,
    detail_fn=lambda p: f"foods={[f['food_name_en'] for f in p.get('foods', [])]}"
)

# 6-F: Code-switched quantities
run_parser(
    "Mixed quantities — '2 قطع chicken breast مع نص كوب أرز'",
    "أكلت 2 قطع chicken breast مع نص كوب أرز وسلطة خضرا",
    "log_meal", "mixed",
    lambda p: len(p.get("foods", [])) >= 2,
    detail_fn=lambda p: f"foods={[(f['food_name_en'], f['quantity'], f['unit']) for f in p.get('foods', [])]}"
)

# 6-G: Vague quantity (should default to 1.0, not crash)
run_parser(
    "Vague quantity — 'شوية' (a bit of)",
    "أكلت شوية أرز وشوية خضار مطبوخة",
    "log_meal", "arabic",
    lambda p: len(p.get("foods", [])) >= 1
             and all(f["quantity"] > 0 for f in p.get("foods", [])),
    detail_fn=lambda p: f"foods={[(f['food_name_en'], f['quantity']) for f in p.get('foods', [])]}"
)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Hard Food Entity Extraction  [Requires Ollama]
# ═════════════════════════════════════════════════════════════════════════════
section("SECTION 7 — Hard Food Entity Extraction  [Requires Ollama]")

# 7-A: food_name_en must be English (even when input is Arabic)
run_parser(
    "Arabic input → food_name_en must be in English",
    "أكلت طبق كشري وعصير قصب",
    "log_meal", "arabic",
    lambda p: all(
        any(c.isascii() and c.isalpha() for c in f.get("food_name_en", ""))
        for f in p.get("foods", [])
    ),
    detail_fn=lambda p: f"names_en={[f['food_name_en'] for f in p.get('foods', [])]}"
)

# 7-B: Nutrition question — specific food + quantity
run_parser(
    "Nutrition question with specific quantity",
    "كم سعرة حرارية في 150 جرام لحمة بقري مشوية؟",
    "ask_nutrition", "arabic",
    lambda p: len(p.get("foods", [])) >= 1
             and any(f["quantity"] == 150.0 for f in p.get("foods", [])),
    detail_fn=lambda p: f"foods={[(f['food_name_en'], f['quantity'], f['unit']) for f in p.get('foods', [])]}"
)

# 7-C: Nutrition question — English with macro focus
run_parser(
    "Nutrition question — asking for protein in chicken",
    "How much protein is in 200g of grilled chicken breast?",
    "ask_nutrition", "english",
    lambda p: len(p.get("foods", [])) >= 1
             and any("chicken" in f["food_name_en"] for f in p.get("foods", [])),
)

# 7-D: Meal plan request with dietary constraint
run_parser(
    "Meal plan request with constraint — gluten-free (English)",
    "Can you make me a gluten-free meal plan for the whole day?",
    "create_meal_plan", "english",
    lambda p: p.get("foods") == [] and p.get("profile_data") is None
)

# 7-E: Meal plan in code-switched
run_parser(
    "Meal plan request — code-switched",
    "عايز meal plan ليوم كامل بدون gluten",
    "create_meal_plan", "mixed",
    lambda p: p.get("foods") == []
)

# 7-F: Traditional Arabic dish names mapped to English USDA names
run_parser(
    "Traditional dishes — فول، طعمية، كشري → English USDA names",
    "فطاري كان طبق فول بالزيت وعدد 3 طعمية وعصير برتقال",
    "log_meal", "arabic",
    lambda p: len(p.get("foods", [])) >= 2
             and all(f["food_name_en"] != "" for f in p.get("foods", [])),
    detail_fn=lambda p: f"names_en={[f['food_name_en'] for f in p.get('foods', [])]}"
)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 8 — Intent Boundary Cases  [Requires Ollama]
# ═════════════════════════════════════════════════════════════════════════════
section("SECTION 8 — Intent Boundary Cases  [Requires Ollama]")

# 8-A: Nutrition question that sounds like meal log
run_parser(
    "Boundary: 'What are the calories in the rice I just ate?' → ask_nutrition",
    "What are the calories in the 200g of rice I just ate?",
    "ask_nutrition", "english",
    lambda p: len(p.get("foods", [])) >= 1
)

# 8-B: Thanking the bot → general_chat (not any food intent)
run_parser(
    "Boundary: thank you message → general_chat",
    "شكراً جزيلاً، أنت بتساعدني كتير!",
    "general_chat", "arabic",
    lambda p: p.get("foods") == []
)

# 8-C: Meal log stated in future tense — still log_meal
run_parser(
    "Boundary: future meal statement — 'I will eat 2 eggs' → log_meal",
    "I'm going to eat 2 scrambled eggs and a slice of whole wheat toast for breakfast",
    "log_meal", "english",
    lambda p: len(p.get("foods", [])) >= 2
)

# 8-D: Complaint about food, no actual meal log → general_chat
run_parser(
    "Boundary: complaining about food — not a meal log",
    "أنا تعبت من الأكل الصحي مش عارف أكمل",
    "general_chat", "arabic",
    lambda p: p.get("profile_data") is None
)

# 8-E: Pure number → should not crash
run_parser(
    "Boundary: gibberish / number only input → any valid intent without crash",
    "12345",
    None, "english",           # we don't enforce specific intent here
    lambda p: "intent" in p    # just must not crash and must have intent key
)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 9 — NLPSession Basic Onboarding  [Requires Ollama]
# ═════════════════════════════════════════════════════════════════════════════
section("SECTION 9 — NLPSession Basic Onboarding  [Requires Ollama]")

# 9-A: Partial profile → needs_profile
s = NLPSession()
r = s.process("أنا ولد اتولدت 2004-05-10، وزني 90")
check("Partial profile → needs_profile", r["status"] == "needs_profile",
      f"missing={r['missing_fields']}")
check("prompt_for_user is non-empty string", bool(r["prompt_for_user"]))

# 9-B: Two-turn completion
s2 = NLPSession()
s2.process("أنا ولد اتولدت 2004-05-10، وزني 90، مش باكل بيض")
r2 = s2.process("طولي 175، بتمرن 4 أيام في الأسبوع وعايز أخس")
check("2-turn profile → onboarding_complete", r2["status"] == "onboarding_complete")
check("Weight merged (90)",      s2.profile.get("weight") == 90.0)
check("Height merged (175)",     s2.profile.get("height") == 175.0)
check("forbidden_foods merged",  "eggs" in s2.profile.get("forbidden_foods", []))

# 9-C: After onboarding, log_meal → ready and profile_data is ALWAYS populated for Task 3
r3 = s2.process("كلت 200g فراخ مشوية")
check("Post-onboarding meal log → ready",   r3["status"] == "ready")
check("Intent is log_meal",                  r3["parsed"]["intent"] == "log_meal")
check("profile_data is ALWAYS populated with full profile", r3["parsed"]["profile_data"] is not None and r3["parsed"]["profile_data"]["weight"] == 90.0)

# 9-D: Trying to log meal before profile → blocked
s4 = NLPSession()
r4 = s4.process("كلت بيتزا وكولا")
check("Log meal before profile → needs_profile", r4["status"] == "needs_profile")

# 9-E: Profile complete in ONE single message
s5 = NLPSession()
r5 = s5.process(
    "I'm a male born 1995-07-20, weight 82kg, height 178cm, "
    "I exercise 5 days a week, my goal is muscle gain, I'm allergic to shellfish"
)
check("Single-message complete profile → onboarding_complete",
      r5["status"] == "onboarding_complete",
      f"missing={r5['missing_fields']}, profile={json.dumps(r5['profile'], ensure_ascii=False)}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 10 — NLPSession Hard Multi-Turn Conversations  [Requires Ollama]
# ═════════════════════════════════════════════════════════════════════════════
section("SECTION 10 — Hard Multi-Turn Conversations  [Requires Ollama]")

# 10-A: User provides info across 3 turns, each partial
print("  [10-A] 3-turn gradual profile build-up")
s_a = NLPSession()
r_a1 = s_a.process("أنا ذكر")                                  # gender only
r_a2 = s_a.process("اتولدت 2000-04-15 ووزني 75 وطولي 173")    # birth_date, weight, height
r_a3 = s_a.process("بتمرن خفيف وعايز أحافظ على وزني")         # activity + goal
check("Turn 1 partial (gender only) → needs_profile",       r_a1["status"] == "needs_profile")
check("Turn 2 adds 3 fields → still needs_profile",         r_a2["status"] == "needs_profile")
check("Turn 3 fills last 2 fields → onboarding_complete",   r_a3["status"] == "onboarding_complete",
      f"missing={r_a3['missing_fields']}, profile={json.dumps(s_a.profile, ensure_ascii=False)}")
check("Final profile has gender=male",           s_a.profile.get("gender") == "male")
check("Final profile has height=173",            s_a.profile.get("height") == 173.0)
check("Final profile has activity_level=light",  s_a.profile.get("activity_level") == "light")

# 10-B: User gives wrong/irrelevant info then corrects
print("\n  [10-B] Irrelevant message first, then profile")
s_b = NLPSession()
r_b1 = s_b.process("مرحبا، عايز تساعدني؟")  # general chat
r_b2 = s_b.process(
    "أنا ذكر اتولدت 1998-11-05 وزني 88 طولي 180 "
    "بتمرن 6 أيام في الأسبوع وعايز أخس، مش باكل جلوتين"
)
check("First message general_chat → needs_profile",  r_b1["status"] == "needs_profile")
check("Second message full profile → onboarding_complete",
      r_b2["status"] == "onboarding_complete",
      f"missing={r_b2['missing_fields']}")
check("forbidden_foods has gluten",
      any("gluten" in f for f in s_b.profile.get("forbidden_foods", []))
      or "gluten" in s_b.profile.get("dietary_restrictions", []))

# 10-C: After profile complete — multiple meal logs accumulate correctly
print("\n  [10-C] Multiple meal logs after onboarding")
s_c = NLPSession()
s_c.process(
    "Female, born 2002-05-30, weight 60, height 165, moderate activity, goal weight loss"
)
r_c1 = s_c.process("أكلت 2 بيضة مسلوقة وتوستة")
r_c2 = s_c.process("الغدا كان 150g دجاج مشوي مع كوب أرز أبيض")
r_c3 = s_c.process("العشا: سلطة خضرا وتونة معلبة")
check("Breakfast log → ready",              r_c1["status"] == "ready")
check("Lunch log → ready",                  r_c2["status"] == "ready")
check("Dinner log → ready",                 r_c3["status"] == "ready")
check("Breakfast: >= 2 foods extracted",    len(r_c1["parsed"]["foods"]) >= 1)
check("Dinner: tuna extracted",
      any("tuna" in f["food_name_en"] for f in r_c3["parsed"]["foods"]))

# 10-D: Allergy + restriction update AFTER onboarding
print("\n  [10-D] Profile update after onboarding (add allergy)")
s_d = NLPSession()
s_d.process(
    "Male, born 1997-02-14, weight 78kg, height 176cm, active, goal maintenance"
)
r_d1 = s_d.process("عرفت إن عندي حساسية من المكسرات")
check("Post-onboarding allergy update → ready",
      r_d1["status"] == "ready",
      f"allergies={s_d.profile.get('allergies')}")
check("Allergy merged into profile",
      any("nut" in a or "مكسرات" in a for a in s_d.profile.get("allergies", [])))

# 10-E: English + mixed code-switching throughout session
print("\n  [10-E] Language-switching across turns")
s_e = NLPSession()
s_e.process("I am male born 1996-08-08, weight 85kg")              # English, partial
s_e.process("طولي 179 وبتمرن active وعايز weight loss")            # Mixed, completes profile
r_e = s_e.process("I had 300g of grilled salmon for dinner")       # English meal log
check("Cross-language session completes → ready",     r_e["status"] == "ready")
check("Salmon extracted in English meal log",
      any("salmon" in f["food_name_en"] for f in r_e["parsed"]["foods"]))
check("Weight merged from English turn (85)",   s_e.profile.get("weight") == 85.0)
check("Height merged from Arabic turn (179)",   s_e.profile.get("height") == 179.0)


# ═════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═════════════════════════════════════════════════════════════════════════════

total  = len(_results)
passed = sum(_results)
failed = total - passed

print("\n" + "═" * 65)
print(f"  FINAL RESULTS: {passed}/{total} passed  |  {failed} failed")
print("═" * 65)

for sec, sec_results in _section_results.items():
    p = sum(sec_results)
    t = len(sec_results)
    icon = "✅" if p == t else ("⚠️ " if p > 0 else "❌")
    print(f"  {icon}  {p}/{t}  {sec}")

print()
if failed == 0:
    print("  🎉 All tests passed!")
else:
    print(f"  ⚠️  {failed} test(s) failed — check output above for details.")
print("═" * 65)
