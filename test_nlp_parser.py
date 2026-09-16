"""
test_nlp_parser.py
==================
Task 1 — NLP User Understanding: Comprehensive Hard Test Suite (Updated Taxonomy)

Sections:
  1. detect_language()                  — deterministic, no model needed
  2. normalize_output()                 — deterministic, no model needed
  3. validate_mandatory_profile_fields() & sanity checks — deterministic
  4. LLM Parser — Basic single-turn (Aligned with project intent taxonomy)
  5. LLM Parser — Hard Profile Extraction (Slang, age-as-number, lbs/ft)
  6. LLM Parser — Hard Quantity & Unit edge cases
  7. LLM Parser — Food entity hard cases (Branded, traditional, USDA mapping)
  8. LLM Parser — Special Intents (meal_recommendation, food_substitution, fitness_nutrition_question)
  9. NLPSession — Selective Onboarding (Unblocked simplified queries vs blocked daily_plan)
 10. NLPSession — Hard Multi-Turn Conversations & State Management

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
    sanitize_extracted_values,
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

raw_ar = {"intent": "calculate_calories", "language_detected": "arabic", "raw_text": "t",
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
raw_dup = {"intent": "calculate_calories", "language_detected": "english", "raw_text": "t",
           "profile_data": None,
           "foods": [
               make_food("grilled chicken", 200, "gram"),
               make_food("grilled chicken", 200, "gram"),
               make_food("grilled chicken", 150, "gram"),
           ]}
out_dup = normalize_output(raw_dup)
check("Exact duplicates removed → 2 foods left", len(out_dup["foods"]) == 2)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — validate_mandatory_profile_fields & Sanity Checks [No model]
# ═════════════════════════════════════════════════════════════════════════════
section("SECTION 3 — Profile Validation, Sanity Checks & Gender Fallback [No model]")

_full = {"birth_date": "2004-05-10", "gender": "male", "weight": 90.0,
         "height": 175.0, "activity_level": "moderate", "goal": "weight_loss",
         "dietary_restrictions": [], "forbidden_foods": [], "preferred_foods": [], "allergies": []}

ok, miss = validate_mandatory_profile_fields(_full)
check("All 6 fields present → complete", ok, f"missing={miss}")

for field in ["birth_date", "gender", "weight", "height", "activity_level", "goal"]:
    p = {**_full, field: None}
    ok2, miss2 = validate_mandatory_profile_fields(p)
    check(f"Missing only '{field}' → incomplete", not ok2 and field in miss2)

# Sanity check tests (Critical Gap 7)
sanity_test_1 = {
    "intent": "update_profile", "target_calories": 450.0,
    "profile_data": {"height": 1750.0, "weight": 85.0} # 1750 mm -> 175.0 cm
}
sanitized_1 = sanitize_extracted_values(sanity_test_1)
check("Sanity: height 1750 mm corrected to 175.0 cm", sanitized_1["profile_data"]["height"] == 175.0)

sanity_test_2 = {
    "intent": "update_profile", "target_calories": 50000.0, # out of range
    "profile_data": {"height": 30.0, "weight": 800.0}       # out of range
}
sanitized_2 = sanitize_extracted_values(sanity_test_2)
check("Sanity: crazy height (30cm) set to None", sanitized_2["profile_data"]["height"] is None)
check("Sanity: crazy weight (800kg) set to None", sanitized_2["profile_data"]["weight"] is None)
check("Sanity: crazy target calories set to None", sanitized_2["target_calories"] is None)

# Gender fallback tests
check("Gender extracted: 'أنا ولد' → male",       _extract_gender_from_text("أنا ولد اتولدت 2004-05-10") == "male")
check("Gender extracted: 'أنا ذكر' → male",      _extract_gender_from_text("أنا ذكر") == "male")
check("Gender extracted: 'أنا بنت' → female",     _extract_gender_from_text("أنا بنت وزني 60") == "female")
check("Gender extracted: 'I am female' → female", _extract_gender_from_text("I am female born 1999") == "female")
check("Gender extracted: 'I am a man' → male",   _extract_gender_from_text("I am a man") == "male")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Basic Single-Turn LLM Tests  [Requires Ollama]
# ═════════════════════════════════════════════════════════════════════════════
section("SECTION 4 — Basic LLM Parser (Aligned Taxonomy)  [Requires Ollama]")

_basic = [
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
        "Arabic meal log — calculate_calories",
        "كلت 200g فراخ مشوية مع طبق رز",
        "calculate_calories", "arabic",
        lambda p: len(p.get("foods", [])) >= 2
    ),
    (
        "English meal log — calculate_calories",
        "I had 2 boiled eggs and a cup of oats for breakfast",
        "calculate_calories", "english",
        lambda p: len(p.get("foods", [])) >= 2
    ),
    (
        "Code-switched meal log",
        "أكلت grilled salmon مع 1 cup برية",
        "calculate_calories", "mixed",
        lambda p: len(p.get("foods", [])) >= 2
    ),
    (
        "Daily Plan request — Arabic",
        "ممكن تعملي خطة أكل ليوم كامل؟",
        "daily_plan", "arabic",
        lambda p: True
    ),
    (
        "Food Nutrition query — English",
        "How many calories are in 100g of avocado?",
        "food_nutrition", "english",
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

run_parser(
    "Age stated as number — calculates birth_date",
    "أنا ذكر عندي 22 سنة، وزني 78 كيلو، طولي 178، بتمرن خفيف، وعايز أكسب عضل",
    "update_profile", "arabic",
    lambda p: (p.get("profile_data") or {}).get("gender") == "male"
              and (p.get("profile_data") or {}).get("goal") == "muscle_gain"
)

run_parser(
    "Egyptian Arabic slang profile",
    "أنا راجل عندي 25 سنة، وزني تمانين، طولي مية وسبعة وسبعين، بروح الجيم كل يوم وعايز أكبر، مش بعمل dairy",
    "update_profile", "mixed",
    lambda p: (p.get("profile_data") or {}).get("gender") == "male"
)

run_parser(
    "Female user — Arabic feminine grammar (عايزة، بتمرن)",
    "أنا بنت اتولدت 2003-09-22، وزني 62 وطولي 165، بتمرن تلت أيام في الأسبوع، وعايزة أخس",
    "update_profile", "arabic",
    lambda p: (p.get("profile_data") or {}).get("gender") == "female"
              and (p.get("profile_data") or {}).get("goal") == "weight_loss"
)

run_parser(
    "Sedentary user — no mention of exercise",
    "Male, born 1990-12-01, weight 95kg, height 180cm, I don't exercise at all, want to lose weight",
    "update_profile", "english",
    lambda p: (p.get("profile_data") or {}).get("activity_level") == "sedentary"
)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Critical New Intent Features (Critical Gaps 1 & 2) [Requires Ollama]
# ═════════════════════════════════════════════════════════════════════════════
section("SECTION 6 — Critical Special Intents (meal_recommendation, food_substitution, fitness_nutrition)")

# 6-A: Meal recommendation with target calories & meal type (Gap 1)
run_parser(
    "Meal recommendation: 400-cal breakfast without eggs",
    "Make me a 400-calorie breakfast without eggs",
    "meal_recommendation", "english",
    lambda p: p.get("meal_type") == "breakfast" and p.get("target_calories") == 400.0,
    detail_fn=lambda p: f"meal_type={p.get('meal_type')}, target_cal={p.get('target_calories')}"
)

# 6-B: Arabic Meal recommendation with target calories
run_parser(
    "Arabic Meal recommendation: 600-cal lunch",
    "عايز وجبة غداء في حدود 600 سعرة بدون سمك",
    "meal_recommendation", "arabic",
    lambda p: p.get("meal_type") == "lunch" and p.get("target_calories") == 600.0,
    detail_fn=lambda p: f"meal_type={p.get('meal_type')}, target_cal={p.get('target_calories')}"
)

# 6-C: Food substitution (Gap 2)
run_parser(
    "Food substitution — substitute for cheese",
    "What can I substitute for cheese in my diet?",
    "food_substitution", "english",
    lambda p: p.get("requested_substitution") is not None or any("cheese" in f["food_name_en"] for f in p.get("foods", []))
)

# 6-D: Fitness nutrition question (Gap 2 - FAISS RAG Trigger)
run_parser(
    "Fitness nutrition question — pre workout meal",
    "أكل إيه قبل التمرين عشان يديني طاقة؟",
    "fitness_nutrition_question", "arabic",
    lambda p: p.get("profile_data") is None
)

# 6-E: Fitness nutrition question (English)
run_parser(
    "Fitness nutrition question — post workout protein",
    "What is the best protein to eat after workout?",
    "fitness_nutrition_question", "english",
    lambda p: p.get("profile_data") is None
)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Hard Quantity & Food Extraction  [Requires Ollama]
# ═════════════════════════════════════════════════════════════════════════════
section("SECTION 7 — Hard Quantity & Food Extraction  [Requires Ollama]")

run_parser(
    "Arabic 'نص كيلو لحمة'",
    "أكلت نص كيلو لحمة مشوية",
    "calculate_calories", "arabic",
    lambda p: len(p.get("foods", [])) >= 1
)

run_parser(
    "Decimal quantity '0.5 cup of olive oil'",
    "I used 0.5 cup of olive oil and 1.5 tablespoons of lemon juice",
    "calculate_calories", "english",
    lambda p: len(p.get("foods", [])) >= 2
)

run_parser(
    "Branded food — Big Mac from McDonald's",
    "أكلت big mac من ماكدونالدز مع medium fries وكوكاكولا",
    "calculate_calories", "mixed",
    lambda p: len(p.get("foods", [])) >= 2
)

run_parser(
    "Traditional dishes — كشري",
    "أكلت طبق كشري وعصير قصب",
    "calculate_calories", "arabic",
    lambda p: len(p.get("foods", [])) >= 1
)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 8 — Selective Onboarding Flow (Critical Gap 3) [Requires Ollama]
# ═════════════════════════════════════════════════════════════════════════════
section("SECTION 8 — Selective Onboarding (Unblocked Queries vs Blocked Daily Plan)")

# 8-A: Simplified single nutrition query MUST NOT be blocked!
s_unblocked = NLPSession()
r_unblocked = s_unblocked.process("How many calories are in 200g chicken breast?")
check("Simplified query 'food_nutrition' → status=ready (NOT blocked!)",
      r_unblocked["status"] == "ready" and r_unblocked["parsed"]["intent"] == "food_nutrition")

# 8-B: Simplified fitness question MUST NOT be blocked!
r_fit = s_unblocked.process("أكل إيه قبل التمرين؟")
check("Simplified fitness query → status=ready (NOT blocked!)",
      r_fit["status"] == "ready" and r_fit["parsed"]["intent"] == "fitness_nutrition_question")

# 8-C: Food substitution MUST NOT be blocked!
r_sub = s_unblocked.process("What can I substitute for cheese?")
check("Food substitution query → status=ready (NOT blocked!)",
      r_sub["status"] == "ready" and r_sub["parsed"]["intent"] == "food_substitution")

# 8-D: Daily plan WITHOUT profile MUST be blocked for onboarding!
s_blocked = NLPSession()
r_block = s_blocked.process("ممكن تعملي خطة أكل ليوم كامل؟")
check("Daily plan without profile → status=needs_profile (Blocked correctly!)",
      r_block["status"] == "needs_profile")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 9 — Multi-Turn Onboarding & State Management [Requires Ollama]
# ═════════════════════════════════════════════════════════════════════════════
section("SECTION 9 — Multi-Turn Onboarding & State Management")

session_full = NLPSession()
# Turn 1
r1 = session_full.process("أنا ولد اتولدت 2004-05-10، وزني 90 وطولي 175، بتمرن 4 أيام وعايز أخس")
check("Turn 1 complete profile → status=onboarding_complete",
      r1["status"] == "onboarding_complete")
check("Profile merged weight (90)", session_full.profile.get("weight") == 90.0)
check("Profile merged height (175)", session_full.profile.get("height") == 175.0)

# Turn 2: Daily plan after onboarding
r2 = session_full.process("اعملي خطة أكل ليوم كامل")
check("Turn 2 daily plan after onboarding → status=ready",
      r2["status"] == "ready")
check("profile_data attached to parsed object",
      r2["parsed"]["profile_data"] is not None and r2["parsed"]["profile_data"]["weight"] == 90.0)


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
