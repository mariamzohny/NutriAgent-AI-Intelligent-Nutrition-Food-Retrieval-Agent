import sys
import io
import json

# Set UTF-8 encoding for Arabic characters
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from nlp_parser import parse_user_input, NLPSession

test_queries = [
    # ── 1. calculate_calories (Arabic, English, Code-Switching) ──────────────
    ("1. Meal Log (Arabic - 3 items)", "كلت 200g فراخ مشوية مع طبق رز وسلطة خضرا"),
    ("2. Meal Log (English - Breakfast)", "I had 2 boiled eggs, 1 cup of oatmeal, and a glass of orange juice for breakfast"),
    ("3. Meal Log (Mixed / Code-Switching)", "أكلت 150g grilled salmon مع 1 cup brown rice وسلطة"),

    # ── 2. food_nutrition (Single food calorie queries) ───────────────────────
    ("4. Food Nutrition Query (Arabic)", "كم سعرة حرارية في 100 جرام أفوكادو؟"),
    ("5. Food Nutrition Query (English)", "How many calories are in 200g of sweet potato?"),

    # ── 3. meal_recommendation (Specific meal type + target calories) ─────────
    ("6. Meal Recommendation (English Target)", "Make me a 400-calorie breakfast without eggs"),
    ("7. Meal Recommendation (Arabic Target)", "عايز وجبة غداء سريعة في حدود 600 سعرة حرارية بدون سمك"),

    # ── 4. daily_plan (Full day diet plan requests) ───────────────────────────
    ("8. Daily Plan Request (Arabic)", "ممكن تعملي دايت وخطة أكل كاملة لإنقاص الوزن؟"),
    ("9. Daily Plan Request (English)", "Give me a full day meal plan for muscle building"),

    # ── 5. food_substitution ──────────────────────────────────────────────────
    ("10. Food Substitution (English)", "What can I substitute for cheese in my diet?"),
    ("11. Food Substitution (Arabic)", "إيه بديل البيض في وجبة الفطار؟"),

    # ── 6. fitness_nutrition_question (FAISS RAG Triggers) ────────────────────
    ("12. Fitness Nutrition (Arabic Pre-Workout)", "أكل إيه قبل التمرين بنص ساعة عشان يديني باور؟"),
    ("13. Fitness Nutrition (English Post-Workout)", "What should I eat after an intense gym workout for muscle recovery?"),

    # ── 7. update_profile (Biometrics & Constraints) ──────────────────────────
    ("14. Profile Onboarding (Arabic Full)", "أنا ولد اتولدت 2004-05-10، وزني 90 وطولي 175، بتمرن 4 أيام وعايز أخس، مش باكل بيض وعندي حساسية ألبان"),
    ("15. Profile Onboarding (English with Age)", "I am a male, 25 years old, weight 80kg, height 180cm, active lifestyle, goal is muscle gain")
]

print("=" * 70)
print("       TASK 1 NLP PARSER: COMPREHENSIVE OUTPUT DEMONSTRATION")
print("=" * 70)

for title, query in test_queries:
    print(f"\n┌{'─' * 68}┐")
    print(f"│ 📌 TEST: {title:<58} │")
    print(f"│ 💬 USER INPUT: {query:<53} │")
    print(f"└{'─' * 68}┘")
    
    parsed = parse_user_input(query)
    print(json.dumps(parsed, ensure_ascii=False, indent=2))
