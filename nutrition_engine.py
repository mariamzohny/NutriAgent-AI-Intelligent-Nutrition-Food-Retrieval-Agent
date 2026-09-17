# -*- coding: utf-8 -*-
"""
nutrition_engine.py — Task 3: Nutrition Calculation Engine.

Public API:
    calculate_age(profile)
    calculate_bmr(profile)
    calculate_tdee(profile)
    calculate_target_calories(profile)
    calculate_macros(profile)
    calculate_food_nutrition(food_nutrition_data, quantity_g)
    calculate_meal_total(foods, food_nutrition_data)
    validate_daily_nutrition(meals, profile)
    validate_meal_nutrition(meal, profile, target_calories=None)

Design:
    - Deterministic — no LLM, no API calls.
    - Pure functions — same input → same output.
    - `foods` come from Task 1 (NLPSession).
    - `food_nutrition_data` come from Task 2 (get_food_nutrition).
    - Unit conversion uses `convert_to_grams` from Task 2.
"""
import logging
import re
from datetime import date
from typing import Dict, List, Optional

from usda_retrieval import convert_to_grams

logger = logging.getLogger(__name__)

# ─── Calorie adjustment constants ───────────────────────────────────────────
# Evidence-based ranges (NHS, HAS, Healthline, MyProtein, etc.):
#   - weight loss:    300–500 kcal deficit
#   - weight gain:    300–500 kcal surplus
#   - muscle gain:    300–500 kcal surplus (lean bulk)
#   - maintenance:    no adjustment
#
# We default to the middle of the recommended range.
DEFAULT_DEFICIT_KCAL = 400     # weight_loss
DEFAULT_SURPLUS_KCAL = 400     # weight_gain / muscle_gain

# Hard floor: never recommend below this, even if the math says so.
# Prolonged very-low-calorie diets are dangerous without medical supervision.
MIN_SAFE_CALORIES = {
    "male":   1500,
    "female": 1200,
}

# ─── Restriction synonyms (Arabic ↔ English) ─────────────────────────────────
# Maps a canonical key → set of aliases (Arabic + English).
# Used to match `forbidden_foods` and `allergies` against actual food names,
# regardless of which language the user wrote them in.
#
# Example: user writes "بيض" → canonical "eggs" → matches food "whole eggs".
_RESTRICTION_SYNONYMS: Dict[str, set] = {
    "eggs": {
        "بيض", "بيضة", "بيضه", "بيض مسلوق",
        "egg", "eggs", "boiled egg", "boiled eggs",
    },
    "dairy": {
        "ألبان", "البان", "حليب", "لبن", "زبادي", "جبنة", "جبن", "قشطة", "كريمة",
        "dairy", "milk", "yogurt", "yoghurt", "cheese", "cream", "butter",
    },
    "nuts": {
        "مكسرات", "فول سوداني", "لوز", "كاجو", "بندق", "بستم", "فستق",
        "nuts", "nut", "peanut", "peanuts", "almond", "almonds",
        "cashew", "hazelnut", "pistachio", "walnut",
    },
    "fish": {
        "سمك", "سلمون", "تونة", "سردين", "ماكريل",
        "fish", "salmon", "tuna", "sardine", "mackerel",
    },
    "shellfish": {
        "جمبري", "جمبري", "كابوريا", "استاكوزا", "محار",
        "shellfish", "shrimp", "prawn", "crab", "lobster", "oyster",
    },
    "gluten": {
        "جلوتين", "قمح", "خبز", "عيش", "مكرونة", "معكرونة", "برغل", "كسكس",
        "gluten", "wheat", "bread", "pasta", "couscous", "bulgur",
    },
    "soy": {
        "صويا", "توفو", "ادامامي",
        "soy", "soya", "tofu", "edamame",
    },
    "chicken": {
        "دجاج", "فراخ", "فرخة",
        "chicken",
    },
    "meat": {
        "لحمة", "لحم", "لحم بقري", "ضاني", "ضأن", "كندوز",
        "meat", "beef", "lamb", "veal", "steak",
    },
    "pork": {
        "خنزير", "لحم خنزير", "بيكون", "هام",
        "pork", "bacon", "ham", "prosciutto",
    },
    "honey": {
        "عسل",
        "honey",
    },
}


# Dietary pattern → canonical restriction keys that pattern forbids.
# `keto` is intentionally empty — it's a macro constraint, not a food ban.
_DIET_PATTERN_RULES: Dict[str, set] = {
    "vegan":       {"eggs", "dairy", "meat", "chicken", "fish", "shellfish", "honey"},
    "vegetarian":  {"meat", "chicken", "fish", "shellfish"},
    "pescatarian": {"meat", "chicken"},
    "halal":       {"pork"},
    "keto":        set(),
}


_ARABIC_DIACRITICS_RE = re.compile(r"[\u064B-\u065F\u0670\u06D6-\u06ED]")
_ARABIC_ALEF_RE = re.compile(r"[إأآٱ]")


def _normalize_food_text(text: str) -> str:
    """Lowercase + Arabic normalization for restriction matching."""
    if not text:
        return ""
    text = text.lower().strip()
    text = _ARABIC_DIACRITICS_RE.sub("", text)   # strip diacritics
    text = text.replace("ـ", "")                  # strip tatweel
    text = _ARABIC_ALEF_RE.sub("ا", text)         # unify alef
    text = text.replace("ى", "ي")                 # unify yaa
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _restriction_keys(text: str) -> set:
    """
    Return the canonical restriction keys a given text matches.

    Single-word aliases are matched as whole tokens (so "أرز أبيض" does NOT
    match the "بيض" alias). Multi-word aliases are matched as substrings.
    """
    normalized = _normalize_food_text(text)
    if not normalized:
        return set()

    tokens = set(normalized.split())
    matched = set()

    for canonical, aliases in _RESTRICTION_SYNONYMS.items():
        for alias in aliases:
            alias_norm = _normalize_food_text(alias)
            if not alias_norm:
                continue
            if " " in alias_norm:
                if alias_norm in normalized:
                    matched.add(canonical)
                    break
            elif alias_norm in tokens:
                matched.add(canonical)
                break

    return matched


def _collect_user_restrictions(profile: dict) -> set:
    """
    Combine forbidden_foods + allergies + dietary_restrictions into a single
    set of canonical restriction keys.
    """
    keys = set()

    for name in profile.get("forbidden_foods", []) or []:
        keys |= _restriction_keys(name)
    for name in profile.get("allergies", []) or []:
        keys |= _restriction_keys(name)

    for pattern in profile.get("dietary_restrictions", []) or []:
        pattern_key = _normalize_food_text(pattern)
        keys |= _DIET_PATTERN_RULES.get(pattern_key, set())

    return keys


def _check_food_against_restrictions(
    food: dict,
    user_restrictions: set,
) -> set:
    """
    Check a single food item against the user's restriction set.
    Looks at both `food_name_original` and `food_name` (USDA name).
    """
    if not user_restrictions:
        return set()

    food_keys = _restriction_keys(food.get("food_name_original", ""))
    food_keys |= _restriction_keys(food.get("food_name", ""))
    food_keys |= _restriction_keys(food.get("food_name_en", ""))

    return food_keys & user_restrictions


def calculate_age(profile: dict) -> int:
    """
    Compute age in years from `birth_date` (YYYY-MM-DD).
    Correctly handles the case where the birthday hasn't happened yet this year.
    """
    birth_date = date.fromisoformat(profile["birth_date"])
    today = date.today()

    age = today.year - birth_date.year
    if (today.month, today.day) < (birth_date.month, birth_date.day):
        age -= 1

    logger.debug("Age computed: %s (birth_date=%s)", age, birth_date)
    return age


def calculate_bmr(profile: dict) -> float:
    """
    Basal Metabolic Rate using the Mifflin-St Jeor equation.

    Male:   BMR = 10*weight + 6.25*height − 5*age + 5
    Female: BMR = 10*weight + 6.25*height − 5*age − 161

    weight in kg, height in cm.
    """
    age = calculate_age(profile)
    weight = profile["weight"]
    height = profile["height"]
    gender = profile["gender"].lower()

    if gender == "male":
        bmr = (10 * weight) + (6.25 * height) - (5 * age) + 5
    elif gender == "female":
        bmr = (10 * weight) + (6.25 * height) - (5 * age) - 161
    else:
        raise ValueError("Gender must be 'male' or 'female'")

    bmr = round(bmr, 2)
    logger.debug("BMR computed: %s kcal (gender=%s, age=%s)", bmr, gender, age)
    return bmr


# Standard activity multipliers applied to BMR to get TDEE.
_ACTIVITY_FACTORS = {
    "sedentary":   1.2,
    "light":       1.375,
    "moderate":    1.55,
    "active":      1.725,
    "very_active": 1.9,
}


def calculate_tdee(profile: dict) -> float:
    """
    Total Daily Energy Expenditure = BMR × activity_factor.

    Activity levels:
        sedentary    1.2     (no exercise / desk job)
        light        1.375   (1–3 days/week)
        moderate     1.55    (3–5 days/week)
        active       1.725   (6–7 days/week)
        very_active  1.9     (intense daily training)
    """
    bmr = calculate_bmr(profile)
    activity_level = profile["activity_level"].lower()

    if activity_level not in _ACTIVITY_FACTORS:
        raise ValueError(
            f"Invalid activity level: {activity_level!r}. "
            f"Must be one of {list(_ACTIVITY_FACTORS.keys())}"
        )

    tdee = round(bmr * _ACTIVITY_FACTORS[activity_level], 2)
    logger.debug("TDEE computed: %s kcal (activity=%s)", tdee, activity_level)
    return tdee


def calculate_target_calories(
    profile: dict,
    deficit_kcal: int = DEFAULT_DEFICIT_KCAL,
    surplus_kcal: int = DEFAULT_SURPLUS_KCAL,
) -> float:
    """
    Compute daily target calories based on TDEE and goal.

    Approach: additive adjustment (TDEE ± fixed kcal), not multiplicative.
    This keeps the deficit/surplus inside the evidence-based 300–500 kcal range
    regardless of the user's TDEE.

    Args:
        profile: must contain `goal` and all fields needed for TDEE.
        deficit_kcal: kcal to subtract for weight_loss (default 400).
        surplus_kcal: kcal to add for weight_gain / muscle_gain (default 400).

    Returns:
        Target calories (float, rounded to 2 decimals).
    """
    tdee = calculate_tdee(profile)
    goal = profile["goal"].lower()

    if goal == "maintenance":
        target = tdee
    elif goal == "weight_loss":
        target = tdee - deficit_kcal
    elif goal in ("weight_gain", "muscle_gain"):
        target = tdee + surplus_kcal
    else:
        raise ValueError(f"Invalid goal: {goal!r}")

    # Safety floor — do not go below medically safe minimums
    gender = profile["gender"].lower()
    floor = MIN_SAFE_CALORIES.get(gender)
    if floor is not None and target < floor:
        logger.warning(
            "Target %s kcal for goal=%s was clamped to the %s kcal floor.",
            round(target, 2), goal, floor,
        )
        target = floor

    return round(target, 2)


def calculate_macros(profile: dict) -> dict:
    """
    Compute daily macro targets (protein / carbs / fat) for the user's goal.

    Protein is set by goal (g per kg bodyweight).
    Fat is set at 25% of target calories.
    Carbs fill the remainder — clamped to 0 to prevent negative values.
    """
    target_calories = calculate_target_calories(profile)
    weight = profile["weight"]
    goal = profile["goal"].lower()

    protein_factors = {
        "weight_loss": 2.0,
        "weight_gain": 1.6,
        "maintenance": 1.6,
        "muscle_gain": 2.0,
    }
    if goal not in protein_factors:
        raise ValueError(f"Invalid goal: {goal!r}")

    protein_grams = weight * protein_factors[goal]

    fat_calories = target_calories * 0.25
    fat_grams = fat_calories / 9

    protein_calories = protein_grams * 4

    # Clamp to 0 — a very low target + high protein can otherwise go negative.
    carbs_calories = max(0.0, target_calories - protein_calories - fat_calories)
    carbs_grams = carbs_calories / 4

    return {
        "calories": round(target_calories, 2),
        "protein_g": round(protein_grams, 2),
        "carbs_g": round(carbs_grams, 2),
        "fat_g": round(fat_grams, 2),
    }


def calculate_food_nutrition(
    food_nutrition_data: dict,
    quantity_g: float,
) -> dict:
    """
    Scale per-100g nutrition to the requested gram quantity.

    Args:
        food_nutrition_data: Task 2 output — must contain `*_per_100g` keys.
        quantity_g: quantity in grams (must be > 0).

    Returns:
        Scaled nutrition dict.
    """
    if quantity_g <= 0:
        raise ValueError("Quantity must be greater than 0")

    factor = quantity_g / 100.0

    return {
        "food_name": food_nutrition_data["food_name"],
        "quantity_g": round(quantity_g, 2),
        "calories": round(food_nutrition_data["calories_per_100g"] * factor, 2),
        "protein_g": round(food_nutrition_data["protein_per_100g"] * factor, 2),
        "carbs_g": round(food_nutrition_data["carbs_per_100g"] * factor, 2),
        "fat_g": round(food_nutrition_data["fat_per_100g"] * factor, 2),
    }


def calculate_meal_total(
    foods: List[dict],
    food_nutrition_data: List[dict],
) -> dict:
    """
    Compute total nutrition for a list of foods.

    Args:
        foods: Task 1 output — each is
               {food_name_original, food_name_en, quantity, unit}.
        food_nutrition_data: Task 2 outputs — same order as `foods`.
                             Each must include `portions` for unit conversion.

    Returns:
        {
          success: bool,
          foods: [{food_name_original, quantity_g, calories, protein_g, ...}],
          unmatched_foods: [{food_name_original, reason}],
          total_calories, total_protein_g, total_carbs_g, total_fat_g
        }

    Unit handling:
        - `quantity` may be in gram, cup, piece, slice, bowl, etc.
        - We call `convert_to_grams` (from Task 2) using the USDA portions.
        - If conversion fails, that food is reported as unmatched — we do NOT
          raise, so the Agent can decide how to handle it.
    """
    if len(foods) != len(food_nutrition_data):
        raise ValueError(
            f"Number of foods ({len(foods)}) must match "
            f"number of nutrition data ({len(food_nutrition_data)})"
        )

    total_calories = 0.0
    total_protein = 0.0
    total_carbs = 0.0
    total_fat = 0.0

    food_results: List[dict] = []
    unmatched_foods: List[dict] = []

    for meal_food, nutrition_data in zip(foods, food_nutrition_data):
        original_name = meal_food.get("food_name_original", "unknown")

        # 1) Did Task 2 succeed for this food?
        if not nutrition_data.get("success", False):
            unmatched_foods.append({
                "food_name_original": original_name,
                "reason": nutrition_data.get("error", "unknown_error"),
            })
            continue

        # 2) Verify Task 1 / Task 2 are talking about the same food
        returned_query = (nutrition_data.get("original_query") or "").strip().lower()
        if returned_query != original_name.strip().lower():
            unmatched_foods.append({
                "food_name_original": original_name,
                "reason": "food_mismatch",
                "details": {
                    "task1_name": original_name,
                    "task2_original_query": nutrition_data.get("original_query"),
                },
            })
            continue

        # 3) Convert quantity → grams using USDA portions
        quantity = meal_food.get("quantity")
        unit = meal_food.get("unit", "gram")
        portions = nutrition_data.get("portions", [])

        conversion = convert_to_grams(quantity, unit, portions)

        if conversion is None:
            unmatched_foods.append({
                "food_name_original": original_name,
                "reason": "unit_conversion_failed",
                "details": {
                    "quantity": quantity,
                    "unit": unit,
                    "portions_available": len(portions),
                },
            })
            continue

        quantity_g = conversion["grams"]

        # 4) Scale nutrition to grams
        result = calculate_food_nutrition(nutrition_data, quantity_g)
        result["food_name_original"] = original_name
        result["unit_used"] = unit
        result["quantity_input"] = quantity
        result["conversion_source"] = conversion["source"]

        food_results.append(result)

        total_calories += result["calories"]
        total_protein += result["protein_g"]
        total_carbs += result["carbs_g"]
        total_fat += result["fat_g"]

    return {
        "success": len(unmatched_foods) == 0,
        "foods": food_results,
        "unmatched_foods": unmatched_foods,
        "total_calories": round(total_calories, 2),
        "total_protein_g": round(total_protein, 2),
        "total_carbs_g": round(total_carbs, 2),
        "total_fat_g": round(total_fat, 2),
    }


def validate_daily_nutrition(
    meals: List[dict],
    profile: dict,
    calorie_tolerance: float = 0.20,
    fat_tolerance: float = 0.20,
    carbs_tolerance: float = 0.20,
) -> dict:
    """
    Validate a full day of meals against the user's targets and restrictions.

    Args:
        meals: list of `calculate_meal_total` outputs.
        profile: Task 1 profile (goal, weight, restrictions, ...).
        calorie_tolerance: ±% around target calories (default ±20%).
        fat_tolerance: ±% around target fat grams (default ±20%).
        carbs_tolerance: ±% around target carbs grams (default ±20%).

    Returns:
        {
          valid: bool,
          daily_total: {total_calories, total_protein_g, total_carbs_g, total_fat_g},
          targets: {target_calories, protein_g, carbs_g, fat_g},
          checks: {calories, protein, fat, carbs, restrictions},
          issues: [str, ...]
        }
    """
    targets = calculate_macros(profile)
    target_calories = targets["calories"]

    total_calories = 0.0
    total_protein = 0.0
    total_carbs = 0.0
    total_fat = 0.0

    all_foods: List[dict] = []
    issues: List[str] = []

    # ── Aggregate ─────────────────────────────────────────────────────────
    for meal in meals:
        if not meal.get("success", False):
            unmatched = meal.get("unmatched_foods", [])
            if unmatched:
                for u in unmatched:
                    name = u.get("food_name_original", "unknown")
                    reason = u.get("reason", "unknown_error")
                    issues.append(f"Could not compute nutrition for '{name}' ({reason})")
            else:
                issues.append("One of the meals could not be calculated")
            continue

        total_calories += meal["total_calories"]
        total_protein += meal["total_protein_g"]
        total_carbs += meal["total_carbs_g"]
        total_fat += meal["total_fat_g"]
        all_foods.extend(meal["foods"])

    # ── Calorie check ─────────────────────────────────────────────────────
    cal_min = target_calories * (1.0 - calorie_tolerance)
    cal_max = target_calories * (1.0 + calorie_tolerance)
    calories_valid = cal_min <= total_calories <= cal_max
    if not calories_valid:
        issues.append(
            f"Daily calories {round(total_calories, 2)} are outside "
            f"the target range ({round(cal_min, 2)}–{round(cal_max, 2)} kcal)"
        )

    # ── Protein check (minimum only) ──────────────────────────────────────
    protein_min = targets["protein_g"]
    protein_valid = total_protein >= protein_min
    if not protein_valid:
        issues.append(
            f"Daily protein {round(total_protein, 2)}g is below "
            f"the minimum target ({protein_min}g)"
        )

    # ── Fat check ─────────────────────────────────────────────────────────
    fat_min = targets["fat_g"] * (1.0 - fat_tolerance)
    fat_max = targets["fat_g"] * (1.0 + fat_tolerance)
    fat_valid = fat_min <= total_fat <= fat_max
    if not fat_valid:
        issues.append(
            f"Daily fat {round(total_fat, 2)}g is outside "
            f"the target range ({round(fat_min, 2)}–{round(fat_max, 2)} g)"
        )

    # ── Carbs check ───────────────────────────────────────────────────────
    carbs_min = targets["carbs_g"] * (1.0 - carbs_tolerance)
    carbs_max = targets["carbs_g"] * (1.0 + carbs_tolerance)
    carbs_valid = carbs_min <= total_carbs <= carbs_max
    if not carbs_valid:
        issues.append(
            f"Daily carbohydrates {round(total_carbs, 2)}g are outside "
            f"the target range ({round(carbs_min, 2)}–{round(carbs_max, 2)} g)"
        )

    # ── Restriction check (forbidden_foods + allergies + dietary_restrictions) ──
    user_restrictions = _collect_user_restrictions(profile)
    restriction_violations: List[dict] = []

    if user_restrictions:
        for food in all_foods:
            matched = _check_food_against_restrictions(food, user_restrictions)
            if matched:
                violation = {
                    "food_name_original": food.get("food_name_original"),
                    "food_name": food.get("food_name"),
                    "matched_restrictions": sorted(matched),
                }
                restriction_violations.append(violation)
                issues.append(
                    f"Food '{violation['food_name_original']}' violates "
                    f"restriction(s): {', '.join(violation['matched_restrictions'])}"
                )

    restrictions_valid = len(restriction_violations) == 0

    return {
        "valid": len(issues) == 0,
        "daily_total": {
            "total_calories": round(total_calories, 2),
            "total_protein_g": round(total_protein, 2),
            "total_carbs_g": round(total_carbs, 2),
            "total_fat_g": round(total_fat, 2),
        },
        "targets": {
            "target_calories": target_calories,
            "protein_g": targets["protein_g"],
            "carbs_g": targets["carbs_g"],
            "fat_g": targets["fat_g"],
        },
        "checks": {
            "calories": calories_valid,
            "protein": protein_valid,
            "fat": fat_valid,
            "carbs": carbs_valid,
            "restrictions": restrictions_valid,
        },
        "restriction_violations": restriction_violations,
        "issues": issues,
    }


def validate_meal_nutrition(
    meal: dict,
    profile: dict,
    target_calories: Optional[float] = None,
    calorie_tolerance: float = 0.15,
) -> dict:
    """
    Validate a SINGLE meal.

    Use case: "Make me a 400-calorie breakfast without eggs" → Task 4 builds
    the meal, then calls this to check it before returning to the user.

    Args:
        meal: output of `calculate_meal_total`.
        profile: Task 1 profile (for restrictions).
        target_calories: optional per-meal calorie budget (e.g. 400).
                         If None, calorie check is skipped.
        calorie_tolerance: ±% around target_calories (default ±15%).

    Returns:
        {
          valid: bool,
          meal_total: {total_calories, total_protein_g, total_carbs_g, total_fat_g},
          checks: {calories, restrictions},
          restriction_violations: [...],
          issues: [str, ...]
        }
    """
    issues: List[str] = []

    if not meal.get("success", False):
        unmatched = meal.get("unmatched_foods", [])
        if unmatched:
            for u in unmatched:
                name = u.get("food_name_original", "unknown")
                reason = u.get("reason", "unknown_error")
                issues.append(f"Could not compute nutrition for '{name}' ({reason})")
        else:
            issues.append("Meal could not be calculated")
        return {
            "valid": False,
            "meal_total": None,
            "checks": {"calories": False, "restrictions": False},
            "restriction_violations": [],
            "issues": issues,
        }

    meal_total = {
        "total_calories": meal["total_calories"],
        "total_protein_g": meal["total_protein_g"],
        "total_carbs_g": meal["total_carbs_g"],
        "total_fat_g": meal["total_fat_g"],
    }

    # ── Calorie check (only if target_calories was given) ─────────────────
    if target_calories is not None:
        cal_min = target_calories * (1.0 - calorie_tolerance)
        cal_max = target_calories * (1.0 + calorie_tolerance)
        calories_valid = cal_min <= meal["total_calories"] <= cal_max

        if not calories_valid:
            issues.append(
                f"Meal calories {round(meal['total_calories'], 2)} are outside "
                f"the target range ({round(cal_min, 2)}–{round(cal_max, 2)} kcal)"
            )
    else:
        calories_valid = True

    # ── Restriction check ─────────────────────────────────────────────────
    user_restrictions = _collect_user_restrictions(profile)
    restriction_violations: List[dict] = []

    if user_restrictions:
        for food in meal.get("foods", []):
            matched = _check_food_against_restrictions(food, user_restrictions)
            if matched:
                violation = {
                    "food_name_original": food.get("food_name_original"),
                    "food_name": food.get("food_name"),
                    "matched_restrictions": sorted(matched),
                }
                restriction_violations.append(violation)
                issues.append(
                    f"Food '{violation['food_name_original']}' violates "
                    f"restriction(s): {', '.join(violation['matched_restrictions'])}"
                )

    restrictions_valid = len(restriction_violations) == 0

    return {
        "valid": len(issues) == 0,
        "meal_total": meal_total,
        "checks": {
            "calories": calories_valid,
            "restrictions": restrictions_valid,
        },
        "restriction_violations": restriction_violations,
        "issues": issues,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Self-test — run with: python nutrition_engine.py
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json
    import sys
    import io

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )

    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8",
        errors="replace", line_buffering=True, write_through=True,
    )

    print("=" * 70)
    print("Task 3 — Nutrition Engine self-test")
    print("=" * 70)

    # ── Demo profile (what Task 1 would produce after onboarding) ─────────
    profile = {
        "birth_date": "2004-05-10",
        "gender": "male",
        "weight": 90.0,
        "height": 175.0,
        "activity_level": "moderate",
        "goal": "weight_loss",
        "dietary_restrictions": [],
        "forbidden_foods": ["بيض"],      # Arabic — should match "whole eggs"
        "preferred_foods": [],
        "allergies": ["dairy"],
    }

    # 1) Targets
    print("\n[1] Targets for weight_loss, male, 90kg")
    print(f"  Age      : {calculate_age(profile)}")
    print(f"  BMR      : {calculate_bmr(profile)} kcal")
    print(f"  TDEE     : {calculate_tdee(profile)} kcal")
    print(f"  Target   : {calculate_target_calories(profile)} kcal")
    print(f"  Macros   : {json.dumps(calculate_macros(profile))}")

    # 2) Macro sanity across goals
    print("\n[2] Target calories across goals (same profile)")
    for goal in ("weight_loss", "maintenance", "weight_gain", "muscle_gain"):
        p = {**profile, "goal": goal}
        print(f"  {goal:15s} → {calculate_target_calories(p):>8.2f} kcal")

    # 3) Food scaling
    print("\n[3] calculate_food_nutrition (200g grilled chicken)")
    chicken_usda = {
        "success": True,
        "original_query": "فراخ مشوية",
        "food_name": "Chicken breast, grilled",
        "calories_per_100g": 165,
        "protein_per_100g": 31,
        "carbs_per_100g": 0,
        "fat_per_100g": 3.6,
        "portions": [
            {"amount": 1, "unit": "piece", "modifier": None, "gram_weight": 120},
        ],
    }
    scaled = calculate_food_nutrition(chicken_usda, 200)
    print(f"  {json.dumps(scaled, ensure_ascii=False)}")

    # 4) Meal total with mixed units
    print("\n[4] calculate_meal_total with mixed units")
    foods = [
        {"food_name_original": "فراخ مشوية", "quantity": 200, "unit": "gram"},
        {"food_name_original": "رز أبيض",    "quantity": 1,   "unit": "cup"},
    ]
    rice_usda = {
        "success": True,
        "original_query": "رز أبيض",
        "food_name": "Rice, white, cooked",
        "calories_per_100g": 130,
        "protein_per_100g": 2.7,
        "carbs_per_100g": 28,
        "fat_per_100g": 0.3,
        "portions": [
            {"amount": 1, "unit": "cup", "modifier": None, "gram_weight": 158},
        ],
    }
    meal = calculate_meal_total(foods, [chicken_usda, rice_usda])
    print(f"  Totals: {meal['total_calories']} kcal, "
          f"{meal['total_protein_g']}g P, "
          f"{meal['total_carbs_g']}g C, "
          f"{meal['total_fat_g']}g F")

    # 5) Daily validation
    print("\n[5] validate_daily_nutrition (this single meal as a day)")
    day = validate_daily_nutrition([meal], profile)
    print(f"  valid: {day['valid']}")
    print(f"  checks: {json.dumps(day['checks'])}")
    for issue in day["issues"]:
        print(f"    - {issue}")

    # 6) Restriction detection (Arabic "بيض" vs "dairy" allergy)
    print("\n[6] Restriction detection — feed eggs + yogurt")
    eggs_usda = {
        "success": True,
        "original_query": "بيض",
        "food_name": "Egg, whole, cooked",
        "calories_per_100g": 155,
        "protein_per_100g": 13,
        "carbs_per_100g": 1.1,
        "fat_per_100g": 11,
        "portions": [],
    }
    yogurt_usda = {
        "success": True,
        "original_query": "زبادي",
        "food_name": "Yogurt, plain",
        "calories_per_100g": 61,
        "protein_per_100g": 3.5,
        "carbs_per_100g": 4.7,
        "fat_per_100g": 3.3,
        "portions": [],
    }
    bad_foods = [
        {"food_name_original": "بيض",   "quantity": 100, "unit": "gram"},
        {"food_name_original": "زبادي", "quantity": 100, "unit": "gram"},
    ]
    bad_meal = calculate_meal_total(bad_foods, [eggs_usda, yogurt_usda])
    bad_check = validate_meal_nutrition(bad_meal, profile, target_calories=300)
    print(f"  valid: {bad_check['valid']}")
    for v in bad_check["restriction_violations"]:
        print(f"    - {v['food_name_original']} → {v['matched_restrictions']}")

    # 7) Unit conversion failure path
    print("\n[7] Unit conversion failure (no portions for 'piece')")
    no_portions = {**chicken_usda, "portions": []}
    fail_foods = [{"food_name_original": "فراخ مشوية", "quantity": 2, "unit": "piece"}]
    fail_meal = calculate_meal_total(fail_foods, [no_portions])
    print(f"  success: {fail_meal['success']}")
    for u in fail_meal["unmatched_foods"]:
        print(f"    - {u['food_name_original']}: {u['reason']} "
              f"(portions_available={u['details']['portions_available']})")

    print("\nDone.")
