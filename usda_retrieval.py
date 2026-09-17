# -*- coding: utf-8 -*-
"""
usda_retrieval.py — Task 2: USDA FoodData Central retrieval & food matching.

Public API:
    get_food_nutrition(food_name, original_query=None, ...) → nutrition dict
    search_food_usda(food_name, max_results=50)             → search hits
    get_food_details(fdc_id)                                → per-100g + portions
    normalize_food_query(food_text)                         → translation info
    convert_to_grams(quantity, unit, portions)              → {grams, source, ...}
    clear_caches()                                          → explicit reset

Environment:
    USDA_API_KEY   (required)
    USDA_BASE_URL  (optional, defaults to production)
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
import torch
from dotenv import load_dotenv
from rapidfuzz import fuzz

from nltk.stem import SnowballStemmer

logger = logging.getLogger(__name__)

# ─── Environment ─────────────────────────────────────────────────────────────
load_dotenv()

API_KEY = os.environ.get("USDA_API_KEY")
if not API_KEY:
    raise ValueError(
        "USDA_API_KEY environment variable is not set. "
        "Set it in a .env file or via the shell — never hardcode it."
    )

BASE_URL = os.environ.get("USDA_BASE_URL", "https://api.nal.usda.gov/fdc/v1")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─── Caches ──────────────────────────────────────────────────────────────────
SEARCH_CACHE: Dict[Tuple[str, int], List[dict]] = {}
DETAILS_CACHE: Dict[int, dict] = {}
TRANSLATION_CACHE: Dict[str, str] = {}


def clear_caches() -> None:
    """Explicit cache reset — call only from tests or a /reset endpoint."""
    SEARCH_CACHE.clear()
    DETAILS_CACHE.clear()
    TRANSLATION_CACHE.clear()
    logger.info("Caches cleared.")


# ─── Lazy model loading ──────────────────────────────────────────────────────
TRANSLATION_MODEL_NAME = "Helsinki-NLP/opus-mt-ar-en"
BI_ENCODER_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_MODELS: Dict[str, Any] = {}


def _ensure_models() -> None:
    """Load heavy models on first use only."""
    if _MODELS:
        return

    from transformers import MarianMTModel, MarianTokenizer
    from sentence_transformers import SentenceTransformer, CrossEncoder, util

    logger.info("Loading NLP models on device=%s ...", DEVICE)

    _MODELS["translation_tokenizer"] = MarianTokenizer.from_pretrained(TRANSLATION_MODEL_NAME)
    _MODELS["translation_model"] = MarianMTModel.from_pretrained(TRANSLATION_MODEL_NAME).to(DEVICE)
    _MODELS["bi_encoder"] = SentenceTransformer(BI_ENCODER_MODEL_NAME, device=DEVICE)
    _MODELS["cross_encoder"] = CrossEncoder(RERANKER_MODEL_NAME, device=DEVICE)
    _MODELS["util"] = util

    logger.info("NLP models loaded.")


# ─── Text utilities ──────────────────────────────────────────────────────────
def clean_food_text(text):
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text.strip())


def contains_arabic(text):
    return bool(re.search(r"[\u0600-\u06FF]", text))


def _normalize_lookup_key(text):
    text = clean_food_text(text).lower()
    text = re.sub(r"[\u064B-\u065F\u0670\u06D6-\u06ED]", "", text)  # strip diacritics
    text = text.replace("ـ", "")                                     # strip tatweel
    text = re.sub(r"[إأآٱ]", "ا", text)                              # unify alef
    text = text.replace("ى", "ي")                                    # unify yaa
    text = re.sub(r"[^\w\s-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ─── Food mapping ────────────────────────────────────────────────────────────
FOOD_QUERY_MAP_RAW = {
    # Arabic / Egyptian
    "فراخ مشوية": "grilled chicken",
    "فراخ مشوي": "grilled chicken",
    "دجاج مشوي": "grilled chicken",
    "دجاج مشوية": "grilled chicken",

    "رز أبيض": "white rice",
    "رز ابيض": "white rice",
    "أرز أبيض": "white rice",
    "ارز ابيض": "white rice",

    "عيش بلدي": "pita bread",
    "خبز بلدي": "pita bread",
    "خبز عربي": "pita bread",

    "بطاطس مسلوقة": "boiled potatoes",
    "بطاطا مسلوقة": "boiled potatoes",

    "بيض": "whole eggs",
    "زبادي": "plain yogurt",

    # English
    "egg": "whole eggs",
    "eggs": "whole eggs",
    "banana": "raw banana",
    "bananas": "raw banana",
    "salmon": "raw salmon",
    "yogurt": "plain yogurt",
    "yoghurt": "plain yogurt",
    "oatmeal": "cooked oatmeal",

    # Common translation outputs
    "arab bread": "pita bread",
    "arabic bread": "pita bread",
    "boiled potato": "boiled potatoes",
    "potato boiled": "boiled potatoes",
    "plain yoghurt": "plain yogurt",
}

FOOD_QUERY_MAP = {_normalize_lookup_key(k): v for k, v in FOOD_QUERY_MAP_RAW.items()}


def apply_food_mapping(text):
    return FOOD_QUERY_MAP.get(_normalize_lookup_key(text))


DIALECT_ALIASES = {
    "عيش بلدي": "خبز عربي",
    "فراخ": "دجاج",
    "رز": "أرز",
    "بطاطس": "بطاطا",
}


def normalize_dialect(text):
    for source, target in DIALECT_ALIASES.items():
        text = text.replace(source, target)
    return text


# ─── Arabic → English translation ────────────────────────────────────────────
def translate_arabic_to_english(text: str) -> str:
    if not text:
        return ""

    cache_key = text.strip()
    if cache_key in TRANSLATION_CACHE:
        return TRANSLATION_CACHE[cache_key]

    _ensure_models()

    tok = _MODELS["translation_tokenizer"]
    mdl = _MODELS["translation_model"]

    inputs = tok(text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        translated_tokens = mdl.generate(**inputs, max_length=100)

    translation = tok.decode(translated_tokens[0], skip_special_tokens=True)
    translation = clean_food_text(translation).lower()

    TRANSLATION_CACHE[cache_key] = translation
    return translation


# ─── Query normalization ─────────────────────────────────────────────────────
def normalize_food_query(food_text):
    """Returns {original_food, normalized_food, method}."""
    original_text = food_text
    cleaned_text = clean_food_text(food_text)

    if not cleaned_text:
        return {"original_food": original_text, "normalized_food": "", "method": "empty"}

    # 1) Direct mapping
    mapped_food = apply_food_mapping(cleaned_text)
    if mapped_food:
        english_food = mapped_food
        method = "direct_food_mapping"

    # 2) Arabic path
    elif contains_arabic(cleaned_text):
        dialect_normalized = normalize_dialect(cleaned_text)
        mapped_after_dialect = apply_food_mapping(dialect_normalized)

        if mapped_after_dialect:
            english_food = mapped_after_dialect
            method = "dialect_normalization+food_mapping"
        else:
            english_food = translate_arabic_to_english(dialect_normalized)
            method = "dialect_normalization+transformer_translation"

    # 3) English path
    else:
        english_food = cleaned_text.lower()
        method = "already_english"

    english_food = re.sub(r"[^\w\s-]", " ", english_food)
    english_food = re.sub(r"\s+", " ", english_food).strip().lower()

    post_mapping = apply_food_mapping(english_food)
    if post_mapping and post_mapping != english_food:
        english_food = post_mapping
        method = method + "+post_mapping"

    return {
        "original_food": original_text,
        "normalized_food": english_food,
        "method": method,
    }


# ─── USDA search ─────────────────────────────────────────────────────────────
def search_food_usda(food_name, max_results=50):
    """USDA search. Returns [{fdc_id, description, data_type}]."""
    if not food_name:
        return []

    cache_key = (food_name.strip().lower(), max_results)
    if cache_key in SEARCH_CACHE:
        logger.debug("Search cache hit: %s", food_name)
        return SEARCH_CACHE[cache_key]

    url = f"{BASE_URL}/foods/search"
    params = {"api_key": API_KEY, "query": food_name, "pageSize": max_results}

    try:
        response = requests.get(url, params=params, timeout=10)

        if response.status_code == 403:
            raise RuntimeError("Invalid USDA API key.")
        if response.status_code == 429:
            raise RuntimeError("USDA API rate limit exceeded.")

        response.raise_for_status()
        data = response.json()

        results = []
        for food in data.get("foods", []):
            description = food.get("description")
            if not description:
                continue
            results.append({
                "fdc_id": food.get("fdcId"),
                "description": description,
                "data_type": food.get("dataType"),
            })

        SEARCH_CACHE[cache_key] = results
        return results

    except requests.exceptions.Timeout:
        logger.warning("USDA search timed out for query=%r", food_name)
        return []
    except requests.exceptions.ConnectionError:
        logger.warning("Could not connect to USDA for query=%r", food_name)
        return []
    except Exception as e:
        logger.error("USDA Search Error for query=%r: %s", food_name, e)
        return []


# ─── USDA details ────────────────────────────────────────────────────────────
_USDA_NUTRIENT_IDS = {
    "calories": (1008, 2047, 2048),
    "protein": (1003,),
    "fat": (1004,),
    "carbs": (1005,),
}


def _first_present(values: Dict[int, float], ids: Tuple[int, ...]) -> Optional[float]:
    """First non-None value (fixes the `x or y` bug when x == 0)."""
    for nid in ids:
        v = values.get(nid)
        if v is not None:
            return v
    return None


def _extract_portions(data: dict) -> List[dict]:
    """USDA foodPortions → [{amount, unit, modifier, gram_weight}]."""
    portions: List[dict] = []
    for p in (data.get("foodPortions") or []):
        unit_obj = p.get("measureUnit") or {}
        unit_name = (unit_obj.get("name") or unit_obj.get("abbreviation") or "").strip().lower()
        gw = p.get("gramWeight")
        amt = p.get("amount")
        if not gw or not amt:
            continue
        portions.append({
            "amount": float(amt),
            "unit": unit_name,
            "modifier": (p.get("modifier") or "").strip().lower() or None,
            "gram_weight": float(gw),
        })
    return portions


def get_food_details(fdc_id: int) -> Optional[dict]:
    """Per-100g nutrition + household portions for a USDA food."""
    if fdc_id in DETAILS_CACHE:
        logger.debug("Details cache hit: %s", fdc_id)
        return DETAILS_CACHE[fdc_id]

    url = f"{BASE_URL}/food/{fdc_id}"
    params = {"api_key": API_KEY}

    try:
        response = requests.get(url, params=params, timeout=10)

        if response.status_code == 404:
            return None
        if response.status_code == 403:
            raise RuntimeError("Invalid USDA API key.")
        if response.status_code == 429:
            raise RuntimeError("USDA API rate limit exceeded.")

        response.raise_for_status()
        data = response.json()

        nutrient_values: Dict[int, float] = {}
        for item in data.get("foodNutrients", []):
            nutrient = item.get("nutrient") or {}
            nid = nutrient.get("id")
            amt = item.get("amount")
            if nid is not None and amt is not None:
                nutrient_values[nid] = amt

        result = {
            "fdc_id": data.get("fdcId"),
            "food_name": data.get("description"),
            "calories_per_100g": _first_present(nutrient_values, _USDA_NUTRIENT_IDS["calories"]),
            "protein_per_100g": _first_present(nutrient_values, _USDA_NUTRIENT_IDS["protein"]),
            "carbs_per_100g": _first_present(nutrient_values, _USDA_NUTRIENT_IDS["carbs"]),
            "fat_per_100g": _first_present(nutrient_values, _USDA_NUTRIENT_IDS["fat"]),
            "portions": _extract_portions(data),
        }

        DETAILS_CACHE[fdc_id] = result
        return result

    except requests.exceptions.Timeout:
        logger.warning("USDA details timed out for fdc_id=%s", fdc_id)
        return None
    except requests.exceptions.ConnectionError:
        logger.warning("Could not connect to USDA for fdc_id=%s", fdc_id)
        return None
    except Exception as e:
        logger.error("USDA Details Error for fdc_id=%s: %s", fdc_id, e)
        return None


def nutrition_is_complete(details) -> bool:
    if not details:
        return False
    required = (
        details.get("calories_per_100g"),
        details.get("protein_per_100g"),
        details.get("carbs_per_100g"),
        details.get("fat_per_100g"),
    )
    return all(v is not None for v in required)


# ─── Stage 1: semantic retrieval ─────────────────────────────────────────────
def semantic_retrieve(
    query: str,
    results: List[dict],
    top_k: int = 20,
    prefer_generic: bool = True,
) -> List[dict]:
    if not results:
        return []

    candidates = results
    if prefer_generic:
        generic = [r for r in results if r.get("data_type") != "Branded"]
        if generic:
            candidates = generic

    _ensure_models()
    bi_encoder = _MODELS["bi_encoder"]
    util = _MODELS["util"]

    descriptions = [item["description"] for item in candidates]

    query_embedding = bi_encoder.encode(
        query, convert_to_tensor=True, normalize_embeddings=True
    )
    description_embeddings = bi_encoder.encode(
        descriptions, convert_to_tensor=True, normalize_embeddings=True
    )

    similarities = util.cos_sim(query_embedding, description_embeddings)[0]

    ranked_candidates = []
    for result, score in zip(candidates, similarities):
        item = result.copy()
        item["semantic_score"] = float(score)
        ranked_candidates.append(item)

    ranked_candidates.sort(key=lambda x: x["semantic_score"], reverse=True)
    return ranked_candidates[:top_k]


# ─── Stage 2: dynamic reranking ──────────────────────────────────────────────
stemmer = SnowballStemmer("english")

STOP_WORDS = {
    "the", "a", "an", "and", "of", "with", "as", "to", "in", "for", "from", "by",
    "grade", "nfs", "ns", "ingredient",
}


def normalize_rank_tokens(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [stemmer.stem(t) for t in text.split() if t not in STOP_WORDS]


def normalize_match_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


PREPARATION_TERMS = {
    "raw": {"raw", "uncooked"},
    "grilled": {"grilled", "grill", "broiled", "broil"},
    "boiled": {"boiled", "boil"},
    "baked": {"baked", "bake"},
    "fried": {"fried", "fry"},
    "roasted": {"roasted", "roast"},
    "sauteed": {"sauteed", "saute"},
    "stewed": {"stewed", "stew"},
    "steamed": {"steamed", "steam"},
    "cooked": {"cooked", "cook"},
}


def extract_preparations(text):
    words = set(normalize_match_text(text).split())
    detected = set()
    for preparation, variants in PREPARATION_TERMS.items():
        if words & variants:
            detected.add(preparation)
    return detected


def preparation_features(query, description):
    query_preps = extract_preparations(query)
    desc_preps = extract_preparations(description)

    if not query_preps or not desc_preps:
        return 0.0, 0.0

    if "cooked" in query_preps:
        cooked_methods = {
            "cooked", "grilled", "boiled", "baked", "fried",
            "roasted", "sauteed", "stewed", "steamed",
        }
        if desc_preps & cooked_methods:
            return 1.0, 0.0
        if "raw" in desc_preps:
            return 0.0, 1.0

    if query_preps & desc_preps:
        return 1.0, 0.0

    return 0.0, 1.0


COMPOSITE_DISH_WORDS = {
    "sandwich", "salad", "soup", "casserole", "pizza",
    "burrito", "wrap", "mixture", "meal", "dish", "lomi",
}


def composite_food_penalty(query, description):
    q_words = set(normalize_match_text(query).split())
    d_words = set(normalize_match_text(description).split())

    penalty = 0.0
    if "and" in d_words and "and" not in q_words:
        penalty = 1.0
    if "with" in d_words and "with" not in q_words:
        penalty = 1.0
    if (d_words & COMPOSITE_DISH_WORDS) - (q_words & COMPOSITE_DISH_WORDS):
        penalty = 1.0
    return penalty


def specific_part_penalty(query, description):
    q_words = set(normalize_match_text(query).split())
    d_words = set(normalize_match_text(description).split())

    penalty = 0.0

    # whole egg vs egg white / yolk
    if (
        ({"egg", "eggs"} & q_words)
        and ({"white", "yolk"} & d_words)
        and not ({"white", "yolk"} & q_words)
    ):
        penalty = 1.0

    if "skin" in d_words and "skin" not in q_words:
        penalty = max(penalty, 0.75)

    return penalty


def modifier_conflict_penalty(query, description):
    q_words = set(normalize_match_text(query).split())
    d_words = set(normalize_match_text(description).split())

    penalty = 0.0

    if "plain" in q_words:
        flavored_words = {
            "flavored", "flavoured", "vanilla",
            "strawberry", "chocolate", "fruit", "sweetened",
        }
        if d_words & flavored_words:
            penalty = 1.0

    if "rice" in q_words and "white" in q_words:
        if {"brown", "wild", "black"} & d_words:
            penalty = 1.0

    return penalty


PROCESSED_FOOD_WORDS = {
    "roll", "patty", "patties", "nugget", "nuggets",
    "luncheon", "deli", "breaded", "battered",
    "processed", "sausage", "frankfurter",
}


def processed_food_penalty(query, description):
    q_words = set(normalize_match_text(query).split())
    d_words = set(normalize_match_text(description).split())
    return 1.0 if (d_words & PROCESSED_FOOD_WORDS) - q_words else 0.0


def unrequested_modifier_penalty(query, description):
    q_text = normalize_match_text(query)
    d_text = normalize_match_text(description)
    q_words = set(q_text.split())

    modifier_phrases = [
        "nonfat", "non fat", "lowfat", "low fat",
        "fat free", "reduced fat", "skim",
        "sweetened", "flavored", "flavoured",
    ]
    for modifier in modifier_phrases:
        if modifier in d_text and modifier not in q_text:
            return 1.0

    if "yogurt" in q_words and "plain" in q_words:
        special_yogurt_terms = [
            "nonfat", "non fat", "lowfat", "low fat",
            "fat free", "reduced fat", "skim",
        ]
        if any(term in d_text for term in special_yogurt_terms):
            return 1.0

    return 0.0


def generic_preparation_bonus(query, description):
    q = normalize_match_text(query)
    d = normalize_match_text(description)

    if "oatmeal" in q and "oatmeal" in d:
        return 1.0
    if "whole eggs" in q and "egg" in d and "whole" in d:
        return 1.0
    if "salmon" in q and "salmon" in d:
        return 1.0
    return 0.0


def dynamic_rerank(query: str, candidates: List[dict]) -> List[dict]:
    if not candidates:
        return []

    _ensure_models()
    cross_encoder = _MODELS["cross_encoder"]

    query_token_list = normalize_rank_tokens(query)
    query_tokens = set(query_token_list)
    normalized_query = normalize_match_text(query)

    pairs = [[query, c["description"]] for c in candidates]
    cross_raw_scores = np.asarray(cross_encoder.predict(pairs), dtype=float).reshape(-1)
    cross_scores = 1.0 / (1.0 + np.exp(-np.clip(cross_raw_scores, -20, 20)))

    semantic_raw_scores = np.asarray(
        [c["semantic_score"] for c in candidates], dtype=float
    )
    semantic_scores = np.clip((semantic_raw_scores + 1.0) / 2.0, 0.0, 1.0)

    query_coverages, candidate_precisions, fuzzy_scores = [], [], []
    leading_matches, phrase_matches = [], []
    preparation_matches, preparation_conflicts = [], []
    composite_penalties, part_penalties, modifier_penalties = [], [], []
    processed_penalties, unrequested_modifier_penalties = [], []
    generic_preparation_bonuses = []

    for candidate in candidates:
        description = candidate["description"]
        description_token_list = normalize_rank_tokens(description)
        description_tokens = set(description_token_list)
        common_tokens = query_tokens & description_tokens

        query_coverages.append(len(common_tokens) / max(len(query_tokens), 1))
        candidate_precisions.append(len(common_tokens) / max(len(description_tokens), 1))
        fuzzy_scores.append(fuzz.token_sort_ratio(query.lower(), description.lower()) / 100.0)

        leading_matches.append(
            float(description_token_list[0] in query_tokens)
            if description_token_list else 0.0
        )
        phrase_matches.append(
            float(normalized_query in normalize_match_text(description))
        )

        prep_match, prep_conflict = preparation_features(query, description)
        preparation_matches.append(prep_match)
        preparation_conflicts.append(prep_conflict)

        composite_penalties.append(composite_food_penalty(query, description))
        part_penalties.append(specific_part_penalty(query, description))
        modifier_penalties.append(modifier_conflict_penalty(query, description))
        processed_penalties.append(processed_food_penalty(query, description))
        unrequested_modifier_penalties.append(unrequested_modifier_penalty(query, description))
        generic_preparation_bonuses.append(generic_preparation_bonus(query, description))

    final_results = []
    for i, candidate in enumerate(candidates):
        final_score = (
            0.32 * cross_scores[i]
            + 0.22 * semantic_scores[i]
            + 0.16 * query_coverages[i]
            + 0.10 * candidate_precisions[i]
            + 0.08 * fuzzy_scores[i]
            + 0.07 * leading_matches[i]
            + 0.05 * phrase_matches[i]
        )
        final_score += 0.10 * preparation_matches[i]
        final_score += 0.06 * generic_preparation_bonuses[i]

        final_score -= 0.22 * preparation_conflicts[i]
        final_score -= 0.20 * composite_penalties[i]
        final_score -= 0.18 * part_penalties[i]
        final_score -= 0.15 * modifier_penalties[i]
        final_score -= 0.20 * processed_penalties[i]
        final_score -= 0.10 * unrequested_modifier_penalties[i]

        final_score = float(np.clip(final_score, 0.0, 1.0))

        result = candidate.copy()
        result.update({
            "semantic_score": round(float(semantic_raw_scores[i]), 4),
            "cross_encoder_score": round(float(cross_raw_scores[i]), 4),
            "query_coverage": round(float(query_coverages[i]), 4),
            "candidate_precision": round(float(candidate_precisions[i]), 4),
            "lexical_score": round(float(fuzzy_scores[i]), 4),
            "leading_match": round(float(leading_matches[i]), 4),
            "phrase_match": round(float(phrase_matches[i]), 4),
            "preparation_match": round(float(preparation_matches[i]), 4),
            "preparation_conflict": round(float(preparation_conflicts[i]), 4),
            "composite_penalty": round(float(composite_penalties[i]), 4),
            "part_penalty": round(float(part_penalties[i]), 4),
            "modifier_penalty": round(float(modifier_penalties[i]), 4),
            "processed_penalty": round(float(processed_penalties[i]), 4),
            "unrequested_modifier_penalty": round(float(unrequested_modifier_penalties[i]), 4),
            "generic_preparation_bonus": round(float(generic_preparation_bonuses[i]), 4),
            "final_score": round(float(final_score), 4),
        })
        final_results.append(result)

    final_results.sort(key=lambda x: x["final_score"], reverse=True)
    return final_results


# ─── Main pipeline ───────────────────────────────────────────────────────────
def get_food_nutrition(food_name, original_query=None, search_k=50, rerank_k=20):
    """
    Full pipeline: normalize → search → semantic → rerank → fetch details.

    `original_query` lets the caller (Task 4) echo back the user-facing name
    (e.g. "فراخ مشوية") so Task 3's matching check still succeeds even though
    USDA is searched with the English name.
    """
    echoed_original = original_query if original_query is not None else food_name

    if not isinstance(food_name, str) or not food_name.strip():
        return {"success": False, "error": "empty_food_name"}

    normalized = normalize_food_query(food_name)
    query = normalized["normalized_food"]

    if not query:
        return {
            "success": False,
            "original_query": echoed_original,
            "error": "normalization_failed",
        }

    search_results = search_food_usda(query, max_results=search_k)
    if not search_results:
        return {
            "success": False,
            "original_query": echoed_original,
            "normalized_query": query,
            "error": "food_not_found",
        }

    semantic_candidates = semantic_retrieve(
        query, search_results, top_k=rerank_k, prefer_generic=True
    )
    if not semantic_candidates:
        return {
            "success": False,
            "original_query": echoed_original,
            "normalized_query": query,
            "error": "semantic_retrieval_failed",
        }

    ranked_results = dynamic_rerank(query, semantic_candidates)
    if not ranked_results:
        return {
            "success": False,
            "original_query": echoed_original,
            "normalized_query": query,
            "error": "ranking_failed",
        }

    selected_match = None
    nutrition = None
    selected_rank = None

    for rank, candidate in enumerate(ranked_results, start=1):
        candidate_details = get_food_details(candidate["fdc_id"])
        if nutrition_is_complete(candidate_details):
            selected_match = candidate
            nutrition = candidate_details
            selected_rank = rank
            break

    if selected_match is None:
        return {
            "success": False,
            "original_query": echoed_original,
            "normalized_query": query,
            "error": "complete_nutrition_not_found",
        }

    return {
        "success": True,
        "original_query": echoed_original,
        "normalized_query": query,
        "normalization_method": normalized["method"],

        "fdc_id": selected_match["fdc_id"],
        "food_name": selected_match["description"],
        "data_type": selected_match["data_type"],

        "selected_rank": selected_rank,
        "semantic_score": selected_match["semantic_score"],
        "cross_encoder_score": selected_match["cross_encoder_score"],
        "final_match_score": selected_match["final_score"],

        "calories_per_100g": nutrition["calories_per_100g"],
        "protein_per_100g": nutrition["protein_per_100g"],
        "carbs_per_100g": nutrition["carbs_per_100g"],
        "fat_per_100g": nutrition["fat_per_100g"],

        "portions": nutrition.get("portions", []),
        "source": "USDA FoodData Central",
    }


# ─── Unit conversion (for Task 3 / Task 4) ───────────────────────────────────
USDA_PORTION_UNIT_ALIASES: Dict[str, str] = {
    "cup": "cup", "cups": "cup",
    "tbsp": "tablespoon", "tablespoon": "tablespoon", "tablespoons": "tablespoon",
    "tsp": "teaspoon", "teaspoon": "teaspoon", "teaspoons": "teaspoon",
    "oz": "ounce", "ounce": "ounce", "ounces": "ounce",
    "slice": "slice", "slices": "slice",
    "piece": "piece", "pieces": "piece",
    "small": "piece", "medium": "piece", "large": "piece",
    "unit": "piece", "each": "piece",
    "can": "can", "bottle": "bottle", "loaf": "loaf", "bowl": "bowl",
    "serving": "serving",
}


def convert_to_grams(
    quantity: float,
    unit: str,
    portions: Optional[List[dict]],
) -> Optional[dict]:
    """
    Convert (quantity, unit) → grams using USDA portions when available.

    Returns:
        {"grams": float, "source": "direct"|"usda_portion", "matched_portion": dict|None}
        or None if conversion is not possible.
    """
    if quantity is None:
        return None

    u = (unit or "").strip().lower()

    if u in ("g", "gram", "grams"):
        return {"grams": float(quantity), "source": "direct", "matched_portion": None}
    if u in ("kg", "kilogram", "kilograms"):
        return {"grams": float(quantity) * 1000.0, "source": "direct", "matched_portion": None}

    if not portions:
        return None

    target = USDA_PORTION_UNIT_ALIASES.get(u, u)

    exact_matches = []
    fallback_matches = []
    for p in portions:
        p_unit = USDA_PORTION_UNIT_ALIASES.get(
            (p.get("unit") or "").lower(), (p.get("unit") or "").lower()
        )
        if p_unit != target:
            continue
        gw = p.get("gram_weight")
        amt = p.get("amount")
        if not gw or not amt:
            continue
        grams_per_unit = float(gw) / float(amt)
        if p.get("modifier") is None:
            exact_matches.append((grams_per_unit, p))
        else:
            fallback_matches.append((grams_per_unit, p))

    chosen = exact_matches[0] if exact_matches else (
        fallback_matches[0] if fallback_matches else None
    )
    if not chosen:
        return None

    grams_per_unit, matched = chosen
    return {
        "grams": round(grams_per_unit * float(quantity), 2),
        "source": "usda_portion",
        "matched_portion": matched,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Self-test — run with: python usda_retrieval.py
# ─────────────────────────────────────────────────────────────────────────────

def _inspect_food_matches(food_name, top_n=10):
    """Test helper: return top-N ranked USDA matches for a query."""
    import pandas as pd

    normalized = normalize_food_query(food_name)
    query = normalized["normalized_food"]

    search_results = search_food_usda(query, max_results=50)
    semantic_candidates = semantic_retrieve(query, search_results, top_k=20)
    ranked_results = dynamic_rerank(query, semantic_candidates)

    if not ranked_results:
        return pd.DataFrame()

    df = pd.DataFrame(ranked_results[:top_n])
    return df[[
        "fdc_id", "description", "data_type",
        "semantic_score", "cross_encoder_score",
        "query_coverage", "candidate_precision",
        "lexical_score", "final_score",
    ]]


if __name__ == "__main__":
    import inspect
    import pandas as pd

    # ─── Top candidates ──────────────────────────────────────────────────
    print(_inspect_food_matches("chicken breast"))
    print(_inspect_food_matches("eggs"))
    print(_inspect_food_matches("رز أبيض"))
    print(_inspect_food_matches("عيش بلدي"))

    # ─── Semantic retrieval signature ────────────────────────────────────
    print(inspect.signature(semantic_retrieve))

    check_results = semantic_retrieve(
        "chicken breast",
        search_food_usda("chicken breast", max_results=50),
        top_k=20,
        prefer_generic=True,
    )
    print("Branded candidates:", sum(
        item["data_type"] == "Branded" for item in check_results
    ))
    print(check_results[0])

    # ─── Error handling ──────────────────────────────────────────────────
    error_tests = ["", "     ", "asdasdasdasdasdfood999999"]
    for food in error_tests:
        print("Input:", repr(food))
        print(get_food_nutrition(food))
        print("-" * 70)

    # ─── Full flow ───────────────────────────────────────────────────────
    test_foods = [
        "chicken breast", "فراخ مشوية", "رز أبيض", "eggs", "عيش بلدي",
        "banana", "salmon", "بطاطس مسلوقة", "زبادي", "oatmeal",
    ]

    test_df = pd.DataFrame([get_food_nutrition(f) for f in test_foods])
    print(test_df)

    # ─── Normalization ───────────────────────────────────────────────────
    normalization_tests = [
        "chicken breast", "فراخ مشوية", "رز أبيض",
        "عيش بلدي", "بطاطس مسلوقة", "زبادي",
    ]
    normalization_df = pd.DataFrame([
        {"input": f, **normalize_food_query(f)} for f in normalization_tests
    ])
    print(normalization_df)

    # ─── Search + details ────────────────────────────────────────────────
    search_test = search_food_usda("grilled chicken", max_results=10)
    print(pd.DataFrame(search_test).head(10))
    print("Number of results:", len(search_test))

    if search_test:
        print(get_food_details(search_test[0]["fdc_id"]))

    # ─── Ranking comparison ──────────────────────────────────────────────
    for food in ["chicken breast", "eggs", "white rice", "grilled chicken", "pita bread"]:
        print("\n", "=" * 80)
        print("QUERY:", food)
        print(_inspect_food_matches(food, top_n=5))

    # ─── Full flow (again, with selected columns) ────────────────────────
    full_flow_df = pd.DataFrame([get_food_nutrition(f) for f in test_foods])
    print(full_flow_df[[
        "original_query", "normalized_query", "food_name",
        "semantic_score", "cross_encoder_score", "final_match_score",
        "calories_per_100g", "protein_per_100g", "carbs_per_100g", "fat_per_100g",
    ]])

    # ─── Cache behaviour ─────────────────────────────────────────────────
    SEARCH_CACHE.clear()
    DETAILS_CACHE.clear()
    print("Caches cleared.")

    print("FIRST CALL")
    get_food_nutrition("grilled chicken")

    print("\nSECOND CALL")
    get_food_nutrition("grilled chicken")

    print("\nSearch cache items:", len(SEARCH_CACHE))
    print("Details cache items:", len(DETAILS_CACHE))

    # ─── Save results ────────────────────────────────────────────────────
    test_df.to_csv("dynamic_usda_retrieval_results.csv", index=False)
    print("Results saved successfully!")
