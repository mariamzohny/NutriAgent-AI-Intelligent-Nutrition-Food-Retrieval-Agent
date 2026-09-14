# NutriAgent AI

NutriAgent AI is an AI-powered nutrition and food retrieval agent that understands Arabic and English food queries, searches USDA FoodData Central, ranks the most relevant food matches, and returns accurate nutritional information.

The system combines NLP, semantic search, Cross-Encoder re-ranking, food-query normalization, and USDA nutrition data retrieval in a single intelligent pipeline.

---

## Features

- Arabic and English food query support
- Egyptian Arabic food normalization
- Arabic-to-English Transformer translation
- Smart food-name mapping
- USDA FoodData Central API integration
- Semantic food retrieval using Sentence Transformers
- Cross-Encoder re-ranking
- Lexical and fuzzy matching
- Cooking-method detection
- Composite-food and processed-food penalties
- Automatic selection of the best food match
- Calories and macronutrient extraction
- Complete integration pipeline through a single agent function

---

## Technologies

- Python
- NLP
- Hugging Face Transformers
- Sentence Transformers
- USDA FoodData Central API
- Pandas
- PyTorch
---

## System Workflow

```text
User Food Query
      ↓
Arabic / English Input Detection
      ↓
Input Validation
      ↓
Text Cleaning & Preprocessing
      ↓
Dialect Normalization
      ↓
Food Query Normalization
      ↓
Direct Food Mapping
      ↓
Arabic → English Transformer Translation
      ↓
Post-Translation Mapping
      ↓
Canonical Food Query
      ↓
USDA FoodData Central API Search
      ↓
Retrieve Food Candidates
      ↓
Generic / Non-Branded Food Filtering
      ↓
Bi-Encoder Semantic Embedding
      ↓
Semantic Similarity Calculation
      ↓
Top-K Candidate Selection
      ↓
Cross-Encoder Re-Ranking
      ↓
Query Coverage Calculation
      ↓
Candidate Precision Calculation
      ↓
Lexical / Fuzzy Similarity
      ↓
Phrase Matching
      ↓
Leading Food Match
      ↓
Preparation Method Matching
      ↓
Preparation Conflict Detection
      ↓
Composite Dish Detection
      ↓
Food-Part Conflict Detection
      ↓
Modifier Conflict Detection
      ↓
Processed Food Detection
      ↓
Unrequested Modifier Detection
      ↓
Smart Penalties & Bonuses
      ↓
Final Match Score Calculation
      ↓
Candidate Re-Ranking
      ↓
Best USDA Food Candidate
      ↓
USDA Food Details Retrieval
      ↓
Nutrition Completeness Validation
      ↓
Calories Extraction
      ↓
Protein Extraction
      ↓
Carbohydrates Extraction
      ↓
Fat Extraction
      ↓
Search & Details Cache
      ↓
Integration Function
      ↓
Structured JSON Output
      ↓
Food Retrieval Agent Response
```


## Example

Input:

```python
integration("رز أبيض")

---
Output:
 Normalized Query: white rice
 Best Match: Rice, white, cooked
 Calories: 130 kcal
 Protein: 2.54 g
 Carbs: 29 g
 Fat: 0.37 g
