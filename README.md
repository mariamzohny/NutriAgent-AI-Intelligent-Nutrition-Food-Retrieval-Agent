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

## System Workflow

```text
User Food Query
      ↓
Input Validation
      ↓
Food Query Normalization
      ↓
Arabic-English Translation / Mapping
      ↓
USDA Food Search
      ↓
Semantic Retrieval
      ↓
Cross-Encoder Re-ranking
      ↓
Smart Matching & Validation
      ↓
Nutrition Data Extraction
      ↓
Final Agent Response

```mermaid
flowchart LR

    subgraph A["1. Query Understanding"]
        A1["Arabic / English Food Query"]
        A2["Text Cleaning"]
        A3["Dialect Normalization"]
        A4["Direct Food Mapping"]
        A5["Transformer Translation"]
        A6["Canonical Query"]

        A1 --> A2 --> A3 --> A4 --> A5 --> A6
    end

    subgraph B["2. Food Retrieval"]
        B1["USDA FoodData Central API"]
        B2["Generic Food Filtering"]
        B3["Bi-Encoder Semantic Retrieval"]
        B4["Top-K Candidates"]

        B1 --> B2 --> B3 --> B4
    end

    subgraph C["3. Intelligent Re-Ranking"]
        C1["Cross-Encoder Score"]
        C2["Semantic Similarity"]
        C3["Query Coverage"]
        C4["Candidate Precision"]
        C5["Lexical / Fuzzy Similarity"]
        C6["Phrase Matching"]
        C7["Preparation Matching"]
        C8["Smart Penalties & Bonuses"]
        C9["Final Match Score"]

        C1 --> C9
        C2 --> C9
        C3 --> C9
        C4 --> C9
        C5 --> C9
        C6 --> C9
        C7 --> C9
        C8 --> C9
    end

    subgraph D["4. Validation & Nutrition"]
        D1["Best Ranked Candidate"]
        D2["USDA Food Details"]
        D3["Nutrition Completeness Check"]
        D4["Calories"]
        D5["Protein"]
        D6["Carbs"]
        D7["Fat"]

        D1 --> D2 --> D3
        D3 --> D4
        D3 --> D5
        D3 --> D6
        D3 --> D7
    end

    subgraph E["5. Agent Output"]
        E1["Search & Details Cache"]
        E2["integration()"]
        E3["Structured JSON Output"]
        E4["Food Retrieval Agent"]

        E1 --> E2 --> E3 --> E4
    end

    A --> B --> C --> D --> E

    style A fill:#EAF4FF,stroke:#5B9BD5,stroke-width:2px
    style B fill:#EAFBF0,stroke:#5CB85C,stroke-width:2px
    style C fill:#FFF6E5,stroke:#F0AD4E,stroke-width:2px
    style D fill:#F3EEFF,stroke:#8A6DDB,stroke-width:2px
    style E fill:#FFEAF2,stroke:#D9539F,stroke-width:2px

    style C9 fill:#FFD9D9,stroke:#D9534F,stroke-width:2px
    style E4 fill:#FFD6E7,stroke:#C2185B,stroke-width:2px
```




---

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
