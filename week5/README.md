# Semantic Product Search Engine — Multilingual Vector Search

A Streamlit-based semantic search application that allows users to search for Moroccan marketplace products using natural language in Arabic, French, or English.

This project demonstrates how to build a multilingual semantic search system using sentence embeddings and a vector database, without relying on keyword matching. Queries are understood by meaning, not exact words.

The application supports two entry points: a command-line script for experimentation (`project.py`) and a full Streamlit web interface (`app.py`).

## Features

* Multilingual semantic search supporting Arabic, French, and English queries.
* Uses sentence embeddings to understand query meaning, not just keywords.
* ChromaDB vector database for fast similarity retrieval.
* Category and price filters combinable with semantic queries.
* Streamlit web interface with interactive controls.
* CLI script with four ready-made search experiments.
* Dataset of 4,350 real Moroccan marketplace listings across 45 categories and 16 regions.
* Cosine similarity scoring via ChromaDB's built-in HNSW index.

## Search Result Structure

For every query, the engine returns the top-K most semantically similar products, each displayed with:

| Field            | Description                                                      |
| ---------------- | ---------------------------------------------------------------- |
| `Product Name`   | Name of the matched product                                      |
| `Full Text`      | Combined description used for embedding (name + category + city) |
| `City`           | Local address of the listing                                     |
| `Price`          | Listed price in Moroccan Dirhams (DH)                            |
| `Category`       | Product category                                                 |
| `Region`         | Region of the listing                                            |
| `Seller Type`    | Whether the seller is a professional or individual               |

Example search and result:

```text
Query: "gaming laptop powerful"

1. PC Gamer | Informatique  | Casablanca Bouskoura
   📍 Bouskoura | 💰 12000 DH

2. Ordinateur Portable Gamer | Informatique  | Rabat Hassan
   📍 Hassan | 💰 9500 DH
```

## Embedding Model

**paraphrase-multilingual-MiniLM-L12-v2**

A compact multilingual sentence-transformer model that maps text from 50+ languages into a shared vector space of 384 dimensions. Chosen because the dataset contains a mix of French, Arabic, and English product listings.

| Property         | Value                                      |
| ---------------- | ------------------------------------------ |
| Architecture     | MiniLM-L12 (Transformer)                  |
| Embedding size   | 384 dimensions                             |
| Languages        | 50+ including Arabic, French, English      |
| Model size       | ~118 MB                                    |
| Source           | `sentence-transformers` library            |

## Chunk Size

Since the dataset consists of short product listings, each product is embedded as a single text entry and no additional chunking is required. For longer documents, chunking into smaller sections with a suitable overlap would help preserve context and improve retrieval quality.

## Dataset Info

| Metric              | Value                           |
| ------------------- | ------------------------------- |
| Total listings      | 4,350                           |
| Product categories  | 45                              |
| Regions covered     | 16                              |
| Language of data    | French (primary) + Arabic       |
| File format         | CSV (latin-1 encoded)           |

## Project Structure

```text
week5/
│
├── app.py               ← Streamlit web interface
├── project.py           ← CLI script with search experiments
├── ProductsData.csv     ← Moroccan marketplace dataset
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/basmalah-2006/ai_engineer_starter_kit.git
cd ai_engineer_starter_kit/week5
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the virtual environment:

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

> **Note:** `sentence-transformers` will automatically download the embedding model on first run. Ensure you have an internet connection the first time.

## Running the Application

### Option 1 — Streamlit Web Interface

```bash
streamlit run app.py
```

The browser will open automatically at:

```text
http://localhost:8501
```

Use the search bar to enter a query in any language, apply category and price filters, and click **Search**.

### Option 2 — CLI Experiments

```bash
python project.py
```

This runs four pre-built search experiments:

```text
🔍 Experiment 1: Search for 'iPhone phone'
🔍 Experiment 2: Search for 'apartment in Casablanca' with category filter
🔍 Experiment 3: Search for 'gaming laptop powerful'
🔍 Experiment 4: Search for 'economic car' with max price 80,000 DH
```

## How It Works

1. The dataset is loaded from `ProductsData.csv` and cleaned.
2. Each product's name, category, and location are combined into a single text string.
3. The multilingual embedding model converts all product texts into 384-dimensional vectors.
4. All vectors are stored in a ChromaDB in-memory collection along with product metadata.
5. At query time, the user's query is embedded using the same model.
6. ChromaDB retrieves the top-K most similar products using cosine similarity.
7. Optional filters on category or price are applied at the ChromaDB query level.
8. Results are displayed with metadata in the Streamlit interface.

```text
User Query (any language)
        │
        ▼
Sentence Transformer
  (paraphrase-multilingual-MiniLM-L12-v2)
        │
        ▼
Query Vector (384 dims)
        │
        ▼
ChromaDB Cosine Similarity Search
  + optional category/price filter
        │
        ▼
Top-K Matching Products + Metadata
```

## Semantic Search vs Keyword Search

| Feature                    | Keyword Search     | Semantic Search (this project) |
| -------------------------- | ------------------ | ------------------------------ |
| Matches exact words        | ✅                 | ✅                              |
| Understands synonyms       | ❌                 | ✅                              |
| Cross-language queries     | ❌                 | ✅                              |
| Finds conceptually similar | ❌                 | ✅                              |
| Example: "laptop" → "PC"  | ❌ no match        | ✅ matched                      |

## Filtering

Filters can be combined with semantic queries at the ChromaDB level using metadata conditions:

```python
# Category filter only
semantic_search("téléphone", category_filter="Téléphones ")

# Price filter only
semantic_search("voiture", max_price=80000)

# Combined filter
semantic_search("appartement", category_filter="Appartements ", max_price=500000)
```

Both `$eq` (exact match) and `$lte` (less than or equal) operators are used through ChromaDB's `where` filter API.

## Prompt Engineering Techniques Used

While this project does not use a language model for generation, it applies embedding-level search engineering techniques:

* **Text enrichment:** Product name, category, and location are concatenated before embedding, giving the model richer context than the name alone.
* **Multilingual model selection:** The model was chosen specifically to handle the mixed French/Arabic nature of the dataset.
* **Metadata-level filtering:** Filters are applied post-embedding at the database level rather than post-retrieval, making filtering exact and efficient.
* **Query–document symmetry:** The same model encodes both the query and the stored documents, ensuring the vector space is shared.

## Reliability Notes

* ChromaDB collection is re-created on each run to avoid stale embeddings.
* Missing or `nan` prices are stored as `"0"` to keep metadata types consistent.
* All text columns are stripped of extra quotation marks before embedding.
* The Streamlit app uses `@st.cache_data` and `@st.cache_resource` to avoid reloading the model or re-embedding the dataset on each interaction.
* Empty queries are rejected before triggering a search.

## Week 5 Alignment

This project was built as **Mini Project 5** for the **Helwan Career Center 12-Week Industry Roadmap**, Week 5: **Embeddings & Vector Search**.

It demonstrates:

* **LO 5.1** — Generated dense vector embeddings from product text using a pretrained sentence-transformer model.
* **LO 5.2** — Stored and indexed embeddings in ChromaDB and understood the role of vector databases in semantic retrieval.
* **LO 5.3** — Implemented cosine similarity search and understood how distance in embedding space reflects semantic meaning.
* **LO 5.4** — Selected a multilingual embedding model appropriate to the language diversity of the dataset.
* **LO 5.5** — Combined semantic search with structured metadata filtering to support real-world search constraints such as price and category.

## Author

**Basmalah Ahmed**  
AI Engineer Starter Kit — Week 5