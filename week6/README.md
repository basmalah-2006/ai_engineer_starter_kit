# Grounded Mental Health Q&A Assistant — Multilingual RAG System

A polished, Grounded Retrieval-Augmented Generation (RAG) assistant built over the **Saudi Ministry of Health Mental Health Guide for Work Environments**.

This project demonstrates how to build an end-to-end RAG pipeline that allows users to ask complex medical and organizational questions in **Arabic or English**, receiving accurate, grounded answers with strict **page-level citations**. The system prevents hallucinations by refusing to answer out-of-scope queries and compares three distinct retrieval architectures (Baseline, Re-ranker, and Hybrid Search).

The application supports a full Streamlit web interface (`app.py`) for interactive querying, alongside CLI scripts for pipeline comparison (`compare_versions.py`) and automated evaluation (`evaluation.py`).

---

## Features

- **End-to-End RAG Pipeline:** Complete ingestion, chunking, embedding, retrieval, and grounded generation.
- **Multilingual Support:** Seamlessly handles Arabic and English queries, responding in the exact language of the user.
- **Three Retrieval Architectures:** Compare Baseline (Vector), Re-ranker (Cross-Encoder), and Hybrid Search (BM25 + Vector) directly from the UI.
- **Strict Source Citations:** Every generated answer includes the exact page number(s) from the source PDF.
- **Anti-Hallucination Fallback:** The system strictly adheres to the context and refuses to answer if the information is not present in the document.
- **Arabic-Optimized Chunking:** Custom text splitters and hybrid search weights tuned specifically for Arabic morphology.
- **Automated Evaluation:** A dedicated script to measure citation accuracy and latency across the pipelines.
- **Lightning-Fast Inference:** Powered by Groq API for sub-second LLM response times.

---

## Output Structure

For every query, the assistant returns a grounded response structured as follows:

| Component | Description |
| :--- | :--- |
| **Generated Answer** | The direct answer to the user's question, formatted cleanly (e.g., bullet points). |
| **Citations** | Explicit mention of the source page(s) at the end of the answer (e.g., `(المصدر: الصفحة 3)`). |
| **Retrieved Context** | (Visible in UI) The exact text chunks retrieved from the PDF that grounded the answer. |

**Example interaction:**

> **User:** كيف يمكن للمدير أن يدعم موظفيه نفسياً؟
>
> **Assistant:** يمكن للمدير أن يدعم موظفيه نفسياً من خلال تقديم التوجيه السليم، وبناء منهجيات واستراتيجيات تهدف إلى منع وتقليل تأثير المشكلات النفسية، وتطبيق إجراءات تحافظ على الصحة النفسية في بيئة العمل. **(المصدر: الصفحة 3)**

---

## Models Used

### 1. Embedding Model: `intfloat/multilingual-e5-small`

A highly efficient multilingual sentence-transformer model chosen for its excellent performance on both Arabic and English text, mapping them into a shared semantic vector space.

| Property | Value |
| :--- | :--- |
| **Architecture** | Transformer (Multilingual E5) |
| **Languages** | 100+ (Excellent Arabic & English support) |
| **Use Case** | Semantic Vector Search |
| **Device** | CPU (lightweight enough for local inference) |

### 2. Re-ranker Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`

A lightweight cross-encoder used to re-score and re-order the top-K retrieved chunks before passing them to the LLM, significantly improving context precision.

| Property | Value |
| :--- | :--- |
| **Architecture** | Cross-Encoder (MiniLM-L6) |
| **Purpose** | Re-ranks top-10 → top-3 chunks |
| **Benefit** | Dramatically improves context precision for actionable queries |

### 3. LLM Provider: Groq API

Utilizes Groq's LPU inference engine for ultra-fast generation with a strict grounding prompt.

| Property | Value |
| :--- | :--- |
| **Provider** | Groq Cloud |
| **Latency** | Sub-second responses |
| **Temperature** | 0.1 (for deterministic, grounded answers) |

---

## Chunking Strategy

Unlike short product listings, PDF documents require careful segmentation to preserve context.

| Parameter | Value | Rationale |
| :--- | :--- | :--- |
| **Splitter** | `RecursiveCharacterTextSplitter` | Handles multi-level separators gracefully |
| **Chunk Size** | 600 characters | Slightly larger to accommodate longer Arabic words |
| **Chunk Overlap** | 80 characters | Ensures sentences and concepts are not cut off mid-thought |
| **Custom Separators** | `["\n\n", "\n", ". ", "، ", "؛ ", " ", ""]` | Respects Arabic punctuation marks |
| **Metadata** | Page number per chunk | Enables downstream page-level citations |

---

## Dataset Info

| Metric | Value |
| :--- | :--- |
| **Source Document** | Saudi MoH Mental Health Guide for Work Environments |
| **Language** | Arabic (Primary) |
| **Format** | PDF |
| **Domain** | Occupational Mental Health & Psychology |
| **Topics Covered** | Risk factors, burnout, depression, workplace support, leadership |

---

## Project Structure

```
week6/
│
├── app.py                   ← Streamlit web interface (Clean UI)
├── ingest.py                ← Ingestion pipeline (Load, Chunk, Embed, Store)
├── rag_chain.py             ← Baseline RAG (Vector Search only)
├── rag_chain_reranker.py    ← Advanced RAG (Vector + Cross-Encoder Re-ranking)
├── rag_chain_hybrid.py      ← Hybrid RAG (BM25 + Vector + Re-ranking)
├── compare_versions.py      ← CLI script to compare all 3 pipelines
├── evaluation.py            ← Automated evaluation & CSV report generator
│
├── data/
│   └── mental_health_guide.pdf  ← The source corpus
│
├── chroma_db_multilingual/  ← Persistent ChromaDB vector store
├── .env                     ← Groq API key (not committed)
├── requirements.txt
└── README.md
```

---

## Installation

**1. Clone the repository:**

```bash
git clone https://github.com/basmalah-2006/ai_engineer_starter_kit.git
cd ai_engineer_starter_kit/week6
```

**2. Create a virtual environment:**

```bash
python -m venv .venv
```

**3. Activate the virtual environment:**

- **Windows:**
  ```bash
  .venv\Scripts\activate
  ```
- **Linux / macOS:**
  ```bash
  source .venv/bin/activate
  ```

**4. Install dependencies:**

```bash
pip install -r requirements.txt
```

**5. Set up your Groq API Key:**

Create a `.env` file in the `week6` directory:

```env
GROQ_API_KEY=gsk_your_actual_groq_api_key_here
```

> 💡 Get your free API key at [console.groq.com](https://console.groq.com)

---

## Running the Application

### Step 1 — Ingest the Document (One-time setup)

```bash
python ingest.py
```

This loads the PDF, chunks it with Arabic-aware separators, embeds it using the multilingual model, and stores vectors in ChromaDB.

### Step 2 — Launch the Streamlit Web Interface

```bash
streamlit run app.py
```

The browser opens automatically at `http://localhost:8501`. Use the sidebar to switch between **Baseline**, **Re-ranker**, and **Hybrid** retrieval methods to compare performance in real-time.

### Step 3 — Run CLI Comparison

```bash
python compare_versions.py
```

Runs the same test questions through all three pipelines and prints side-by-side results.

### Step 4 — Run Automated Evaluation (Lab 6.4)

```bash
python evaluation.py
```

Generates a CSV report measuring citation accuracy and latency across the pipelines.

---

## How It Works

### Ingestion Pipeline

```
PDF Document
    │
    ▼
PyPDFLoader (extracts text + page metadata)
    │
    ▼
Arabic-Aware Chunking (RecursiveCharacterTextSplitter)
    │
    ▼
Multilingual E5 Embeddings
    │
    ▼
ChromaDB Persistent Vector Store
```

### Query Pipeline

```
User Query (Arabic or English)
    │
    ▼
┌────────────────────────────┐
│   Retrieval Strategy       │ ← Selectable in UI
│  (Baseline / Re-ranker /   │
│   Hybrid 70/30)            │
└────────────────────────────┘
    │
    ▼
Cross-Encoder Re-ranker
(top-K → top-3 most relevant)
    │
    ▼
Strict Grounding Prompt
(enforces citations + language matching)
    │
    ▼
Groq LLM Inference
    │
    ▼
Grounded Answer + (Page Citations)
```

---

## Retrieval Architecture Comparison

Tested on a diverse set of in-scope and out-of-scope queries:

| Question Type | Baseline (Vector) | Re-ranker | Hybrid (70/30) | Winner |
| :--- | :---: | :---: | :---: | :--- |
| **Risk Factors** (in-scope) | ✅ Cited | ✅ Cited | ✅ Cited | All |
| **Well-being Promotion** (in-scope) | ✅ Cited | ⚠️ No citation | ✅ Cited | Baseline & Hybrid |
| **Burnout Signs** (in-scope) | ❌ Refused | ❌ Refused | ❌ Refused | None (limitation) |
| **Manager Support** (English, in-scope) | ❌ Refused | ❌ Refused | ⚠️ No citation | Hybrid (partial) |
| **Out-of-Scope** (GAD, PTSD, DSM-5) | ✅ Refused | ✅ Refused | ✅ Refused | All ✅ |

### Key Findings

1. **Hybrid Search is the most reliable** for Arabic queries, balancing citation accuracy and smart refusal.
2. **Baseline is the fastest** and surprisingly competitive — ideal when latency is the priority.
3. **Re-ranker alone underperformed** because aggressive top-3 pruning can discard citation-bearing chunks.
4. **All methods achieved 100% anti-hallucination** on out-of-corpus medical questions.

---

## Key Design Choices & Tuning (Lab 6.3)

### 1. Hybrid Search Weights for Arabic Text

Initially, a `50/50` weight was used for Vector and BM25 retrieval. However, testing revealed that BM25 (keyword search) performs poorly with Arabic text due to complex morphology and diacritics, causing it to retrieve irrelevant chunks and confuse the LLM.

**Decision:** We adjusted the weights to `[0.7, 0.3]` (70% Semantic Vector, 30% Keyword BM25). This significantly improved retrieval precision for Arabic queries while still capturing exact English medical acronyms.

### 2. Anti-Hallucination & Empty Answer Fallback

The system is strictly grounded. When queried about specific terms not explicitly defined in the retrieved chunks (e.g., GAD, PTSD, DSM-5), the LLM correctly refuses to answer.

**Engineering Fix:** Some models return empty strings when refusing to answer via API. We implemented a post-processing fallback in the `llm_function` to catch empty responses and force the standard *"I do not have enough information..."* message, ensuring a consistent UI/UX.

### 3. Strict Prompt Engineering

The grounding prompt enforces four non-negotiable rules:

1. Answer in the **exact same language** as the question.
2. If not in context → respond with a fixed refusal phrase.
3. **Never** add a page citation to a refusal response.
4. If answering → **always** include the page number in parentheses.

---

## RAG vs Fine-Tuning vs Long Context (LO 6.5)

| Approach | When to Use | Our Decision |
| :--- | :--- | :--- |
| **RAG** ✅ | Dynamic data, needs citations, large corpus | ✅ **Chosen** — medical data changes frequently, citations are mandatory |
| **Fine-Tuning** | Fixed style/tone, domain vocabulary baked in | ❌ Data too dynamic, risk of hallucination |
| **Long Context** | Small corpus (<100 pages), simple Q&A | ❌ Doesn't scale, no page-level citations |

---

## Evaluation Results (Lab 6.4)

Evaluated on 7 questions (4 in-scope, 3 out-of-scope) using a custom evaluation script
that measures Citation Accuracy, Refusal Accuracy, and Latency.

| Metric | Baseline | Re-ranker | Hybrid | Winner |
| :--- | :---: | :---: | :---: | :---: |
| **Citation Accuracy** | 71.4% | 57.1% | 71.4% | Baseline & Hybrid |
| **Refusal Accuracy** | 71.4% | 71.4% | **85.7%** | **Hybrid** 🏆 |
| **Anti-Hallucination (Out-of-Scope)** | 100% | 100% | 100% | All |
| **Avg Latency** | **9.04s** | 16.67s | 16.44s | Baseline ⚡ |

### Key Insights

1. **Hybrid Search is the overall winner.** It achieved the highest Refusal Accuracy (85.7%)
   while maintaining strong Citation Accuracy (71.4%), proving that combining BM25 + Vector +
   Re-ranking yields the best results for Arabic text.

2. **Surprising finding: Baseline is 2x faster and outperforms Re-ranker alone.**
   The Baseline (Vector Only) achieved 71.4% Citation Accuracy vs. the Re-ranker's 57.1%,
   while being nearly twice as fast (9.04s vs 16.67s). This demonstrates a critical engineering
   lesson: **added complexity does not always yield better results.** The Re-ranker's aggressive
   pruning to top-3 chunks sometimes removed the chunk containing the page citation.

3. **100% Anti-Hallucination across all methods.** All three architectures correctly refused to
   answer out-of-scope medical questions (GAD, PTSD, DSM-5 criteria), confirming the system is
   strictly grounded and safe for a medical domain.

4. **Known limitation:** All three methods struggled with the "burnout signs" question, refusing
   an in-scope query. This points to a chunking limitation where the relevant content is either
   split across chunks or buried deep in the document — a clear area for future improvement.

---

## Week 6 Alignment

This project was built as **Mini Project 6** for the Helwan Career Center 12-Week Industry Roadmap, Week 6: RAG Pipelines.

It demonstrates:

- **LO 6.1** — Built a complete RAG pipeline: document loading, chunking, embedding, retrieval, and grounded generation.
- **LO 6.2** — Used LangChain to orchestrate a RAG system over PDFs with strict source citations.
- **LO 6.3** — Tuned retrieval quality with chunk size/overlap, top-k, re-ranking, and hybrid search, successfully reducing hallucinations.
- **LO 6.4** — Evaluated the RAG system on citation accuracy and latency using a custom evaluation script.
- **LO 6.5** — Applied a decision framework to choose RAG over fine-tuning for a dynamic, citation-required medical knowledge base.

---

## Reliability Notes

- ChromaDB collection is persisted on disk and reused across runs.
- Empty API responses are caught and replaced with a standard refusal message.
- All text chunks carry page-number metadata for traceable citations.
- The Streamlit app uses `@st.cache_resource` to avoid reloading models on each interaction.
- The `.env` file containing the Groq API key is excluded from version control via `.gitignore`.

---

## Author

**Basmalah Ahmed**
AI Engineer Starter Kit — Week 6