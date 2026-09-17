# 🤖 Smart Study Buddy — Conversational AI Agent with Tool Calling, Memory & RAG

A production-grade conversational assistant built over 706 pages of Helwan National University's CS212 (Introduction to Artificial Intelligence) lecture notes.

This project demonstrates how to transform a simple chatbot into an autonomous AI agent that **does things** — it decides *by itself* when to search its knowledge base, when to compute grade averages, remembers the user across sessions, and handles API errors gracefully. The system combines **Gemini function calling**, a **dual-layer memory system**, and **RAG with FAISS** into one coherent assistant.

The application supports a full **Streamlit web interface** (`app.py`) deployed live on Streamlit Cloud, alongside a CLI agent (`main.py`) for local use.

> 🌐 **Live Demo:** [studybuddy123.streamlit.app](https://studybuddy123.streamlit.app/) — try it instantly, no installation needed!

---

## Features

- **Autonomous Tool Calling:** The agent decides independently which tool to invoke (RAG search vs. grade calculator) based on the user's intent — no hardcoded routing.
- **Self-Retrying Retrieval:** In the demo below, the agent called the search tool *twice* on its own because it judged the first retrieval insufficient — emergent agentic behavior.
- **Dual-Layer Memory:** Rolling short-term window (last 10 messages) + persistent long-term facts (`user_memory.json`) that survive restarts.
- **Grounded RAG Answers:** Every course-related answer is retrieved from 706 pages of lecture PDFs and cited with exact page numbers.
- **Production-Grade Safety:** Tool allow-list, infinite-loop guard (max 4 iterations), retry-with-backoff on rate limits, and protobuf-safe input validation.
- **Bilingual Support:** Understands and replies in Arabic or English, responding in the exact language of the user.
- **Zero-Cost Embeddings:** RAG runs on a local HuggingFace model (`all-MiniLM-L6-v2`) — no paid embedding API, no quota surprises.
- **Live Deployment:** Full Streamlit web UI running free on Streamlit Community Cloud.

---

## Output Structure

For every user turn, the assistant produces a response structured as follows:

| Component | Description |
| --- | --- |
| **Final Answer** | The natural-language reply to the user, grounded in tools or memory. |
| **Tool Activity** *(visible in UI)* | A collapsible expander showing which tools were called, with what arguments, and what they returned. |
| **Source Citations** | Explicit mention of the source page(s) when using the RAG tool (e.g., *(Source: Pages 174–176, 270)*). |
| **Memory Updates** *(visible in sidebar)* | New user facts are automatically saved to long-term memory. |

### Example Interaction

```text
👤 You: Hello, my name is Basmalah and I study Computer Engineering.
💾 [Saved to long-term memory]
🤖 Assistant: Hello Basmalah! It's great to meet you. As a Computer Engineering student,
   how can I help you with your studies today?

👤 You: Briefly explain the A* Search algorithm.
⚙️  [Agent calling tool: 'search_course_materials']
🔧 [Tool Result]: --- [Source 1: Page 174] --- Search: Basic idea ...
⚙️  [Agent calling tool: 'search_course_materials']   ← agent re-searched on its own
🔧 [Tool Result]: --- [Source 1: Page 270] --- Informed Search [Heuristic Search] ...
🤖 Assistant: The **A* Search algorithm** is a widely used pathfinding and graph traversal
   algorithm in artificial intelligence.
   * Evaluation Function: f(n) = g(n) + h(n)
   * Optimality & Completeness: guaranteed to find the shortest path if the heuristic
     is admissible.
   *(Source: Pages 174–176, 270)*

👤 You: My quiz grades were 85, 90, and 95. What is my average?
⚙️  [Agent calling tool: 'calculate_average']
🔧 [Tool Result]: {"average": 90.0, "count": 3, "grades": [85.0, 90.0, 95.0]}
🤖 Assistant: Your average quiz grade is **90.0**! Great job, Basmalah!

👤 You: Do you remember my name and what I study?
🤖 Assistant: Yes, of course, Basmalah! You told me that your name is Basmalah
   and that you study Computer Engineering.
```

---

## Models Used

### 1. Embedding Model: `sentence-transformers/all-MiniLM-L6-v2`

A highly efficient local sentence-transformer chosen for zero API cost and CPU-friendly inference — ideal for large corpora without paid embedding quotas.

| Property | Value |
| --- | --- |
| Architecture | Sentence Transformer (MiniLM-L6) |
| Languages | English (primary), multilingual support |
| Use Case | Semantic vector search for RAG |
| Device | CPU (runs locally, no network needed) |
| Vector Dimension | 384 |

### 2. Vector Store: FAISS (Facebook AI Similarity Search)

A high-performance in-memory vector index persisted to disk. Chosen over ChromaDB for speed and deterministic retrieval on a fixed corpus.

| Property | Value |
| --- | --- |
| Provider | FAISS (via LangChain `langchain_community`) |
| Index Type | Flat L2 (exact similarity) |
| Persistence | Saved to `vector_store/index.faiss` + `index.pkl` |
| Top-k | 3 chunks per query |

### 3. LLM Provider: Google Gemini API (Flash-Lite)

Google's fast, free-tier model with native function calling support — chosen after discovering that older Gemini 1.5/2.5 models were retired for new API keys.

| Property | Value |
| --- | --- |
| Model | `gemini-3.5-flash-lite` |
| Provider | Google AI Studio (free tier) |
| Rate Limit | ~5 requests/minute (handled via retry-with-backoff) |
| Function Calling | Native (schemas auto-generated from docstrings) |

---

## Tool Definitions

Two real tools are exposed to the agent:

| Tool | Purpose | Parameters | Validation & Failure Mode |
| --- | --- | --- | --- |
| `search_course_materials` | Semantic search over 706 pages of CS212 AI lecture notes | `query: string` | Coerced with `str()`; returns explicit "not found" message on empty retrieval |
| `calculate_average` | Arithmetic mean of numerical grades | `grades: array<number>` | Coerced with `float()` (protobuf-safe); rejects empty lists; errors returned as JSON |

Schemas are **auto-generated by Gemini from Python docstrings + type hints**, keeping definitions and implementations in a single source of truth. A portable OpenAI-style schema is also provided in `tools.py` with an allow-list dispatcher:

```python
AVAILABLE_FUNCTIONS = {
    "search_course_materials": search_course_materials,
    "calculate_average": calculate_average,
}
# Any tool name outside this allow-list is rejected BEFORE execution (LO 7.5)
```

---

## Memory Architecture

| Layer | Storage | Scope | Injection Point |
| --- | --- | --- | --- |
| **Short-term** | In-memory rolling window (last 10 messages) | Current session | Passed as Gemini `chat history` |
| **Long-term** | `user_memory.json` (persistent fact list) | Across sessions & restarts | Injected into the `system_instruction` on **every turn** |

Facts are captured with a lightweight keyword heuristic (`my name is`, `i study`, `my major`, `اسمي`, `أدرس`, `تخصصي`…), deduplicated, and persisted immediately:

```json
{
  "facts": [
    "Hello, my name is Basmalah and I study Computer Engineering."
  ]
}
```

The 10-message cap on short-term memory prevents context overflow and keeps token costs bounded.

---

## Chunking Strategy

Unlike short product listings, PDF lecture slides require careful segmentation to preserve bullet-point context across page boundaries.

| Parameter | Value | Rationale |
| --- | --- | --- |
| Splitter | `RecursiveCharacterTextSplitter` | Handles multi-level separators gracefully |
| Chunk Size | 1000 characters | Dense lecture slides need larger context windows |
| Chunk Overlap | 200 characters | Ensures concepts spanning chunk boundaries aren't lost |
| Separators | `["\n\n", "\n", ". ", " ", ""]` | Respects paragraph and sentence structure |
| Metadata | `page`, `chunk_index` per chunk | Enables downstream page-level citations |

---

## Dataset Info

| Metric | Value |
| --- | --- |
| Source Document | CS212 — Introduction to Artificial Intelligence (HNU, Spring 2025) |
| Total Pages | 706 pages of lecture notes |
| Language | English (primary) |
| Format | PDF |
| Domain | Artificial Intelligence, Search, Knowledge Representation |
| Topics Covered | A* Search, Heuristics, Uninformed Search, Informed Search, CSPs, Logic |
| Chunks Produced | 738 chunks (with page metadata) |

---

## Project Structure

```
week7/
 │
 ├── app.py                    ← Streamlit web UI (deployed live)
 ├── main.py                   ← CLI agent loop
 ├── memory.py                 ← Short/long-term memory manager
 ├── rag.py                    ← Ingestion pipeline + FAISS retrieval
 ├── tools.py                  ← Portable schemas + allow-list dispatcher
 ├── test_tools.py             ← Independent tool unit tests
 ├── check_models.py           ← Live Gemini model discovery utility
 │
 ├── data/
 │   └── CS212-Intro.-to-Artificial-Intelligence-...pdf   ← 706-page source corpus
 │
 ├── vector_store/
 │   ├── index.faiss           ← Persisted FAISS embeddings
 │   └── index.pkl             ← Persisted chunk metadata
 │
 ├── user_memory.json          ← Persistent long-term facts
 ├── requirements.txt          ← Python dependencies
 ├── .gitignore
 └── README.md                 ← This file
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/basmalah-2006/ai_engineer_starter_kit.git
cd ai_engineer_starter_kit/week7
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

**Windows:**
```bash
.venv\Scripts\activate
```

**Linux / macOS:**
```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Set up your Google Gemini API Key

Create a `.env` file in the **repository root** (gitignored):

```env
GOOGLE_API_KEY=AIzaSy_your_actual_key_here
```

💡 Get your free API key at [Google AI Studio](https://aistudio.google.com/app/apikey)

---

## Running the Application

### Step 1 — Build the Knowledge Base (one-time setup)

```bash
python rag.py
```

This extracts 705 pages, splits them into 738 chunks, embeds them locally with `all-MiniLM-L6-v2`, and persists the FAISS index to `vector_store/`. Takes ~2 minutes on first run.

### Step 2 — Launch the CLI Agent

```bash
python main.py
```

Interact in the terminal. Type `exit` to quit. The agent uses tools, updates memory, and cites page numbers.

### Step 3 — Launch the Streamlit Web UI

```bash
streamlit run app.py
```

The browser opens automatically at `http://localhost:8501`. Features include:
- Modern chat interface with message history
- Expandable tool-activity panels for every response
- Sidebar memory panel showing long-term facts
- Buttons to clear memory or start a new conversation

### Step 4 — Run Tool Unit Tests (optional)

```bash
python test_tools.py
```

Independently validates calculation, search, and unauthorized-tool handling.

---

## How It Works

### Ingestion Pipeline (runs once)

```
CS212 Lecture PDF (706 pages)
        │
        ▼
pypdf extraction (+ page metadata)         → 705 text pages
        │
        ▼
RecursiveCharacterTextSplitter              → 738 chunks
        │
        ▼
sentence-transformers/all-MiniLM-L6-v2      → free, local embeddings
        │
        ▼
FAISS index (persisted: vector_store/)
```

### Agent Loop (runs on every user message)

```
User message
     │
     ▼
Gemini Flash-Lite (tools bound + long-term context injected)
     │
     ▼
 ┌─────────────────────────────┐
 │ function_call requested?    │
 └─────────────────────────────┘
     │ Yes                            │ No
     ▼                                ▼
Allow-list check                 Extract text parts
     │                                │
     ▼                                ▼
Execute tool (try/except)        Final answer
     │                                │
     ▼                                ▼
FunctionResponse (role="USER")   Update memory
     │                                │
     ▼                                ▼
Back to Gemini                   Render in UI
     │
     ▼
Loop (guarded: MAX_ITERATIONS = 4)
```

---

## Key Design Choices & Tuning (Lab 7.x)

### 1. Choosing Gemini over OpenAI

Originally planned to use OpenAI's API, but hit `credit_balance_exhausted` on the free tier mid-build. Evaluated alternatives:

| Provider | Pros | Cons | Verdict |
| --- | --- | --- | --- |
| OpenAI | Industry standard | Exhausted free credits | ❌ |
| Groq | Very fast | Less mature function calling | ⚠️ |
| **Gemini** | Generous free tier, native function calling, modern | Newer API (model names change) | ✅ Chosen |

**Engineering lesson:** always design for provider-agnostic tool schemas so you can swap LLMs without rewriting the agent.

### 2. Retry-with-Backoff for Rate Limits (LO 7.5)

Gemini's free tier enforces ~5 requests/minute. Naively, the agent would crash on the second tool call. Instead, we parse the `retry_delay` from the 429 payload and sleep exactly that duration:

```python
def send_with_retry(send_fn, payload):
    for attempt in range(MAX_RETRIES + 1):
        try:
            return send_fn(payload)
        except Exception as e:
            if "429" in str(e) and attempt < MAX_RETRIES:
                wait = parse_retry_seconds(e) or 15 * (attempt + 1)
                time.sleep(wait)
                continue
            raise
```

This makes the agent **resilient to transient API errors** — a production requirement, not a nice-to-have.

### 3. Protobuf Coercion for Function Arguments (LO 7.5)

Gemini's Python SDK returns `RepeatedComposite` objects for list arguments, which aren't JSON-serializable. Solution: coerce explicitly:

```python
grades = [float(g) for g in grades]  # protobuf → plain Python
```

Without this, `calculate_average` crashes with `Object of type RepeatedComposite is not JSON serializable`.

### 4. Infinite Loop Guard (LO 7.5)

Tool-calling agents can loop forever if the model keeps requesting tools without resolving to a final text answer. We cap iterations:

```python
MAX_TOOL_ITERATIONS = 4
for iteration in range(MAX_TOOL_ITERATIONS + 1):
    response = send_with_retry(chat.send_message, message)
    fc = extract_function_call(response)
    if fc is None:
        final_text = response_text(response)
        break
    if iteration == MAX_TOOL_ITERATIONS:
        final_text = "I kept needing tools without reaching an answer. Please rephrase."
        break
    # ... execute tool, loop
```

**Real-world demo:** when asked about A* Search, the agent called `search_course_materials` **twice** because it judged the first retrieval insufficient — then stopped. The guard prevented a third unnecessary call.

### 5. Fresh Long-Term Context on Every Turn

Initially, the model was cached with `@st.cache_resource` and the system instruction was frozen. This broke memory updates. Fix: rebuild the model (cheap) on every turn with the **latest** `user_context`:

```python
def build_model():
    user_context = st.session_state.memory.get_long_term_context()
    return genai.GenerativeModel(
        model_name=MODEL_NAME,
        tools=[...],
        system_instruction=f"... Current User Context: {user_context} ...",
    )
```

### 6. Safe Response Parsing

Gemini 3.x changed the role for `FunctionResponse` from `"function"` to `"USER"`, and accessing `.text` on tool-only responses raises `Could not convert part.function_call to text`. Both fixed via:

- Part-level scanning with `getattr(part, 'function_call', None)`
- Using `role="USER"` when sending tool results back
- Helper `response_text()` that joins only `p.text` parts

---

## Engineering Challenges & Solutions (the real debugging journey)

| Challenge | Root Cause | Fix |
| --- | --- | --- |
| `404 model not found` | Gemini 1.5/2.5 retired for new API keys | Wrote `check_models.py` to discover live models programmatically → switched to Flash-Lite |
| `400 Role 'function' not supported` | Gemini 3.x API role change | Send `FunctionResponse` with `role="USER"` |
| `Could not convert part.function_call to text` | `.text` accessed on tool-only responses | Part-level `function_call` scanning helpers |
| `429 quota exceeded (5 req/min)` | Free-tier rate limit | Retry-with-backoff parsing the server-suggested delay |
| OpenAI credits exhausted mid-build | Paid embedding API | Migrated embeddings to local HuggingFace (zero cost) |
| `RepeatedComposite is not JSON serializable` | Protobuf list types | `[float(g) for g in grades]` coercion |
| Windows paths break on Linux cloud | `r"week7\data\..."` backslashes | POSIX forward slashes for cross-platform deployment |

**Key lesson:** added complexity (retries, guards, coercion) is exactly what separates a *demo* from a *production system*.

---

## Week 7 Alignment

This project was built as **Mini Project 7** for the Helwan Career Center 12-Week Industry Roadmap, Week 7: Tool Calling, Memory & Conversational Systems.

It demonstrates:

- **LO 7.1** — Implemented tool/function calling: Gemini function calling with 2 real tools (`search_course_materials`, `calculate_average`) invoked autonomously by the model.
- **LO 7.2** — Designed clean tool schemas (auto-generated from Python docstrings + type hints) and parse tool-call requests/results reliably with safe part-level scanning.
- **LO 7.3** — Added both short-term memory (rolling 10-message window) and long-term memory (persistent `user_memory.json`) to the conversational system.
- **LO 7.4** — Built a multi-turn chatbot that combines RAG (FAISS over 706 pages) + tools + memory into one coherent assistant.
- **LO 7.5** — Handled errors, retries, and loops safely: allow-list authorization, `MAX_TOOL_ITERATIONS = 4` guard, retry-with-backoff on 429 rate limits, protobuf coercion, structured error JSON from every tool.

---

## Reliability Notes

- FAISS vector store is persisted on disk and reused across runs — no re-embedding needed.
- All tool executions are wrapped in `try/except` and return structured `{"error": ...}` JSON so the model can recover conversationally.
- Short-term memory is capped at 10 messages to prevent context overflow and bound token costs.
- Long-term memory is saved to `user_memory.json` immediately on every new fact, surviving crashes and restarts.
- The Streamlit app uses `@st.cache_resource` to load the FAISS index once per server process for sub-second retrieval.
- The `.env` file containing the Google API key is excluded from version control via `.gitignore`; cloud deployments read it from Streamlit Secrets.
- All text chunks carry page-number metadata for traceable citations.

---

## Author

**Basmalah Ahmed**  
AI Engineer Starter Kit — Week 7  