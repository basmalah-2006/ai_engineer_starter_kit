import os
import re
import json
import time
import warnings

import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv

from memory import MemoryManager
import rag

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

load_dotenv()

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GEMINI_API_KEY and "GOOGLE_API_KEY" in st.secrets:
    GEMINI_API_KEY = st.secrets["GOOGLE_API_KEY"]

if not GEMINI_API_KEY:
    st.error("❌ GOOGLE_API_KEY not found. Set it in .env (local) or Streamlit Secrets (cloud).")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

MODEL_NAME = "gemini-3.5-flash-lite"
MAX_TOOL_ITERATIONS = 4
MAX_RETRIES = 3


@st.cache_resource(show_spinner="Loading knowledge base (706 pages)...")
def get_vector_store():
    """Loads the FAISS vector store once and reuses it for all queries."""
    try:
        return rag.load_vector_store()
    except FileNotFoundError:
        return None


def retrieve_context_fast(query: str, top_k: int = 3) -> str:
    """Retrieves the most relevant lecture chunks with page citations."""
    vs = get_vector_store()
    if vs is None:
        return "Knowledge base not built yet. Run: python week7/rag.py"
    docs = vs.similarity_search(query, k=top_k)
    context = ""
    for i, doc in enumerate(docs, 1):
        page = doc.metadata.get("page", "?")
        context += f"\n--- [Source {i}: Page {page}] ---\n{doc.page_content}\n"
    return context


def search_course_materials(query: str) -> str:
    """Searches the AI course lecture notes (706 pages) for specific concepts, definitions, or algorithms."""
    try:
        context = retrieve_context_fast(str(query), top_k=3)
        return context if context.strip() else "No relevant information found in the course materials."
    except Exception as e:
        return json.dumps({"error": f"RAG search failed: {str(e)}"})


def calculate_average(grades: list) -> str:
    """Calculates the arithmetic mean of a list of numerical grades. Use when the user provides grades and asks for the average."""
    try:
        grades = [float(g) for g in grades]   
        if len(grades) == 0:
            return json.dumps({"error": "Grades list is empty."})
        avg = sum(grades) / len(grades)
        return json.dumps({"average": round(avg, 2), "count": len(grades), "grades": grades})
    except Exception as e:
        return json.dumps({"error": f"Calculation failed: {str(e)}"})


AVAILABLE_FUNCTIONS = {
    "search_course_materials": search_course_materials,
    "calculate_average": calculate_average,
}


def extract_function_call(response):
    """Safely returns the function_call part if the model requested a tool, else None."""
    try:
        parts = response.candidates[0].content.parts
    except Exception:
        return None
    for part in parts:
        if getattr(part, "function_call", None) and part.function_call.name:
            return part.function_call
    return None


def response_text(response) -> str:
    """Safely extracts text from a response without touching function_call parts."""
    try:
        parts = response.candidates[0].content.parts
    except Exception:
        return ""
    return "\n".join(p.text for p in parts if getattr(p, "text", ""))


def send_with_retry(send_fn, payload):
    """Sends a message; on 429 quota errors waits the suggested delay and retries."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            return send_fn(payload)
        except Exception as e:
            msg = str(e)
            if ("429" in msg or "quota" in msg.lower()) and attempt < MAX_RETRIES:
                m = re.search(r"seconds:\s*(\d+)", msg)
                wait = min(int(m.group(1)) + 2, 65) if m else 15 * (attempt + 1)
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("Max retries exceeded.")


st.set_page_config(page_title="Smart Study Buddy", page_icon="🤖", layout="wide")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "memory" not in st.session_state:
    st.session_state.memory = MemoryManager()


with st.sidebar:
    st.title("🧠 Memory Panel")

    st.subheader("📌 Long-term Memory")
    facts = st.session_state.memory.long_term_memory.get("facts", [])
    if facts:
        for i, fact in enumerate(facts, 1):
            st.markdown(f"{i}. {fact}")
    else:
        st.info("No user facts stored yet. Tell the bot your name or major!")

    if st.button("🗑️ Clear Long-term Memory", use_container_width=True):
        st.session_state.memory.long_term_memory = {"facts": []}
        st.session_state.memory.save_long_term_memory()
        st.rerun()

    st.divider()

    if st.button("🔄 New Conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.memory.clear_short_term()
        st.rerun()

    st.divider()

    st.subheader("ℹ️ About")
    st.markdown(
        """
        **Smart Study Buddy** — Week 7 Mini Project

        - 🔧 Tool Calling (Gemini function calling)
        - 🧠 Memory (short + long term)
        - 📚 RAG over 706 AI lecture pages
        - 🛡️ Allow-list, retries & loop guards
        """
    )
    st.caption("Free-tier note: the app auto-waits and retries on rate limits.")


def build_model():
    """Builds the model with the LATEST long-term memory injected into the system prompt."""
    user_context = st.session_state.memory.get_long_term_context()
    return genai.GenerativeModel(
        model_name=MODEL_NAME,
        tools=[search_course_materials, calculate_average],
        system_instruction=f"""
        You are a helpful and friendly AI Study Assistant for a university student.
        Current User Context: {user_context}

        Rules:
        1. ALWAYS use the provided tools when the user asks for calculations or course-specific information.
        2. If the user shares a personal fact (name, major, goals), acknowledge it clearly.
        3. Keep answers concise and cite page numbers when using search_course_materials.
        4. If a tool returns an error, explain it politely and suggest an alternative.
        5. Reply in the same language the user writes in (Arabic or English).
        """,
    )


st.title("🤖 Smart Study Buddy")
st.caption("An AI assistant that uses tools, memory, and your course materials to help you study.")

if st.session_state.pop("just_saved", False):
    st.toast("💾 Saved to long-term memory", icon="🧠")

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🤖"):
        st.markdown(msg["content"])
        if msg.get("tool_calls"):
            with st.expander("🔧 See tool activity"):
                for tc in msg["tool_calls"]:
                    st.markdown(tc)


def run_agent_turn(user_input: str):
    """Runs the full agent loop (tool calling + memory) for one user message."""
    model = build_model()

    gemini_history = [
        {"role": ("user" if m["role"] == "user" else "model"), "parts": m["content"]}
        for m in st.session_state.chat_history
    ]
    chat = model.start_chat(history=gemini_history)

    tool_activity = []
    final_text = None
    message = user_input

    for iteration in range(MAX_TOOL_ITERATIONS + 1):
        response = send_with_retry(chat.send_message, message)
        fc = extract_function_call(response)

        if fc is None:
            final_text = response_text(response)
            break

        if iteration == MAX_TOOL_ITERATIONS:
            final_text = ("I kept needing tools without reaching an answer "
                          "(loop guard triggered). Please rephrase.")
            break

        tool_name = fc.name
        tool_args = dict(fc.args)
        tool_activity.append(f"**⚙️ Calling:** `{tool_name}` with args `{tool_args}`")

        if tool_name in AVAILABLE_FUNCTIONS:
            try:
                result = AVAILABLE_FUNCTIONS[tool_name](**tool_args)
            except Exception as e:
                result = json.dumps({"error": f"Tool execution failed: {str(e)}"})
        else:
            result = json.dumps({"error": f"Tool '{tool_name}' is not authorized."})

        tool_activity.append(f"**🔧 Result:** `{result[:300]}`")

        message = genai.protos.Content(
            role="USER",
            parts=[genai.protos.Part(
                function_response=genai.protos.FunctionResponse(
                    name=tool_name, response={"result": result}
                )
            )],
        )

    if final_text is None:
        final_text = "(no response)"

    st.session_state.memory.add_to_short_term("user", user_input)
    st.session_state.memory.add_to_short_term("model", final_text)

    saved_fact = False
    lower = user_input.lower()
    if any(k in lower for k in ["my name is", "i study", "my major", "اسمي", "أدرس", "تخصصي"]):
        st.session_state.memory.add_fact(user_input)
        saved_fact = True

    return final_text, tool_activity, saved_fact


if user_input := st.chat_input("Ask me anything about your AI course..."):
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            try:
                final_text, tool_activity, saved_fact = run_agent_turn(user_input)

                if saved_fact:
                    st.session_state.just_saved = True

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": final_text,
                    "tool_calls": tool_activity,
                })
                st.rerun()
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                st.caption("The free tier allows ~5 requests/minute — please wait a moment and try again.")