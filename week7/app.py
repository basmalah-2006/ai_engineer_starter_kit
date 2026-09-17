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
if not GEMINI_API_KEY:
    st.error("❌ GOOGLE_API_KEY not found. Please set it in your .env file or Streamlit secrets.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

MODEL_NAME = "gemini-3.5-flash-lite"  
MAX_TOOL_ITERATIONS = 4
MAX_RETRIES = 3


def search_course_materials(query: str) -> str:
    """Searches the AI course lecture notes (706 pages) for specific concepts."""
    try:
        context = rag.retrieve_context(str(query), top_k=3)
        return context if context.strip() else "No relevant information found in the course materials."
    except Exception as e:
        return json.dumps({"error": f"RAG search failed: {str(e)}"})


def calculate_average(grades: list) -> str:
    """Calculates the arithmetic mean of a list of numerical grades."""
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
    try:
        parts = response.candidates[0].content.parts
    except Exception:
        return None
    for part in parts:
        if getattr(part, "function_call", None) and part.function_call.name:
            return part.function_call
    return None


def response_text(response) -> str:
    try:
        parts = response.candidates[0].content.parts
    except Exception:
        return ""
    return "\n".join(p.text for p in parts if getattr(p, "text", ""))


def send_with_retry(send_fn, payload):
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


st.set_page_config(
    page_title="Smart Study Buddy",
    page_icon="🤖",
    layout="wide"
)

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
    
    if st.button("🗑️ Clear Long-term Memory"):
        st.session_state.memory.long_term_memory = {"facts": []}
        st.session_state.memory.save_long_term_memory()
        st.rerun()
    
    st.divider()
    
    st.subheader("💬 Session")
    if st.button("🔄 New Conversation"):
        st.session_state.chat_history = []
        st.session_state.memory.clear_short_term()
        st.rerun()
    
    st.divider()
    
    st.subheader("ℹ️ About")
    st.markdown("""
    **Smart Study Buddy**  
    Week 7 Mini Project
    
    Combines:
    - 🔧 Tool Calling
    - 🧠 Memory (Short + Long)
    - 📚 RAG (706 AI pages)
    - 🛡️ Safety & Error Handling
    """)


@st.cache_resource
def get_model():
    return genai.GenerativeModel(
        model_name=MODEL_NAME,
        tools=[search_course_materials, calculate_average],
        system_instruction="""
        You are a helpful and friendly AI Study Assistant for a university student.
        
        Rules:
        1. ALWAYS use the provided tools when the user asks for calculations or course-specific information.
        2. If the user shares a personal fact (name, major, goals), acknowledge it clearly.
        3. Keep answers concise and cite page numbers when using search_course_materials.
        4. Reply in the same language the user writes in (Arabic or English).
        """
    )


model = get_model()


st.title("🤖 Smart Study Buddy")
st.caption("An AI assistant that uses tools, memory, and your course materials to help you study.")

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🤖"):
        st.markdown(msg["content"])
        if msg.get("tool_calls"):
            with st.expander("🔧 See tool activity"):
                for tc in msg["tool_calls"]:
                    st.markdown(tc)


def run_agent_turn(user_input: str):
    """Runs the full agent loop (tool calling + memory) for a user message."""
    
    user_context = st.session_state.memory.get_long_term_context()
    
    gemini_history = []
    for m in st.session_state.chat_history:
        role = "user" if m["role"] == "user" else "model"
        gemini_history.append({"role": role, "parts": m["content"]})
    
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
            final_text = "I kept needing tools without reaching an answer. Please rephrase."
            break
        
        tool_name = fc.name
        tool_args = dict(fc.args)
        
        tool_activity.append(f"**⚙️ Calling:** `{tool_name}` with args `{tool_args}`")
        
        if tool_name in AVAILABLE_FUNCTIONS:
            try:
                result = AVAILABLE_FUNCTIONS[tool_name](**tool_args)
            except Exception as e:
                result = json.dumps({"error": str(e)})
        else:
            result = json.dumps({"error": f"Tool '{tool_name}' is not authorized."})
        
        tool_activity.append(f"**🔧 Result:** ```{result[:300]}```")
        
        message = genai.protos.Content(
            role="user",
            parts=[genai.protos.Part(
                function_response=genai.protos.FunctionResponse(
                    name=tool_name, response={"result": result}
                )
            )]
        )
    
    if final_text is None:
        final_text = "(no response)"
    
    st.session_state.memory.add_to_short_term("user", user_input)
    st.session_state.memory.add_to_short_term("model", final_text)
    
    lower = user_input.lower()
    if any(k in lower for k in ["my name is", "i study", "my major", "اسمي", "أدرس", "تخصصي"]):
        st.session_state.memory.add_fact(user_input)
    
    return final_text, tool_activity


if user_input := st.chat_input("Ask me anything about your AI course..."):
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)
    
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            try:
                final_text, tool_activity = run_agent_turn(user_input)
                st.markdown(final_text)
                if tool_activity:
                    with st.expander("🔧 See tool activity"):
                        for tc in tool_activity:
                            st.markdown(tc)
                
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": final_text,
                    "tool_calls": tool_activity
                })
                st.rerun()
            except Exception as e:
                st.error(f"Error: {str(e)}")