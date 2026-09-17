import os
import re
import json
import time
import warnings

import google.generativeai as genai
from dotenv import load_dotenv

from memory import MemoryManager
import rag

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

load_dotenv()

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file.")

genai.configure(api_key=GEMINI_API_KEY)

MODEL_NAME = "gemini-3.5-flash-lite"  
MAX_TOOL_ITERATIONS = 4           
MAX_RETRIES = 3                  

memory = MemoryManager()


def search_course_materials(query: str) -> str:
    """Searches the AI course lecture notes (706 pages) for specific concepts, definitions, or algorithms."""
    try:
        query = str(query)
        context = rag.retrieve_context(query, top_k=3)
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
        return "(empty response)"
    texts = [p.text for p in parts if getattr(p, "text", "")]
    return "\n".join(texts) if texts else "(empty response)"


def send_with_retry(send_fn, payload):
    """Sends a message; on 429 quota errors waits the suggested delay and retries."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            return send_fn(payload)
        except Exception as e:
            msg = str(e)
            is_quota = ("429" in msg) or ("quota" in msg.lower())
            if is_quota and attempt < MAX_RETRIES:
                m = re.search(r"seconds:\s*(\d+)", msg)
                wait = min(int(m.group(1)) + 3, 70) if m else 20 * (attempt + 1)
                print(f"⏳ [Rate limit reached — waiting {wait}s, retry {attempt + 1}/{MAX_RETRIES}]")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("Max retries exceeded.")


def run_agent():
    print("🤖 Welcome to the Smart Study Buddy! (Type 'exit' to quit)")
    print("-" * 70)

    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        tools=[search_course_materials, calculate_average],
        system_instruction=f"""
        You are a helpful and friendly AI Study Assistant for a university student.
        Current User Context: {memory.get_long_term_context()}

        Rules:
        1. ALWAYS use the provided tools when the user asks for calculations or course-specific information.
        2. If the user shares a personal fact (name, major, goals), acknowledge it clearly.
        3. Keep answers concise and cite page numbers when using search_course_materials.
        4. If a tool returns an error, explain it politely and suggest an alternative.
        5. Reply in the same language the user writes in (Arabic or English).
        """,
    )

    history = [
        {"role": ("model" if m["role"] == "model" else "user"), "parts": m["parts"]}
        for m in memory.get_short_term_history()
    ]
    chat = model.start_chat(history=history)

    while True:
        user_input = input("\n👤 You: ").strip()
        if user_input.lower() in ["exit", "quit", "خروج"]:
            print("👋 Goodbye! Have a great study session.")
            break
        if not user_input:
            continue

        try:
            message = user_input
            final_text = None

            for iteration in range(MAX_TOOL_ITERATIONS + 1):
                response = send_with_retry(chat.send_message, message)
                fc = extract_function_call(response)

                if fc is None:
                    final_text = response_text(response)
                    break

                if iteration == MAX_TOOL_ITERATIONS:
                    final_text = ("I kept needing tools without reaching an answer "
                                  "(loop guard triggered). Please rephrase your question.")
                    break

                tool_name = fc.name
                tool_args = dict(fc.args)
                print(f"\n⚙️  [Agent calling tool: '{tool_name}']")

                if tool_name in AVAILABLE_FUNCTIONS:
                    try:
                        result = AVAILABLE_FUNCTIONS[tool_name](**tool_args)
                    except Exception as e:
                        result = json.dumps({"error": f"Tool execution failed: {str(e)}"})
                else:
                    result = json.dumps({"error": f"Tool '{tool_name}' is not authorized."})

                preview = result[:200] + "..." if len(result) > 200 else result
                print(f"🔧 [Tool Result]: {preview}")

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

            memory.add_to_short_term("user", user_input)
            memory.add_to_short_term("model", final_text)

            lower = user_input.lower()
            if any(k in lower for k in ["my name is", "i study", "my major", "اسمي", "أدرس", "تخصصي"]):
                memory.add_fact(user_input)
                print("💾 [Saved to long-term memory]")

            print(f"\n🤖 Assistant: {final_text}")

        except Exception as e:
            print(f"\n❌ An error occurred: {str(e)}")
            print("Please try again in a moment.")


if __name__ == "__main__":
    run_agent()