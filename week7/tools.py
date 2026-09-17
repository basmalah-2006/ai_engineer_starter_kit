import json
from functools import lru_cache
import rag


@lru_cache(maxsize=1)
def get_vector_store():
    """Loads and caches the FAISS vector store."""
    try:
        return rag.load_vector_store()
    except FileNotFoundError:
        return None


def search_course_materials(query: str) -> str:
    """
    Searches the AI course lecture notes for specific concepts,
    definitions, algorithms, or other course-related information.
    """
    try:
        vector_store = get_vector_store()

        if vector_store is None:
            return (
                "Knowledge base not built yet. "
                "Run: python rag.py"
            )

        docs = vector_store.similarity_search(str(query), k=3)

        if not docs:
            return "No relevant information found in the course materials."

        context = ""

        for i, doc in enumerate(docs, 1):
            page = doc.metadata.get("page", "?")
            context += (
                f"\n--- [Source {i}: Page {page}] ---\n"
                f"{doc.page_content}\n"
            )

        return context

    except Exception as e:
        return json.dumps(
            {"error": f"RAG search failed: {str(e)}"}
        )


def calculate_average(grades: list) -> str:
    """
    Calculates the arithmetic mean of a list of numerical grades.
    Use when the user provides grades and asks for the average.
    """
    try:
        grades = [float(g) for g in grades]

        if len(grades) == 0:
            return json.dumps(
                {"error": "Grades list is empty."}
            )

        avg = sum(grades) / len(grades)

        return json.dumps(
            {
                "average": round(avg, 2),
                "count": len(grades),
                "grades": grades,
            }
        )

    except Exception as e:
        return json.dumps(
            {"error": f"Calculation failed: {str(e)}"}
        )


tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "search_course_materials",
            "description": (
                "Search the AI course lecture notes for "
                "concepts, definitions, algorithms, and "
                "course-related information."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The course-related search query.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_average",
            "description": (
                "Calculate the arithmetic mean of numerical grades."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "grades": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "A list of numerical grades.",
                    }
                },
                "required": ["grades"],
            },
        },
    },
]


ALLOWED_TOOLS = {
    "search_course_materials": search_course_materials,
    "calculate_average": calculate_average,
}


def execute_tool(tool_name: str, tool_args: dict) -> str:
    """
    Executes an authorized tool safely.
    Any tool outside the allow-list is rejected.
    """
    if tool_name not in ALLOWED_TOOLS:
        return json.dumps(
            {
                "error": (
                    f"Tool '{tool_name}' is not authorized."
                )
            }
        )

    try:
        return ALLOWED_TOOLS[tool_name](**tool_args)

    except Exception as e:
        return json.dumps(
            {
                "error": (
                    f"Tool execution failed: {str(e)}"
                )
            }
        )