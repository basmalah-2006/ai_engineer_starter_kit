import json

tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "calculate_average",
            "description": "Calculates the arithmetic mean of a set of grades. Use this when the user requests to calculate their grade average or total grades.",
            "parameters": {
                "type": "object",
                "properties": {
                    "grades": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "List of grades (numbers) to calculate the average for"
                    }
                },
                "required": ["grades"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Searches the internet for up-to-date information. Use this ONLY when the answer is not found in the course materials or when the user explicitly requests information outside the curriculum.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query in English or Arabic"
                    }
                },
                "required": ["query"]
            }
        }
    }
]


def calculate_average(grades: list) -> str:
    """Calculates the average of grades with input validation"""
    try:
        if not all(isinstance(g, (int, float)) for g in grades):
            return json.dumps({"error": "All grades must be numbers"})
        
        if len(grades) == 0:
            return json.dumps({"error": "Grades list is empty"})
        
        avg = sum(grades) / len(grades)
        return json.dumps({
            "grades": grades,
            "average": round(avg, 2),
            "count": len(grades)
        })
    except Exception as e:
        return json.dumps({"error": f"Calculation error: {str(e)}"})


def search_web(query: str) -> str:
    """
    Mock implementation of web search tool.
    In production, replace with real APIs like:
    - DuckDuckGo Search API
    - Tavily API
    - SerpAPI
    """
    return json.dumps({
        "query": query,
        "results": [
            f"Search result for: {query} — This is an example web result.",
            f"A reliable source discussing {query} with additional details."
        ],
        "note": "These are mock results. Replace with a real API like Tavily or DuckDuckGo."
    })



ALLOWED_TOOLS = {
    "calculate_average": calculate_average,
    "search_web": search_web
}

def execute_tool(tool_name: str, arguments: dict) -> str:
    """
    Unified function to execute tools with:
    - Allow-list verification (security)
    - Error handling (reliability)
    """
    if tool_name not in ALLOWED_TOOLS:
        return json.dumps({"error": f"Tool '{tool_name}' is not authorized"})
    
    try:
        func = ALLOWED_TOOLS[tool_name]
        result = func(**arguments)
        return result
    except Exception as e:
        return json.dumps({"error": f"Tool execution failed: {str(e)}"})