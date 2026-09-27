import uuid

from dotenv import load_dotenv

load_dotenv() 

from agents.graph import build_graph
from agents.state import AgentState

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_PATH = str(PROJECT_ROOT / "data" / "netflix_titles.csv")

USER_REQUEST = (
    "Analyze the Netflix catalog to support the 2026 content-acquisition "
    "strategy. Cover: (1) yearly growth of added titles, (2) top producing "
    "countries and expansion opportunities, (3) most common genres and catalog "
    "gaps, (4) movies vs TV shows split. End with actionable recommendations."
)


def build_initial_state() -> AgentState:
    """Fresh state for a new run."""
    return {
        "file_path": DATASET_PATH,
        "user_request": USER_REQUEST,
        "data_loaded": False,
        "data_valid": False,
        "columns": [],
        "dataframe_shape": "",
        "statistics": {},
        "charts_paths": [],
        "insights": [],
        "report_markdown": "",
        "error": None,
        "retry_count": 0,
        "max_retries": 3,
        "revision_count": 0,
        "max_revisions": 3,
        "human_approved": False,
        "human_feedback": "",
        "step_number": 0,
        "messages": [],
    }


def main() -> None:
    print("=" * 70)
    print("NETFLIX DATA ANALYSIS AGENT  |  LangGraph + Groq + LangSmith")
    print("=" * 70)

    graph = build_graph()

    config = {
        "configurable": {"thread_id": str(uuid.uuid4())},
        "recursion_limit": 30,
    }

    final_state = graph.invoke(build_initial_state(), config)

    print("\n" + "=" * 70)
    if final_state.get("error"):
        print(f"RUN FAILED: {final_state['error']}")
    else:
        print("RUN COMPLETED SUCCESSFULLY")
        print(f"  Steps executed : {final_state['step_number']}")
        print(f"  Report         : output/final_report.md")
        print(f"  Charts         : output/*.png")
        print(f"  Traces         : https://smith.langchain.com")
    print("=" * 70)


if __name__ == "__main__":
    main()