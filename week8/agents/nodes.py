import os
from typing import Any, Dict
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from agents.state import AgentState
from tools.data_tools import (
    clean_data,
    compute_statistics,
    create_charts,
    load_csv,
    validate_data,
)

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = str(PROJECT_ROOT / "output")

def get_llm() -> ChatGroq:
    """Return a configured Groq chat model.

    The model id is read from the GROQ_MODEL environment variable so that
    retired models can be swapped without touching the code (LO 8.4: reliability).
    """
    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        temperature=0.3,
        api_key=os.getenv("GROQ_API_KEY"),
    )


def _log(state: AgentState, message: str) -> Dict[str, Any]:
    """Helper: increment the step counter and append a log entry."""
    return {
        "step_number": state.get("step_number", 0) + 1,
        "messages": state.get("messages", []) + [message],
    }


# Node 1: Load Data
def load_data_node(state: AgentState) -> Dict[str, Any]:
    """Read the raw CSV file and store its schema in the state."""
    print("\n[LOAD DATA] Reading CSV file ...")
    success, error, columns, shape = load_csv(state["file_path"])
    print(f"    -> {'OK: ' + shape if success else 'FAILED: ' + error}")

    updates: Dict[str, Any] = {
        "data_loaded": success,
        "columns": columns,
        "dataframe_shape": shape,
        "error": None if success else error,
    }
    updates.update(_log(state, f"[load_data] success={success}, shape={shape}"))
    return updates


# Node 2: Validate Data
def validate_data_node(state: AgentState) -> Dict[str, Any]:
    """Run quality checks on the dataset."""
    print("\n[VALIDATE] Checking data quality ...")
    is_valid, error, details = validate_data(state["file_path"])
    print(f"    -> {'VALID' if is_valid else 'INVALID: ' + error}")

    updates: Dict[str, Any] = {
        "data_valid": is_valid,
        "error": None if is_valid else error,
    }
    updates.update(_log(state, f"[validate] valid={is_valid}"))
    return updates


# Node 3: Retry / Self-Heal
def retry_node(state: AgentState) -> Dict[str, Any]:
    """
    Recovery step (LO 8.2): attempt to clean the dataset,
    then the graph loops back to validation.
    """
    attempt = state.get("retry_count", 0) + 1
    print(f"\n[RETRY] Attempt {attempt}/{state.get('max_retries', 3)}: repairing dataset ...")
    success, error, cleaned_path, rows_removed = clean_data(state["file_path"], OUTPUT_DIR)
    if success:
        print(f"    -> removed {rows_removed} bad rows, saved to {cleaned_path}")
    else:
        print(f"    -> cleaning failed: {error}")
        
    updates: Dict[str, Any] = {
        "retry_count": attempt,
        "file_path": cleaned_path if success else state["file_path"],
        "error": None if success else error,
    }
    updates.update(_log(state, f"[retry] attempt={attempt}, rows_removed={rows_removed}"))
    return updates


# Node 4: Fail(terminal) 
def fail_node(state: AgentState) -> Dict[str, Any]:
    """Terminal node reached when retries are exhausted (guardrail)."""
    print("\n[FAIL] Max retries exceeded - aborting pipeline.")
    updates: Dict[str, Any] = {"error": "Validation failed after maximum retries."}
    updates.update(_log(state, "[fail] max retries exceeded"))
    return updates


# Node 5: Analyze Data(uses Tools: pandas + matplotlib + LLM) 
def analyze_data_node(state: AgentState) -> Dict[str, Any]:
    """Compute statistics, generate charts, and extract LLM insights."""
    print("\n[ANALYZE] Computing statistics and charts ...")

    statistics = compute_statistics(state["file_path"])
    print(f"    -> analyzed {statistics['total_titles']} titles")

    charts = create_charts(state["file_path"], OUTPUT_DIR)
    print(f"    -> saved {len(charts)} charts in {OUTPUT_DIR}/")

    llm = get_llm()
    prompt = (
        "You are a senior content analyst at a streaming company. "
        "Based ONLY on the statistics below, produce 5-7 concise strategic "
        "insights as a bullet list. Cite concrete numbers.\n\n"
        f"STATISTICS:\n{statistics}"
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    insights = [line.strip() for line in response.content.splitlines() if line.strip()]

    updates: Dict[str, Any] = {
        "statistics": statistics,
        "charts_paths": charts,
        "insights": insights,
    }
    updates.update(_log(state, f"[analyze] charts={len(charts)}, insights={len(insights)}"))
    return updates


# Node 6: Generate report(also handles revision feedback) 
def generate_report_node(state: AgentState) -> Dict[str, Any]:
    """Write the Markdown report. If the human rejected a previous draft, revise it."""
    feedback = state.get("human_feedback", "").strip()
    revising = bool(feedback)
    print("\n[REPORT] " + ("Revising report per human feedback ..." if revising
                           else "Generating first draft ..."))

    llm = get_llm()
    prompt = (
        "You are a senior content strategist at Netflix. "
        "Write a professional analysis report in Markdown.\n\n"
        f"USER REQUEST:\n{state['user_request']}\n\n"
        f"STATISTICS:\n{state['statistics']}\n\n"
        f"INSIGHTS:\n" + "\n".join(state["insights"]) + "\n\n"
        f"CHART FILES:\n" + "\n".join(state["charts_paths"]) + "\n\n"
        "STRUCTURE:\n"
        "1. # Netflix Content Strategy Analysis Report\n"
        "2. ## Executive Summary\n"
        "3. ## Content Overview\n"
        "4. ## Geographic Analysis\n"
        "5. ## Genre Analysis\n"
        "6. ## Growth Trends\n"
        "7. ## Strategic Recommendations\n"
        "Use headers, bold text and bullet points. Be data-driven."
    )
    if revising:
        prompt += (
            "\n\nA human reviewer rejected the previous draft with this feedback:\n"
            f'"""{feedback}"""\n'
            "The previous draft was:\n"
            f'"""{state["report_markdown"]}"""\n'
            "Produce the FULL revised report addressing every point of feedback."
        )

    response = llm.invoke([HumanMessage(content=prompt)])

    updates: Dict[str, Any] = {"report_markdown": response.content}
    updates.update(_log(state, f"[report] revision={revising}"))
    return updates


# Node 7: Human Approval Checkpoint 
def human_approval_node(state: AgentState) -> Dict[str, Any]:
    """Pause the pipeline and let a human approve or reject the report."""
    print("\n[HUMAN CHECKPOINT] Report preview:")
    print("-" * 70)
    preview = state["report_markdown"]
    print(preview[:1200] + ("\n... [truncated]" if len(preview) > 1200 else ""))
    print("-" * 70)

    answer = input("Approve the report? (y = approve, or type feedback): ").strip()
    approved = answer.lower() in ("y", "yes", "")

    updates: Dict[str, Any] = {
        "human_approved": approved,
        "human_feedback": "" if approved else answer,
        "revision_count": state.get("revision_count", 0) + (0 if approved else 1),
    }
    updates.update(_log(state, f"[human] approved={approved}"))
    return updates


# Node 8: Save Output (terminal, success path) 
def save_output_node(state: AgentState) -> Dict[str, Any]:
    """Persist the final report and the execution log to disk."""
    print("\n[SAVE] Writing final deliverables ...")
    out = out = Path(OUTPUT_DIR)
    out.mkdir(exist_ok=True)

    report_path = out / "final_report.md"
    report_path.write_text(state["report_markdown"], encoding="utf-8")

    log_path = out / "execution_log.txt"
    log_path.write_text("\n".join(state["messages"]), encoding="utf-8")

    print(f"    -> report : {report_path}")
    print(f"    -> log    : {log_path}")

    updates: Dict[str, Any] = {}
    updates.update(_log(state, f"[save] report written to {report_path}"))
    return updates