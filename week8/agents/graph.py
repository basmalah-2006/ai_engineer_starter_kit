from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agents.nodes import (
    analyze_data_node,
    fail_node,
    generate_report_node,
    human_approval_node,
    load_data_node,
    retry_node,
    save_output_node,
    validate_data_node,
)
from agents.state import AgentState


def route_after_validation(state: AgentState) -> str:
    """Conditional edge: branch on data quality and retry budget."""
    if state["data_valid"]:
        return "analyze_data"
    if state["retry_count"] < state["max_retries"]:
        return "retry"
    return "fail"


def route_after_approval(state: AgentState) -> str:
    """Conditional edge: approve, or loop back for a revision (with a cap)."""
    if state["human_approved"]:
        return "save_output"
    if state["revision_count"] >= state["max_revisions"]:
        return "save_output"  
    return "generate_report"


def build_graph():
    """Assemble and compile the state machine with persistent memory."""
    workflow = StateGraph(AgentState)

    #Nodes
    workflow.add_node("load_data", load_data_node)
    workflow.add_node("validate_data", validate_data_node)
    workflow.add_node("retry", retry_node)
    workflow.add_node("fail", fail_node)
    workflow.add_node("analyze_data", analyze_data_node)
    workflow.add_node("generate_report", generate_report_node)
    workflow.add_node("human_approval", human_approval_node)
    workflow.add_node("save_output", save_output_node)

    #Edges
    workflow.add_edge(START, "load_data")
    workflow.add_edge("load_data", "validate_data")

    workflow.add_conditional_edges(
        "validate_data",
        route_after_validation,
        {"analyze_data": "analyze_data", "retry": "retry", "fail": "fail"},
    )
    workflow.add_edge("retry", "validate_data")
    workflow.add_edge("fail", END)

    workflow.add_edge("analyze_data", "generate_report")
    workflow.add_edge("generate_report", "human_approval")

    workflow.add_conditional_edges(
        "human_approval",
        route_after_approval,
        {"save_output": "save_output", "generate_report": "generate_report"},
    )
    workflow.add_edge("save_output", END)

    # MemorySaver = persistent state
    return workflow.compile(checkpointer=MemorySaver())