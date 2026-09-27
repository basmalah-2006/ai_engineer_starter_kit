from typing import Any, Dict, List, Optional, TypedDict

class AgentState(TypedDict):
    """Shared state passed between all nodes of the LangGraph state machine."""

    #Inputs
    file_path: str             # Path to the input CSV dataset
    user_request: str          # Natural-language analysis request from the user

    #Data loading & validation
    data_loaded: bool          # Was the CSV read successfully?
    data_valid: bool           # Did the data pass quality checks?
    columns: List[str]         # Column names of the dataset
    dataframe_shape: str       # e.g. "8807 rows x 12 columns"

    #Analysis outputs
    statistics: Dict[str, Any] # Computed statistics (distributions, tops, trends)
    charts_paths: List[str]    # File paths of generated charts (PNG)
    insights: List[str]        # LLM-generated strategic insights

    #Reporting
    report_markdown: str       # Final report in Markdown format

    #Error handling & guardrails
    error: Optional[str]       # Last error message, None if healthy
    retry_count: int           # How many repair attempts were made
    max_retries: int           # Hard cap on repair attempts (default 3)
    revision_count: int        # How many times the human rejected the report
    max_revisions: int         # Hard cap on revisions (default 3)

    #Human-in-the-loop
    human_approved: bool       # Did the human approve the report?
    human_feedback: str        # Free-text feedback when rejected

    #Observability
    step_number: int           # Counter of executed steps
    messages: List[str]        # Execution log (one entry per node)