"""
LangGraph State definition for the 4-stage parallel Fan-out / Fan-in funnel.
"""
from typing import Any, Dict, List, Optional, TypedDict
from recommender.schemas import (
    CandidateProduct,
    ConfirmedProductItem,
    FormulationAnalysisOutput,
    SocialConsensusOutput,
)


class OrchestratorState(TypedDict, total=False):
    """
    Shared state across the LangGraph nodes.
    Note: Parallel nodes (Agent 1A and Agent 1B) write to distinct keys
    (`social_output` and `formulation_output`) so LangGraph Fan-out / Fan-in
    synchronizes cleanly without state collisions.
    """

    # Input profile from the Discriminated Union
    flow_type: str  # "skincare_rotina" | "skincare_especifico" | "maquiagem"
    skin_type: str  # "oleosa" | "seca" | "mista" | "normal"
    is_sensitive: bool
    budget: float
    category: Optional[str]
    active_problems: List[str]
    target_problem: Optional[str]
    prefers_tinted_sunscreen: bool
    reference_shade: Optional[str]
    chromatic_expectation: Optional[str]
    confirmed_products: List[ConfirmedProductItem]

    # Stage 1 Parallel Outputs (Fan-out)
    social_output: SocialConsensusOutput
    formulation_output: FormulationAnalysisOutput

    # Stage 2 Synchronization & Budget Cut Outputs (Fan-in)
    unified_queries: List[str]
    total_scraped_count: int
    approved_candidates: List[CandidateProduct]
    budget_exceeded_alert: bool
    budget_alert_message: Optional[str]

    # Stage 3 Conditional Findation Routing & Output
    stage_3_executed: bool
    stage_3_routing_reason: str
    shade_translated_candidates: List[CandidateProduct]

    # Stage 4 Dermatological Guardrail & Final Curation Output
    final_recommendations: List[CandidateProduct]
    curated_summary: str
    models_used: Dict[str, str]
