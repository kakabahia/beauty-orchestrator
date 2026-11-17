"""
LangGraph Multi-Agent Funnel Pipeline with Parallel Execution (Fan-out / Fan-in)
and Conditional Transverse Shade Routing.

Pipeline Architecture:
    START
      ├──> [agent_1a_social] (Consenso Social & Experiência de Uso)
      └──> [agent_1b_formulation] (Formulação, Textura & Ativos - GPT-4o-mini)
                  │
                  ▼ (Fan-in Synchronization)
        [stage_2_store_budget] (Official Stores + Jina Reader + Gemini 2.5 Flash + Strict Budget Cutoff)
                  │
                  ├── (Conditional Edge: Base/Corretivo OR Protetor Solar com Cor) ──> [stage_3_findation] (GPT-4o + Findation)
                  │                                                                            │
                  └────────────────────── (Otherwise: Bypass Shade Translation) ───────────────┼
                                                                                               ▼
                                                                                    [stage_4_derm_curator] (Claude 3.7 Sonnet)
                                                                                               │
                                                                                               ▼
                                                                                              END
"""
from typing import Any, Dict
from langgraph.graph import END, START, StateGraph

from recommender.schemas import (
    OrchestrationTelemetry,
    RecommendationRequestUnion,
    RecommendationResponse,
)
from .derm_curator import run_stage_4_derm_curator
from .disambiguator import resolve_freeform_product_items
from .formulation_expert import run_agent_1b_formulation
from .shade_translator import run_stage_3_findation, should_translate_shades
from .social_consensus import run_agent_1a_social
from .state import OrchestratorState
from .store_search_budget import run_stage_2_store_budget
from .webhook import trigger_n8n_webhook_async


def build_beauty_orchestrator_graph():
    """Builds and compiles the 4-stage LangGraph StateGraph with Fan-out / Fan-in."""
    workflow = StateGraph(OrchestratorState)

    # Register the 5 specialized agent nodes across the 4 funnel stages
    workflow.add_node("agent_1a_social", run_agent_1a_social)
    workflow.add_node("agent_1b_formulation", run_agent_1b_formulation)
    workflow.add_node("stage_2_store_budget", run_stage_2_store_budget)
    workflow.add_node("stage_3_findation", run_stage_3_findation)
    workflow.add_node("stage_4_derm_curator", run_stage_4_derm_curator)

    # Stage 1: Parallel Execution (Fan-out from START to 1A and 1B simultaneously)
    workflow.add_edge(START, "agent_1a_social")
    workflow.add_edge(START, "agent_1b_formulation")

    # Stage 2: Synchronization (Fan-in from both 1A and 1B into Store Search & Budget Cut)
    workflow.add_edge(["agent_1a_social", "agent_1b_formulation"], "stage_2_store_budget")

    # Stage 3: Conditional Transverse Routing (Findation Shade Translation only when needed)
    workflow.add_conditional_edges(
        "stage_2_store_budget",
        should_translate_shades,
        {
            "stage_3_findation": "stage_3_findation",
            "stage_4_derm_curator": "stage_4_derm_curator",
        },
    )

    # Stage 3 -> Stage 4 -> END
    workflow.add_edge("stage_3_findation", "stage_4_derm_curator")
    workflow.add_edge("stage_4_derm_curator", END)

    return workflow.compile()


# Singleton compiled LangGraph instance
compiled_beauty_graph = build_beauty_orchestrator_graph()


def run_orchestrator_pipeline(payload: RecommendationRequestUnion) -> RecommendationResponse:
    """
    Executes the compiled LangGraph pipeline for any of the 3 Discriminated Union flows
    and triggers the non-blocking n8n webhook upon completion.
    """
    resolved_products = resolve_freeform_product_items(list(payload.confirmed_products))

    initial_state: OrchestratorState = {
        "flow_type": payload.flow_type,
        "skin_type": payload.skin_type,
        "is_sensitive": payload.is_sensitive,
        "budget": float(payload.budget),
        "confirmed_products": resolved_products,
        "category": getattr(payload, "category", None),
        "active_problems": list(getattr(payload, "active_problems", [])),
        "target_problem": getattr(payload, "target_problem", None),
        "prefers_tinted_sunscreen": bool(getattr(payload, "prefers_tinted_sunscreen", False)),
        "reference_shade": getattr(payload, "reference_shade", None),
        "chromatic_expectation": getattr(payload, "chromatic_expectation", None),
    }

    final_state: Dict[str, Any] = compiled_beauty_graph.invoke(initial_state)

    recommendations = final_state.get("final_recommendations", [])
    total_price = round(sum(item.price for item in recommendations), 2)

    telemetry = OrchestrationTelemetry(
        flow_type=payload.flow_type,
        models_used=final_state.get("models_used", {}),
        stage_1a_social=final_state["social_output"],
        stage_1b_formulation=final_state["formulation_output"],
        stage_2_total_scraped=final_state.get("total_scraped_count", 0),
        stage_2_approved_count=len(recommendations),
        budget_limit=float(payload.budget),
        total_recommended_price=total_price,
        budget_exceeded_alert=bool(final_state.get("budget_exceeded_alert", False)),
        budget_alert_message=final_state.get("budget_alert_message"),
        stage_3_findation_executed=bool(final_state.get("stage_3_executed", False)),
        stage_3_routing_reason=final_state.get("stage_3_routing_reason", ""),
        n8n_webhook_queued=True,
    )

    response_obj = RecommendationResponse(
        recommendations=recommendations,
        resolved_user_products=resolved_products,
        telemetry=telemetry,
        curated_summary=final_state.get("curated_summary", ""),
    )

    # Fire non-blocking n8n background webhook (3s timeout, full exception isolation)
    trigger_n8n_webhook_async(payload.model_dump(mode="json"), response_obj)

    return response_obj
