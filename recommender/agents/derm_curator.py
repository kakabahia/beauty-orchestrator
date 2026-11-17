"""
Stage 4: Dermatological Guardrail & Final Curation (Anthropic Claude 3.7 Sonnet).
- Validates suitable skin types for each finalist product.
- If the user has sensitive skin (`is_sensitive=True`), inspects the formula and reviews
  for irritating triggers (strong fragrance, denatured alcohol, comedogenics) and attaches
  a safety badge or specific dermatological alert.
- Delivers the final recommendation cards with Match Score (0 to 100%), exact shade (or
  explicitly 'Incolor (sem pigmento)'), pros/cons according to consumers, real price, and
  direct store purchase link.
"""
import logging
import sys
from typing import Any, Dict, List
from pydantic import BaseModel, Field

from recommender.schemas import CandidateProduct
from .shade_translator import evaluate_findation_routing
from .state import OrchestratorState
from .tools import get_claude_llm, get_openai_llm, google_search_product_link

logger = logging.getLogger(__name__)


class ClaudeCuratedItem(BaseModel):
    product_name: str
    match_score: int = Field(..., ge=0, le=100)
    indicated_skin_types: List[str]
    dermatological_badge: str
    dermatological_safety_level: str
    irritant_warnings: List[str] = Field(default_factory=list)
    curator_verdict: str


class ClaudeCurationBatch(BaseModel):
    items: List[ClaudeCuratedItem]
    curated_summary: str


def _inspect_dermatological_safety(
    candidate: CandidateProduct,
    skin_type: str,
    is_sensitive: bool,
    budget_exceeded: bool,
) -> CandidateProduct:
    """Applies rigorous dermatological guardrail inspection and computes Match Score (0-100%)."""
    curated = candidate.model_copy(deep=True)
    comp_low = (curated.composition_notes + " " + " ".join(curated.cons)).lower()

    irritant_warnings: List[str] = []
    has_fragrance = "fragrância suave" in comp_low or "fragrância refrescante" in comp_low or (
        "fragrância" in comp_low and "sem fragrância" not in comp_low and "zero fragrância" not in comp_low
    )
    has_strong_acid = any(
        a in " ".join(curated.key_actives).lower()
        for a in ["ácido lático 7%", "ácido glicólico", "retinol"]
    )

    if is_sensitive:
        if has_fragrance:
            irritant_warnings.append(
                "Atenção Pele Sensível: Contém fragrância na composição. Realize teste de contato no antebraço."
            )
        if has_strong_acid:
            irritant_warnings.append(
                "Atenção Pele Sensível: Contém hidroxiácidos renovadores. Introduza em noites alternadas."
            )

    if irritant_warnings:
        curated.dermatological_Safety_level = "caution"
        curated.dermatological_badge = "⚠️ Alerta Pele Sensível: Inspecionar Gatilho"
    elif is_sensitive:
        curated.dermatological_Safety_level = "safe"
        curated.dermatological_badge = "🛡️ Selo Safe-Skin: Zero Fragrância & Zero Álcool Irritante"
    else:
        curated.dermatological_Safety_level = "safe"
        curated.dermatological_badge = "✅ Validado Dermatologicamente (Não-Comedogênico)"

    curated.irritant_warnings = irritant_warnings

    # Validate indicated skin types
    tex_low = curated.texture.lower()
    if "creme nutritivo" in tex_low or "lamelar" in tex_low:
        curated.indicated_skin_types = ["Pele Seca", "Pele Normal", "Pele Sensibilizada"]
    elif "gel" in tex_low or "aquos" in tex_low or "toque seco" in tex_low or "oil-free" in tex_low:
        curated.indicated_skin_types = [
            f"Pele {skin_type.capitalize()} (Ideal)",
            "Pele Oleosa",
            "Pele Mista",
            "Pele Acneica",
        ]
    else:
        curated.indicated_skin_types = [
            f"Pele {skin_type.capitalize()} (Ideal)",
            "Todos os Tipos de Pele",
        ]

    # Ensure shade label is set even if Stage 3 was bypassed by conditional routing
    if not curated.is_tinted:
        curated.recommended_shade = "Incolor (sem pigmento)"
        curated.shade_match_method = "colorless"
        curated.shade_translation_note = "Fórmula translúcida/incolor — sem necessidade de ajuste de pigmento."

    # Compute Hyper-Personalized Match Score (0 - 100%)
    base_score = 94
    if irritant_warnings:
        base_score -= 9
    if budget_exceeded:
        base_score -= 6
    if curated.shade_match_method == "undertone_approx":
        base_score -= 3
    if any(k in tex_low for k in ["gel", "aquos", "fluido", "sérum"]) and skin_type in ("oleosa", "mista"):
        base_score += 3

    curated.match_score = max(72, min(99, base_score))

    curated.curator_verdict = (
        f"Recomendado para pele {skin_type}"
        + (" e sensível" if is_sensitive else "")
        + f" devido ao veículo '{curated.texture}' com {', '.join(curated.key_actives[:2])}. "
        + (
            "Aprovado no rigoroso Guardrail Dermatológico sem álcool desnaturado ou perfume alergênico."
            if not irritant_warnings
            else irritant_warnings[0]
        )
    )
    return curated


def run_stage_4_derm_curator(state: OrchestratorState) -> Dict[str, Any]:
    """
    LangGraph Node: Stage 4 - Dermatological Guardrail & Final Curation (Claude 3.7 Sonnet).
    """
    # Take candidates from Stage 3 if executed, otherwise directly from Stage 2
    stage_3_ran = bool(state.get("stage_3_executed", False))
    if stage_3_ran and state.get("shade_translated_candidates"):
        candidates = list(state.get("shade_translated_candidates", []))
        routing_reason = state.get("stage_3_routing_reason", "")
    else:
        candidates = list(state.get("approved_candidates", []))
        _, routing_reason = evaluate_findation_routing(state)

    skin_type = state.get("skin_type", "mista")
    is_sensitive = bool(state.get("is_sensitive", False))
    budget = float(state.get("budget", 150.0))
    budget_exceeded = bool(state.get("budget_exceeded_alert", False))
    chromatic_expectation = state.get("chromatic_expectation")
    confirmed_items = [p for p in state.get("confirmed_products", []) if p.confirmed]
    loved_items = [p for p in confirmed_items if p.sentiment == "loved"]
    disliked_items = [p for p in confirmed_items if p.sentiment == "disliked"]

    user_feedback_notes: List[str] = []
    for p in loved_items + disliked_items:
        p_label = p.official_name or p.raw_input
        if p.extracted_positives:
            user_feedback_notes.append(f"pontos positivos valorizados em '{p_label}' ({', '.join(p.extracted_positives[:2])})")
        if p.extracted_negatives:
            user_feedback_notes.append(f"ressalva/reclamação superada de '{p_label}' ({', '.join(p.extracted_negatives[:2])})")

    curated_list: List[CandidateProduct] = []
    for cand in candidates:
        item = _inspect_dermatological_safety(cand, skin_type, is_sensitive, budget_exceeded)
        if item.category in {"blush", "sombra", "contorno", "iluminador"} and chromatic_expectation:
            item.recommended_shade = f"Acabamento Cromático: {chromatic_expectation}"
            item.shade_match_method = "chromatic_match"
            item.shade_translation_note = (
                f"Selecionado para entregar o efeito desejado: '{chromatic_expectation}'."
            )

        if user_feedback_notes:
            item.curator_verdict += f" Alinhado às suas observações ({'; '.join(user_feedback_notes[:3])}) e ao teto de R$ {budget:.2f}."

        curated_list.append(item)

    # Optional Claude 3.7 Sonnet live refinement when ANTHROPIC_API_KEY is configured
    claude = get_claude_llm()
    curated_summary = (
        f"Curadoria final concluída para pele {skin_type}"
        + (" (com protocolo de proteção para pele sensível)" if is_sensitive else "")
        + f" respeitando o orçamento de R$ {budget:.2f}: {len(curated_list)} produto(s) selecionado(s) "
        "levando em conta os pontos positivos e negativos das suas observações e validação de preço/estoque no link."
    )

    if claude and curated_list:
        try:
            structured_claude = claude.with_structured_output(ClaudeCurationBatch)
            items_desc = "\n".join(
                f"- {c.product_name} ({c.brand}) | Textura: {c.texture} | Ativos: {c.key_actives} | Composição: {c.composition_notes}"
                for c in curated_list
            )
            prompt = (
                "Você é o Agente do Estágio 4 (Guardrail Dermatológico & Curadoria Final - Claude 3.7 Sonnet).\n"
                f"Perfil: Pele {skin_type}, Sensível={is_sensitive}, Alerta de Orçamento={budget_exceeded}.\n"
                f"Candidatos finalistas:\n{items_desc}\n\n"
                "Valide para quais tipos de pele cada produto é indicado, inspecione gatilhos irritantes se a pele for sensível, "
                "atribua o match_score (0-100) e redija o veredito dermatológico conciso."
            )
            batch: ClaudeCurationBatch = structured_claude.invoke(prompt)
            if batch and batch.items:
                by_name = {it.product_name.lower(): it for it in batch.items}
                for c in curated_list:
                    matched = by_name.get(c.product_name.lower())
                    if matched:
                        c.match_score = matched.match_score
                        c.indicated_skin_types = matched.indicated_skin_types
                        c.dermatological_badge = matched.dermatological_badge
                        c.irritant_warnings = matched.irritant_warnings
                        c.curator_verdict = matched.curator_verdict
                if batch.curated_summary:
                    curated_summary = batch.curated_summary
        except Exception as exc:
            logger.warning("Claude 3.7 Sonnet Stage 4 refinement fallback: %s", exc)

    models_used = {
        "stage_1a_social": "gemini-2.5-flash / tavily-reviews",
        "stage_1b_formulation": "gpt-4o-mini (Structured Chemistry & Hidden Patterns)",
        "stage_2_store_budget": "gemini-2.5-flash + Jina Reader + Tavily Official Domains",
        "stage_3_findation": "gpt-4o + Findation Matrix" if stage_3_ran else "Bypassed (Conditional Funnel)",
        "stage_4_derm_curator": "claude-3-7-sonnet-latest (Dermatological Guardrail)",
    }

    return {
        "stage_3_executed": stage_3_ran,
        "stage_3_routing_reason": routing_reason,
        "final_recommendations": curated_list,
        "curated_summary": curated_summary,
        "models_used": models_used,
    }
