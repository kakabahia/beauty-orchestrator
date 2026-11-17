"""
Agent 1A (Consenso Social & Experiência de Uso) — Parallel Fan-out Node.
Analyzes the user's free-form loved and disliked products (corresponded via Generative AI)
together with the user's personal reasons (`user_reason`) for liking or disliking each item,
plus real consumer reviews, extracting practical success drivers and complaints to avoid.
"""
import logging
from typing import Any, Dict, List

from recommender.schemas import ConfirmedProductItem, SocialConsensusOutput
from .state import OrchestratorState
from .tools import get_gemini_llm, get_openai_llm, tavily_search_reviews

logger = logging.getLogger(__name__)


def run_agent_1a_social(state: OrchestratorState) -> Dict[str, Any]:
    """LangGraph Node: Agent 1A - Social Consensus & Practical User Experience."""
    confirmed: List[ConfirmedProductItem] = [
        p for p in state.get("confirmed_products", []) if p.confirmed
    ]
    loved = [p for p in confirmed if p.sentiment == "loved"]
    disliked = [p for p in confirmed if p.sentiment == "disliked"]

    skin_type = state.get("skin_type", "mista")
    flow_type = state.get("flow_type", "skincare_rotina")
    category = state.get("category") or "rotina skincare"
    budget = float(state.get("budget", 150.0))

    # Gather live social reviews via Tavily if configured
    review_snippets: List[str] = []
    for item in (loved + disliked)[:4]:
        hits = tavily_search_reviews(
            f"resenha avaliações opiniões '{item.official_name or item.raw_input}' pele {skin_type} prós e contras",
            max_results=2,
        )
        for h in hits:
            review_snippets.append(
                f"[{item.sentiment.upper()} - {item.official_name or item.raw_input}]: {h.get('content', '')[:300]}"
            )

    llm = get_gemini_llm() or get_openai_llm(smart=False)
    if llm:
        try:
            structured_llm = llm.with_structured_output(SocialConsensusOutput)
            loved_desc = (
                " | ".join(
                    f"'{p.raw_input}' -> Correspondência IA: {p.official_name} ({p.predominant_texture})"
                    + (f" [Por que gosta: {p.user_reason}]" if p.user_reason else "")
                    for p in loved
                )
                or "Nenhum informado"
            )
            disliked_desc = (
                " | ".join(
                    f"'{p.raw_input}' -> Correspondência IA: {p.official_name} ({p.predominant_texture})"
                    + (f" [Por que NÃO gosta / Reclamação: {p.user_reason}]" if p.user_reason else "")
                    for p in disliked
                )
                or "Nenhum informado"
            )
            context_reviews = "\n".join(review_snippets) if review_snippets else "Sem snippets externos adicionais."

            prompt = (
                "Você é o Agente 1A (Consenso Social & Experiência do Usuário) do Beauty AI Orchestrator.\n"
                f"Perfil do usuário: Pele {skin_type}, Sensível={state.get('is_sensitive')}, "
                f"Orçamento Máximo=R$ {budget:.2f}, Fluxo={flow_type}, Categoria={category}.\n"
                f"Produtos Favoritos descritos pelo usuário: {loved_desc}\n"
                f"Produtos Não Favoritos / Rejeitados descritos pelo usuário: {disliked_desc}\n"
                f"Avaliações de Consumidores Coletadas:\n{context_reviews}\n\n"
                "Extraia:\n"
                "1. Motivos práticos de aprovação (priorizando o que o próprio usuário disse que gosta nos favoritos + consenso social).\n"
                "2. Reclamações a evitar (priorizando o porquê o usuário disse que não gostou dos produtos rejeitados + reclamações recorrentes).\n"
                "3. 3 a 4 queries de busca focadas em performance real e limite de preço.\n"
                "4. Uma síntese clara do consenso social e das preferências/reclamações do usuário."
            )
            result: SocialConsensusOutput = structured_llm.invoke(prompt)
            if result and result.performance_search_queries:
                return {"social_output": result}
        except Exception as exc:
            logger.warning("Agent 1A LLM structured call fallback: %s", exc)

    # Intelligent synthesis prioritizing mixed positive/negative clauses from `user_reason` across ALL items
    success_drivers: List[str] = []
    complaints_to_avoid: List[str] = []

    for p in loved + disliked:
        label = p.official_name or p.raw_input
        # 1. Add extracted positive clauses (even if mentioned inside a disliked item!)
        for pos in p.extracted_positives:
            success_drivers.append(f"Ponto positivo valorizado em '{label}': {pos}")
        # 2. Add extracted negative clauses / caveats (even if mentioned as a caveat inside a loved item!)
        for neg in p.extracted_negatives:
            prefix = "Ressalva a corrigir de" if p.sentiment == "loved" else "Reclamação direta em"
            complaints_to_avoid.append(f"{prefix} '{label}': Evitar/superar '{neg}'")

        if p.sentiment == "loved":
            if not p.extracted_positives and p.user_reason and p.user_reason.strip() and not p.extracted_negatives:
                success_drivers.append(f"Aprovação direta em '{label}': {p.user_reason.strip()}")
            tex_low = (p.predominant_texture or "").lower()
            if "gel" in tex_low or "aquos" in tex_low or "fluido" in tex_low:
                success_drivers.append(f"Sensorial leve/aquoso correspondente a '{label}'")
            if p.is_tinted_sunscreen:
                success_drivers.append(f"Cobertura uniforme com proteção solar inspirada em '{label}'")
        elif p.sentiment == "disliked" and not p.extracted_negatives:
            complaints_to_avoid.append(
                f"Evitar sensorial/fórmula similar ao rejeitado '{label}' ({p.predominant_texture})"
            )

    # Add baseline practical drivers
    success_drivers.extend(
        [
            "Absorve rápido sem deixar filme residual pesado",
            "Não esfarela na sobreposição de camadas (skincare/maquiagem)",
            "Não arde os olhos ao longo do dia",
        ]
    )
    complaints_to_avoid.extend(
        [
            "Oxida ao longo do dia alterando o subtom na pele",
            "Deixa a pele pegajosa ou com brilho excessivo após 2 horas",
            "Acabamento pesado que acumula nos poros ou linhas finas",
        ]
    )

    if skin_type in ("oleosa", "mista"):
        success_drivers.append("Controle de oleosidade duradouro com acabamento matte/natural confortável")
        complaints_to_avoid.append("Efeito rebote de oleosidade ou obstrução de poros (comedogenicidade)")
    else:
        success_drivers.append("Viço saudável e hidratação prolongada sem repuxar")
        complaints_to_avoid.append("Textura esturricada que craquela em áreas ressecadas")

    # Generate performance-focused queries
    if flow_type == "skincare_rotina":
        queries = [
            f"gel de limpeza suave pele {skin_type} não repuxa até R$ {budget:.0f}",
            f"hidratante facial toque seco absorve rápido não esfarela pele {skin_type}",
            f"protetor solar facial fluido não arde os olhos pele {skin_type}",
        ]
    elif flow_type == "skincare_especifico":
        queries = [
            f"{category} pele {skin_type} absorve rápido não esfarela resenha",
            f"melhor {category} dermocosmético toque seco não pegajoso pele {skin_type}",
        ]
    else:
        queries = [
            f"{category} maquiagem pele {skin_type} não oxida longa duração não craquela",
            f"{category} acabamento confortável alta performance pele {skin_type}",
        ]

    reasons_count = sum(1 for p in (loved + disliked) if p.user_reason and p.user_reason.strip())
    summary = (
        f"Análise IA de {len(loved)} favorito(s) e {len(disliked)} não favorito(s) "
        + (f"(incluindo {reasons_count} justificativa(s)/reclamação(ões) direta(s) do usuário) " if reasons_count else "")
        + f"dentro do limite de R$ {budget:.2f}: priorizando fórmulas que atendam às aprovações relatadas "
        f"e eliminando candidatos que apresentem as reclamações apontadas para pele {skin_type}."
    )

    return {
        "social_output": SocialConsensusOutput(
            success_drivers=list(dict.fromkeys(success_drivers))[:6],
            recurring_complaints_to_avoid=list(dict.fromkeys(complaints_to_avoid))[:6],
            performance_search_queries=queries,
            summary_rationale=summary,
        )
    }
