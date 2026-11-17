"""
Agent 1B (Formulação, Textura & Ativos) — Parallel Fan-out Node (OpenAI GPT-4o-mini / GPT-4o).
Analyzes the physicochemical structure of the user's free-form loved and disliked products
and the optional reasons (`user_reason`) why the user liked or disliked each item:
- Identifies hidden preference patterns (e.g. if the user likes multiple gel products or praises
  fast-absorbing/fluid textures, enforces gel/aqueous texture as a strict rule; if the user listed
  a tinted sunscreen among favorites, automatically activates `auto_tinted_flag = True`).
- Translates user complaints (`user_reason` on disliked products) into forbidden chemical triggers
  (e.g., fragrance, ocular-irritating UV filters, heavy pilling silicones, comedogenic esters).
"""
import logging
from typing import Any, Dict, List

from recommender.schemas import ConfirmedProductItem, FormulationAnalysisOutput
from .state import OrchestratorState
from .tools import get_gemini_llm, get_openai_llm

logger = logging.getLogger(__name__)

PROBLEM_TO_ACTIVES_MAP: Dict[str, List[str]] = {
    "acne": ["Ácido Salicílico (BHA)", "Niacinamida", "Zinco PCA", "Ácido Succínico"],
    "manchas_melasma": ["Ácido Tranexâmico", "Alfa-Arbutin", "Niacinamida", "Vitamina C Estabilizada", "Thiamidol"],
    "rosacea": ["Ácido Azelaico", "Madecassoside (Centella Asiática)", "Pantenol (Vitamina B5)", "Alantoína"],
    "poros_dilatados": ["Niacinamida 10%", "Ácido Salicílico", "Zinco PCA", "LHA"],
    "comedoes": ["Ácido Salicílico", "Ácido Glicólico / Mandélico", "Niacinamida"],
    "cravos": ["Ácido Salicílico (BHA 2%)", "Zinco PCA", "Argila Verde / Sílica Matificante"],
}


def run_agent_1b_formulation(state: OrchestratorState) -> Dict[str, Any]:
    """LangGraph Node: Agent 1B - Physicochemical Formulation, Texture & Actives."""
    confirmed: List[ConfirmedProductItem] = [
        p for p in state.get("confirmed_products", []) if p.confirmed
    ]
    loved = [p for p in confirmed if p.sentiment == "loved"]
    disliked = [p for p in confirmed if p.sentiment == "disliked"]

    skin_type = state.get("skin_type", "mista")
    is_sensitive = bool(state.get("is_sensitive", False))
    flow_type = state.get("flow_type", "skincare_rotina")
    category = state.get("category") or "rotina_completa"
    active_problems: List[str] = state.get("active_problems") or []
    target_problem = state.get("target_problem") or ""
    budget = float(state.get("budget", 150.0))

    # Rule 1: Detect if user explicitly requested tinted sunscreen OR has a tinted sunscreen in loved products
    has_tinted_in_loved = any(
        p.is_tinted_sunscreen
        or ("com cor" in (p.raw_input or "").lower())
        or ("com cor" in (p.user_reason or "").lower())
        for p in loved
    )
    auto_tinted_flag = bool(state.get("prefers_tinted_sunscreen", False) or has_tinted_in_loved)

    # Rule 2: Detect hidden texture pattern from loved products AND their `user_reason`
    gel_aqueous_count = sum(
        1
        for p in loved
        if any(
            k in f"{p.predominant_texture} {p.raw_input} {p.user_reason or ''}".lower()
            for k in ["gel", "aquos", "fluido", "water", "serum", "sérum", "leve", "sequinho", "toque seco"]
        )
    )
    cream_rich_count = sum(
        1
        for p in loved
        if any(
            k in f"{p.predominant_texture} {p.raw_input} {p.user_reason or ''}".lower()
            for k in ["creme rico", "balm", "nutritiv", "manteiga", "óleo", "hidratante denso"]
        )
    )

    hidden_patterns: List[str] = []
    if gel_aqueous_count >= 2 or (gel_aqueous_count >= 1 and skin_type in ("oleosa", "mista")):
        enforced_texture = "Regra Rígida: Textura Gel, Sérum Aquoso ou Fluido Ultraleve Oil-Free"
        hidden_patterns.append(
            f"Afinidade química detectada por veículos em Gel/Fase Aquosa ({gel_aqueous_count} favorito(s)): "
            "impondo textura gel/aquosa oil-free como regra de corte."
        )
    elif cream_rich_count >= 1 and skin_type == "seca":
        enforced_texture = "Regra Rígida: Emulsão Lamelar Rica / Gel-Creme Reparador de Barreira"
        hidden_patterns.append(
            "Preferência detectada por veículos nutritivos reparadores de barreira lipídica (Ceramidas + Ésteres leves)."
        )
    elif skin_type == "oleosa":
        enforced_texture = "Textura Gel-Creme Toque Seco / Fluido Matificante Não-Comedogênico"
        hidden_patterns.append("Pele oleosa: priorizando veículos com sílicas matificantes e baixa carga oleosa.")
    elif skin_type == "seca":
        enforced_texture = "Textura Creme Acetinado / Loção Hidratante Biomimética"
        hidden_patterns.append("Pele seca: priorizando veículos umectantes e oclusivos leves (Ácido Hialurônico + Esqualano).")
    else:
        enforced_texture = "Textura Gel-Creme Equilibrada / Fluido Confortável de Rápida Absorção"
        hidden_patterns.append("Pele mista/normal: priorizando equilíbrio hidrolipídico em veículo gel-creme leve.")

    if has_tinted_in_loved:
        hidden_patterns.append(
            "Protetor solar com cor detectado entre os favoritos: flag automática de busca com pigmento (`auto_tinted_flag=True`) ativada!"
        )
    elif state.get("prefers_tinted_sunscreen"):
        hidden_patterns.append(
            "Preferência explícita por fotoproteção com cor ativada para roteamento Findation."
        )

    # Rule 3: Map beneficial actives from active problems, target problem, and user reasons
    beneficial_actives: List[str] = []
    for prob in active_problems:
        beneficial_actives.extend(PROBLEM_TO_ACTIVES_MAP.get(prob, []))

    combined_goals_and_reasons = " ".join(
        [target_problem] + [p.user_reason or "" for p in loved]
    ).lower()
    if any(w in combined_goals_and_reasons for w in ["mancha", "melasma", "clarear", "hiperpigment", "uniformiz"]):
        beneficial_actives.extend(["Ácido Tranexâmico", "Alfa-Arbutin", "Niacinamida", "Vitamina C"])
    if any(w in combined_goals_and_reasons for w in ["acne", "espinha", "cravo", "oleos", "poro", "brilho", "seco"]):
        beneficial_actives.extend(["Ácido Salicílico 2%", "Niacinamida 10%", "Zinco PCA", "Sílica Matificante"])
    if any(w in combined_goals_and_reasons for w in ["rugas", "linha", "firmeza", "anti-idade", "retinol"]):
        beneficial_actives.extend(["Retinol / Retinaldeído", "Peptídeos Matrixyl", "Ácido Hialurônico"])
    if any(w in combined_goals_and_reasons for w in ["sensibilidade", "vermelhidão", "rosácea", "barreira", "acalma"]):
        beneficial_actives.extend(["Pantenol B5", "Madecassoside", "Ceramidas Biomiméticas", "Ácido Azelaico"])

    if not beneficial_actives:
        if skin_type in ("oleosa", "mista"):
            beneficial_actives = ["Niacinamida", "Zinco PCA", "Ácido Hialurônico de Baixo Peso Molecular", "Pantenol"]
        else:
            beneficial_actives = ["Ceramidas Essenciais", "Pantenol (Pró-Vitamina B5)", "Ácido Hialurônico", "Esqualano Vegetal"]

    # Rule 4: Forbidden ingredients based on skin sensitivity, disliked items, AND user complaint reasons
    forbidden_ingredients: List[str] = []
    if is_sensitive:
        forbidden_ingredients.extend(
            [
                "Álcool Desnaturado (Alcohol Denat.) em alta concentração",
                "Fragrância Sintética Forte / Parfum Alergênico (Linalool, Limonene)",
                "Óleos Essenciais Voláteis Irritantes",
                "Lauril Sulfato de Sódio (SLS) agressivo",
            ]
        )
    if skin_type in ("oleosa", "mista") or "acne" in active_problems or "comedoes" in active_problems:
        forbidden_ingredients.extend(
            [
                "Miristato de Isopropila (Altamente Comedogênico)",
                "Óleo Mineral Pesado / Petrolato Oclusivo em veículo facial diurno",
            ]
        )

    for item in loved + disliked:
        d_label = item.official_name or item.raw_input
        neg_text_low = " ".join(item.extracted_negatives).lower() if item.extracted_negatives else (
            (item.user_reason or "").lower() if item.sentiment == "disliked" else ""
        )
        tex_low = (item.predominant_texture or "").lower()
        if (item.sentiment == "disliked" and "creme rico" in tex_low) or any(
            k in neg_text_low for k in ["pesad", "oleos", "gorduros", "pegajos", "espinha", "acne"]
        ):
            forbidden_ingredients.append(f"Veículos oclusivos/comedogênicos similares a '{d_label}'")
        if any(k in neg_text_low for k in ["selar com pó", "selar com po", "precisa selar", "necessidade de selar"]):
            forbidden_ingredients.append(f"Emolientes de brilho residual que exigem selagem com pó (ressalva em '{d_label}')")
            hidden_patterns.append(
                f"Ressalva detectada em '{d_label}' ('necessidade de selar com pó'): priorizando sílicas matificantes de toque seco imediato."
            )
        if any(k in neg_text_low for k in ["white cast", "esbranquiçad", "fantasminha"]):
            forbidden_ingredients.append(f"Filtros minerais não-micronizados que deixam white cast (reclamação em '{d_label}')")
        if any(k in neg_text_low for k in ["cheiro", "perfume", "fragrância", "fragrancia", "enjoa"]):
            forbidden_ingredients.append(f"Fragrância / Parfum cosmético (reclamação direta em '{d_label}')")
        if any(k in neg_text_low for k in ["arde", "olho", "lacrimej", "irrit", "pinic", "vermelh"]):
            forbidden_ingredients.append(f"Filtros UV não-oftalmológicos / Álcool irritante ocular (reclamação em '{d_label}')")
        if any(k in neg_text_low for k in ["esfarel", "rolinho", "craquel", "abre"]):
            forbidden_ingredients.append(f"Polímeros filmógenos incompatíveis que esfarelam/craquelam (reclamação em '{d_label}')")
        if any(k in neg_text_low for k in ["oxid", "laranja", "escurec"]):
            forbidden_ingredients.append(f"Pigmentos não-revestidos propensos à oxidação (reclamação em '{d_label}')")

    # Optional enrichment via OpenAI GPT-4o-mini / Gemini if key is configured
    llm = get_openai_llm(smart=False) or get_gemini_llm()
    if llm:
        try:
            structured_llm = llm.with_structured_output(FormulationAnalysisOutput)
            loved_str = (
                " | ".join(
                    f"{p.raw_input} -> {p.official_name} [{p.predominant_texture}]"
                    + (f" (Por que ama: {p.user_reason})" if p.user_reason else "")
                    for p in loved
                )
                or "Nenhum"
            )
            disliked_str = (
                " | ".join(
                    f"{p.raw_input} -> {p.official_name} [{p.predominant_texture}]"
                    + (f" (Reclamação/Por que não gosta: {p.user_reason})" if p.user_reason else "")
                    for p in disliked
                )
                or "Nenhum"
            )
            prompt = (
                "Você é o Agente 1B (Formulação, Textura & Ativos - Química Cosmética).\n"
                f"Pele: {skin_type} | Sensível: {is_sensitive} | Orçamento: R$ {budget:.2f} | Problemas: {active_problems} | Alvo: {target_problem}\n"
                f"Favoritos do usuário e motivos: {loved_str}\n"
                f"Não Favoritos do usuário e reclamações: {disliked_str}\n"
                f"Flag protetor com cor detectada: {auto_tinted_flag}\n"
                "Gere a regra de textura obrigatória, padrões ocultos, ativos benéficos, ingredientes proibidos e queries técnicas."
            )
            llm_res: FormulationAnalysisOutput = structured_llm.invoke(prompt)
            if llm_res:
                llm_res.auto_tinted_flag = auto_tinted_flag
                return {"formulation_output": llm_res}
        except Exception as exc:
            logger.warning("Agent 1B structured output fallback: %s", exc)

    # Build technical formulation queries
    top_actives = " ".join(beneficial_actives[:2])
    if flow_type == "skincare_rotina":
        solar_qualifier = "com cor" if auto_tinted_flag else "incolor toque seco"
        queries = [
            f"gel de limpeza facial {skin_type} sem sulfato ceramidas glicerina",
            f"hidratante facial gel aquoso {top_actives} oil free",
            f"protetor solar facial fluido {solar_qualifier} FPS 50 FPS 60 {skin_type}",
        ]
    elif flow_type == "skincare_especifico":
        tint_q = "com cor" if (category == "protetor_solar" and auto_tinted_flag) else ""
        queries = [
            f"{category} {tint_q} {top_actives} textura gel fluido pele {skin_type}".strip(),
            f"{category} dermocosmético hipoalergênico sem fragrância pele {skin_type}",
        ]
    else:
        chromatic = state.get("chromatic_expectation") or ""
        queries = [
            f"{category} maquiagem {chromatic} pele {skin_type} não comedogênico".strip(),
            f"{category} dermatologicamente testado longa duração pele {skin_type}",
        ]

    return {
        "formulation_output": FormulationAnalysisOutput(
            enforced_texture_rule=enforced_texture,
            detected_hidden_patterns=hidden_patterns,
            beneficial_actives=list(dict.fromkeys(beneficial_actives))[:6],
            forbidden_ingredients=list(dict.fromkeys(forbidden_ingredients))[:6],
            auto_tinted_flag=auto_tinted_flag,
            formulation_search_queries=queries,
        )
    }
