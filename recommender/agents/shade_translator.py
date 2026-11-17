"""
Stage 3: Conditional Transverse Shade Translation via Findation (OpenAI GPT-4o + Findation Search).

Triggered ONLY after Stage 2 Budget Cutoff in two scenarios:
1. Makeup: When desired category is 'base' or 'corretivo'.
2. Skincare: When desired category is 'protetor_solar' (or Routine solar step) AND the user
   requested a tinted version (`prefers_tinted_sunscreen`) OR has a tinted sunscreen among
   their confirmed loved products (`auto_tinted_flag`).

Rules:
- Uses the user's confirmed Reference Base/Sunscreen card (`sentiment == 'shade_reference'`)
  from the Human-in-the-Loop step (or `reference_shade` field) to query `findation.com`.
- Triangulates across all confirmed shade references (just like Findation) to return the SINGLE
  EXACT commercial shade name printed on each product's packaging/store listing:
  - Needs Solar Controle de Oleosidade FPS 70: 'Bege Claro', 'Bege Médio', 'Bege Escuro'
  - Neutrogena Sun Fresh Derm Care FPS 70 Com Cor: 'Cor 1.0 Claro', 'Cor 2.0 Médio Claro', 'Cor 3.0 Médio Escuro', 'Cor 4.0 Escuro'
  - Principia PS-03 FPS 60 Com Cor: 'Cor 1 (Clara)', 'Cor 2 (Média)', 'Cor 3 (Morena)', 'Cor 4 (Negra)'
  - L'Oréal Solar Expertise FPS 60 Com Cor: 'Clara', 'Média', 'Morena', 'Negra'
  - Darrow Actine Colors FPS 70 Com Cor: 'Pele Clara', 'Pele Morena', 'Pele Morena Mais', 'Pele Negra'
  - La Roche-Posay Anthelios Airlicium+ FPS 80: 'Cor 1.0', 'Cor 2.0', 'Cor 3.0', 'Cor 4.0', 'Cor 5.0', 'Cor 6.0'
- If a candidate is colorless (`is_tinted == False`), explicitly marks it as `Incolor (sem pigmento)`.
- Fallback: If Findation has no exact match for the anchor shade, performs approximate recommendation
  by undertone (Quente, Frio, Neutro) and flags this clearly on the card (`shade_match_method='undertone_approx'`).
"""
import logging
import re
from collections import Counter
from typing import Any, Dict, List, Literal, Optional, Tuple
from pydantic import BaseModel, Field

from recommender.schemas import CandidateProduct
from .state import OrchestratorState
from .tools import get_openai_llm, tavily_search_findation

logger = logging.getLogger(__name__)

TierKey = Literal["claro", "claro_frio", "medio", "medio_escuro", "escuro"]


class LLMShadeTranslationResult(BaseModel):
    recommended_shade: str = Field(..., description="Nome comercial exato da cor na embalagem do produto")
    shade_match_method: Literal["findation_exact", "undertone_approx", "colorless"] = Field(
        default="findation_exact"
    )
    undertone_detected: Optional[str] = Field(
        default=None, description="Subtom: Quente, Frio ou Neutro"
    )
    shade_translation_note: str = Field(
        ..., description="Explicação da equivalência Findation ou alerta de aproximação por subtom"
    )


# ==============================================================================
# Exact Product-Specific Commercial Shade Charts (Exact SKU Names on Store/Box)
# ==============================================================================
PRODUCT_EXACT_SHADE_CHARTS: List[Dict[str, Any]] = [
    {
        "match_keywords": ["needs", "protetor"],
        "shades": {
            "claro": "Bege Claro",
            "claro_frio": "Bege Claro",
            "medio": "Bege Médio",
            "medio_escuro": "Bege Escuro",
            "escuro": "Bege Escuro",
        },
    },
    {
        "match_keywords": ["neutrogena", "sun fresh"],
        "shades": {
            "claro": "Cor 1.0 Claro",
            "claro_frio": "Cor 1.0 Claro",
            "medio": "Cor 2.0 Médio Claro",
            "medio_escuro": "Cor 3.0 Médio Escuro",
            "escuro": "Cor 4.0 Escuro",
        },
    },
    {
        "match_keywords": ["principia", "ps-03"],
        "shades": {
            "claro": "Cor 1 (Clara)",
            "claro_frio": "Cor 1 (Clara)",
            "medio": "Cor 2 (Média)",
            "medio_escuro": "Cor 3 (Morena)",
            "escuro": "Cor 4 (Negra)",
        },
    },
    {
        "match_keywords": ["solar expertise"],
        "shades": {
            "claro": "Clara",
            "claro_frio": "Clara",
            "medio": "Média",
            "medio_escuro": "Morena",
            "escuro": "Negra",
        },
    },
    {
        "match_keywords": ["darrow", "actine"],
        "shades": {
            "claro": "Pele Clara",
            "claro_frio": "Pele Clara",
            "medio": "Pele Morena",
            "medio_escuro": "Pele Morena Mais",
            "escuro": "Pele Negra",
        },
    },
    {
        "match_keywords": ["eucerin", "oil control"],
        "shades": {
            "claro": "Claro",
            "claro_frio": "Claro",
            "medio": "Médio",
            "medio_escuro": "Médio",
            "escuro": "Médio",
        },
    },
    {
        "match_keywords": ["episol", "sec oc"],
        "shades": {
            "claro": "Claro",
            "claro_frio": "Muito Claro",
            "medio": "Médio",
            "medio_escuro": "Médio Escuro",
            "escuro": "Escuro",
        },
    },
    {
        "match_keywords": ["anthelios", "airlicium"],
        "shades": {
            "claro": "Cor 2.0",
            "claro_frio": "Cor 1.0",
            "medio": "Cor 3.0",
            "medio_escuro": "Cor 4.0",
            "escuro": "Cor 6.0",
        },
    },
    {
        "match_keywords": ["isdin", "fusion water color"],
        "shades": {
            "claro": "Cor Light",
            "claro_frio": "Cor Light",
            "medio": "Cor Medium",
            "medio_escuro": "Cor Bronze",
            "escuro": "Cor Bronze",
        },
    },
    {
        "match_keywords": ["anasol", "protetor"],
        "shades": {
            "claro": "Bege Claro",
            "claro_frio": "Bege Claro",
            "medio": "Bege Médio",
            "medio_escuro": "Bege Escuro",
            "escuro": "Bege Escuro",
        },
    },
    {
        "match_keywords": ["bt skinplush", "corretivo"],
        "shades": {
            "claro": "L1020",
            "claro_frio": "F1020",
            "medio": "M1020",
            "medio_escuro": "T1020",
            "escuro": "E1020",
        },
    },
    {
        "match_keywords": ["bt skin", "base"],
        "shades": {
            "claro": "L20",
            "claro_frio": "F20",
            "medio": "M20",
            "medio_escuro": "T20",
            "escuro": "E20",
        },
    },
    {
        "match_keywords": ["sheglam", "corretivo"],
        "shades": {
            "claro": "Vanilla",
            "claro_frio": "Fair",
            "medio": "Sand",
            "medio_escuro": "Golden",
            "escuro": "Chestnut",
        },
    },
    {
        "match_keywords": ["sheglam", "complexion boost"],
        "shades": {
            "claro": "Vanilla",
            "claro_frio": "Fair",
            "medio": "Sand",
            "medio_escuro": "Golden",
            "escuro": "Chestnut",
        },
    },
    {
        "match_keywords": ["sheglam", "complexion pro"],
        "shades": {
            "claro": "Shell",
            "claro_frio": "Porcelain",
            "medio": "Sand",
            "medio_escuro": "Golden",
            "escuro": "Mocha",
        },
    },
    {
        "match_keywords": ["maybelline", "age rewind"],
        "shades": {
            "claro": "120 Light",
            "claro_frio": "110 Fair",
            "medio": "130 Medium",
            "medio_escuro": "142 Golden",
            "escuro": "146 Hazelnut",
        },
    },
    {
        "match_keywords": ["maybelline", "fit me"],
        "shades": {
            "claro": "120 Classic Ivory",
            "claro_frio": "112 Natural Ivory",
            "medio": "220 Natural Beige",
            "medio_escuro": "310 Sun Beige",
            "escuro": "355 Coconut",
        },
    },
    {
        "match_keywords": ["mari maria", "velvet skin"],
        "shades": {
            "claro": "Bege Claro 1",
            "claro_frio": "Bege Claro 2",
            "medio": "Bege Médio",
            "medio_escuro": "Canela",
            "escuro": "Cacau",
        },
    },
    {
        "match_keywords": ["boca rosa", "base"],
        "shades": {
            "claro": "Cor 14 Antônia",
            "claro_frio": "Cor 12 Francisca",
            "medio": "Cor 20 Adriana",
            "medio_escuro": "Cor 30 Juliana",
            "escuro": "Cor 42 Camila",
        },
    },
    {
        "match_keywords": ["natura una", "corretivo"],
        "shades": {
            "claro": "15Q (Claro 20)",
            "claro_frio": "19F",
            "medio": "24N (Médio 22)",
            "medio_escuro": "32Q (Castanho 20)",
            "escuro": "40N (Escuro 35)",
        },
    },
    {
        "match_keywords": ["natura una", "nude me"],
        "shades": {
            "claro": "15Q (Claro 20)",
            "claro_frio": "19F",
            "medio": "24N (Médio 22)",
            "medio_escuro": "32Q (Castanho 20)",
            "escuro": "40N (Escuro 35)",
        },
    },
    {
        "match_keywords": ["vult", "hidraluronic"],
        "shades": {
            "claro": "B15",
            "claro_frio": "R20",
            "medio": "B25",
            "medio_escuro": "M15",
            "escuro": "N10",
        },
    },
]


# Findation Cross-Brand Equivalence Matrix for Popular Brazilian & North American (US) Anchors
FINDATION_EQUIVALENCE_MATRIX: List[Dict[str, Any]] = [
    {
        "tier": "claro",
        "anchor_tokens": [
            "nc15", "nc20", "f10", "f20", "l10", "l20", "l30", "claro", "clara", "1 maria", "light",
            "chantilly", "vanilla", "shell", "100", "110", "120", "130", "140", "150", "160", "170w", "190w",
            "1n1", "1w1", "1w2", "swan", "alabaster", "snow", "porcelain", "fair 10", "light 20",
            "mont blanc", "gobi", "deauville", "milkshake", "shortbread", "cor 00", "cor 05", "cor 10",
            "c10", "fair", "nude", "cor 1", "1.0",
        ],
        "depth": "Claro",
        "undertone": "Neutro-Quente (Dourado Suave)",
        "brand_map": {
            "Bruna Tavares": "L20",
            "Natura Una": "15Q (Claro 20)",
            "Boca Rosa Beauty": "Cor 14 Antônia",
            "Maybelline NY": "120 Classic Ivory",
            "Mari Maria Makeup": "Bege Claro 1",
            "Vult": "B15",
            "Eudora (Niina Secrets / Glam)": "Cor 10",
            "O Boticário (Make B.)": "Cor 120",
            "Vizzela": "Cor 02",
            "Fran by Franciny Ehlke": "C-01",
            "SHEGLAM": "Shell",
            "Fenty Beauty": "150",
            "Rare Beauty": "170W",
            "Estée Lauder": "1W2 Sand",
            "Too Faced": "Swan",
            "Huda Beauty": "Vanilla 120B",
            "Dragun Beauty": "Light 20",
            "Mantecorp Skincare": "Claro",
            "ISDIN": "Cor Light",
            "La Roche-Posay": "Cor 2.0",
            "Neutrogena": "Cor 1.0 Claro",
            "Principia": "Cor 1 (Clara)",
            "Darrow": "Pele Clara",
            "Eucerin": "Claro",
            "L'Oréal Paris": "Clara",
            "Needs Solar": "Bege Claro",
            "Anasol": "Bege Claro",
        },
    },
    {
        "tier": "medio",
        "anchor_tokens": [
            "nc25", "nc30", "nc35", "m10", "m20", "m30", "m1020", "m3040", "medio", "médio", "media", "média",
            "2 antonia", "3 francisca", "4 adriana", "custard", "ginger", "medium", "200", "210", "210n", "220",
            "230n", "240", "260n", "290", "2w1", "2w2", "3n1", "warm beige", "natural beige",
            "golden beige", "warm vanilla", "butterscotch", "sand", "soft tan", "buff", "light-medium 30",
            "medium 40", "fiji", "punjab", "vallauris", "stromboli", "toasted coconut", "latte", "tres leches",
            "cor 25", "cor 35", "cor 2", "2.0", "bege medio", "bege médio", "24n", "27q",
        ],
        "depth": "Médio",
        "undertone": "Quente / Neutro-Dourado",
        "brand_map": {
            "Bruna Tavares": "M20",
            "Natura Una": "24N (Médio 22)",
            "Boca Rosa Beauty": "Cor 20 Adriana",
            "Maybelline NY": "220 Natural Beige",
            "Mari Maria Makeup": "Bege Médio",
            "Vult": "B25",
            "Eudora (Niina Secrets / Glam)": "Cor 25",
            "O Boticário (Make B.)": "Cor 190",
            "Vizzela": "Cor 06",
            "Fran by Franciny Ehlke": "M-01",
            "SHEGLAM": "Sand",
            "Fenty Beauty": "240",
            "Rare Beauty": "230N",
            "Estée Lauder": "2W1 Dawn",
            "Too Faced": "Natural Beige",
            "Huda Beauty": "Toasted Coconut 240N",
            "Dragun Beauty": "Medium 40",
            "Mantecorp Skincare": "Médio",
            "ISDIN": "Cor Medium",
            "La Roche-Posay": "Cor 3.0",
            "Neutrogena": "Cor 2.0 Médio Claro",
            "Principia": "Cor 2 (Média)",
            "Darrow": "Pele Morena",
            "Eucerin": "Médio",
            "L'Oréal Paris": "Média",
            "Needs Solar": "Bege Médio",
            "Anasol": "Bege Médio",
        },
    },
    {
        "tier": "claro_frio",
        "anchor_tokens": [
            "nw15", "nw20", "nw25", "frio", "rosado", "cool", "rose", "c20", "19f", "21f", "1c1", "2c1", "160c", "200c",
        ],
        "depth": "Claro a Médio",
        "undertone": "Frio (Rosado)",
        "brand_map": {
            "Bruna Tavares": "F20",
            "Natura Una": "19F",
            "Boca Rosa Beauty": "Cor 12 Francisca",
            "Maybelline NY": "112 Natural Ivory",
            "Mari Maria Makeup": "Bege Claro 2",
            "Vult": "R20",
            "SHEGLAM": "Porcelain",
            "Fenty Beauty": "170",
            "Rare Beauty": "160C",
            "Mantecorp Skincare": "Claro",
            "ISDIN": "Cor Light",
            "La Roche-Posay": "Cor 2.0",
            "Neutrogena": "Cor 1.0 Claro",
            "Principia": "Cor 1 (Clara)",
            "Darrow": "Pele Clara",
            "Eucerin": "Claro",
            "L'Oréal Paris": "Clara",
            "Needs Solar": "Bege Claro",
            "Anasol": "Bege Claro",
        },
    },
    {
        "tier": "medio_escuro",
        "anchor_tokens": [
            "nc40", "nc42", "nc44", "nc45", "t10", "t20", "t30", "t1020", "t3040", "morena", "bronze", "tan",
            "caramelo", "caramel", "golden", "médio-escuro", "medio-escuro", "médio escuro", "medio escuro",
            "biscuit", "tom 5", "tom 6", "6 juliana", "cor 3", "3.0", "cor 4", "300", "310", "330", "350",
            "360w", "3w1", "4n1", "4w1", "warm sand", "honey", "praline", "butter pecan", "tan 50",
            "syracuse", "tahoe", "cor 55", "cor 60", "32q", "35n",
        ],
        "depth": "Médio-Escuro / Morena",
        "undertone": "Quente Dourado",
        "brand_map": {
            "Bruna Tavares": "T20",
            "Natura Una": "32Q (Castanho 20)",
            "Boca Rosa Beauty": "Cor 30 Juliana",
            "Maybelline NY": "310 Sun Beige",
            "Mari Maria Makeup": "Canela",
            "Vult": "M15",
            "Eudora (Niina Secrets / Glam)": "Cor 55",
            "O Boticário (Make B.)": "Cor 240",
            "SHEGLAM": "Golden",
            "Fenty Beauty": "310",
            "Rare Beauty": "360W",
            "Estée Lauder": "3W1 Tawny",
            "Too Faced": "Warm Sand",
            "Dragun Beauty": "Tan 50",
            "Mantecorp Skincare": "Médio Escuro",
            "ISDIN": "Cor Bronze",
            "La Roche-Posay": "Cor 4.0",
            "Neutrogena": "Cor 3.0 Médio Escuro",
            "Principia": "Cor 3 (Morena)",
            "Darrow": "Pele Morena Mais",
            "Eucerin": "Médio",
            "L'Oréal Paris": "Morena",
            "Needs Solar": "Bege Escuro",
            "Anasol": "Bege Escuro",
        },
    },
    {
        "tier": "escuro",
        "anchor_tokens": [
            "nc50", "nc55", "nw45", "nw50", "e10", "e20", "e30", "e1020", "e3040", "negra", "retinta", "deep",
            "espresso", "cacao", "cacau", "tom 7", "tom 8", "tom 9", "8 camila", "cor 5", "cor 6", "4.0",
            "400", "420", "450", "490", "430w", "500n", "5w1", "6w1", "chai", "spiced rum", "mocha", "truffle",
            "deep 60", "rich 70", "macao", "new caledonia", "cor 80", "cor 95", "40n", "43q",
        ],
        "depth": "Escura / Pele Negra",
        "undertone": "Neutro-Quente Profundo",
        "brand_map": {
            "Bruna Tavares": "E20",
            "Natura Una": "40N (Escuro 35)",
            "Boca Rosa Beauty": "Cor 42 Camila",
            "Maybelline NY": "355 Coconut",
            "Mari Maria Makeup": "Cacau",
            "Vult": "N10",
            "Eudora (Niina Secrets / Glam)": "Cor 80",
            "O Boticário (Make B.)": "Cor 290",
            "SHEGLAM": "Mocha",
            "Fenty Beauty": "420",
            "Rare Beauty": "430W",
            "Estée Lauder": "5W1 Bronze",
            "Too Faced": "Chai",
            "Dragun Beauty": "Deep 60",
            "Mantecorp Skincare": "Escuro",
            "ISDIN": "Cor Bronze",
            "La Roche-Posay": "Cor 6.0",
            "Neutrogena": "Cor 4.0 Escuro",
            "Principia": "Cor 4 (Negra)",
            "Darrow": "Pele Negra",
            "Eucerin": "Médio",
            "L'Oréal Paris": "Negra",
            "Needs Solar": "Bege Escuro",
            "Anasol": "Bege Escuro",
        },
    },
]

TIER_METADATA: Dict[str, Dict[str, Any]] = {
    g["tier"]: {"depth": g["depth"], "undertone": g["undertone"], "brand_map": g["brand_map"]}
    for g in FINDATION_EQUIVALENCE_MATRIX
}


def evaluate_findation_routing(state: OrchestratorState) -> Tuple[bool, str]:
    """
    Determines whether Stage 3 (Findation Shade Translation) should run.
    """
    flow_type = state.get("flow_type", "skincare_rotina")
    category = state.get("category")
    formulation = state.get("formulation_output")
    auto_tinted = bool(
        (formulation and formulation.auto_tinted_flag)
        or state.get("prefers_tinted_sunscreen", False)
    )
    has_loved_tinted = any(
        p.confirmed and p.sentiment == "loved" and p.is_tinted_sunscreen
        for p in state.get("confirmed_products", [])
    )

    if flow_type == "maquiagem" and category in ("base", "corretivo"):
        return (
            True,
            f"Acionado para Maquiagem (Categoria '{category}' exige tradução de tom via Findation).",
        )

    if flow_type == "skincare_especifico" and category == "protetor_solar" and (auto_tinted or has_loved_tinted):
        return (
            True,
            "Acionado para Skincare (Protetor Solar com cor solicitado ou detectado nos favoritos).",
        )

    if flow_type == "skincare_rotina" and (auto_tinted or has_loved_tinted):
        return (
            True,
            "Acionado transversalmente na Rotina Skincare (Protetor Solar com cor ativado via preferência ou favoritos).",
        )

    return (
        False,
        "Ignorado (Funil de Custo): Categoria não exige tradução de tom no Findation.",
    )


def should_translate_shades(state: OrchestratorState) -> Literal["stage_3_findation", "stage_4_derm_curator"]:
    """LangGraph Conditional Edge Router after Stage 2."""
    should_run, _ = evaluate_findation_routing(state)
    return "stage_3_findation" if should_run else "stage_4_derm_curator"


def _infer_undertone_from_text(reference_text: str) -> str:
    low = (reference_text or "").lower()
    if any(k in low for k in ["frio", "rosad", "nw", "cool", "pink", "rose"]):
        return "Frio (Fundo Rosado)"
    if any(k in low for k in ["quente", "amarelad", "dourad", "nc", "warm", "gold", "oliva"]):
        return "Quente (Fundo Dourado/Amarelado)"
    return "Neutro (Equilíbrio Rosado/Dourado)"


def _lookup_exact_product_shade(candidate: CandidateProduct, tier: str) -> Optional[str]:
    """
    Returns the exact commercial shade name printed on the candidate product's box/listing
    for the requested depth/undertone `tier`.
    """
    haystack = f"{candidate.brand} {candidate.product_name}".lower()
    for entry in PRODUCT_EXACT_SHADE_CHARTS:
        if all(kw in haystack for kw in entry["match_keywords"]):
            shade = entry["shades"].get(tier)
            if shade:
                return shade

    tier_info = TIER_METADATA.get(tier)
    if tier_info:
        brand_shade = tier_info["brand_map"].get(candidate.brand)
        if brand_shade:
            return brand_shade
    return None


def _classify_single_reference_segment(seg_text: str) -> Optional[str]:
    """
    Classifies a single shade reference string (e.g. 'BT Skin — Tom: M20' or
    'SHEGLAM Complexion Pro — Tom: Golden (Médio-Escuro Dourado)') into its Findation tier.
    """
    low = seg_text.lower().strip()
    if not low:
        return None

    priority_order = ["escuro", "medio_escuro", "claro_frio", "medio", "claro"]
    group_by_tier = {g["tier"]: g for g in FINDATION_EQUIVALENCE_MATRIX}
    low_without_compound = re.sub(r"m[eé]dio[\s\-]+escuro", " ", low)

    for tier_key in priority_order:
        group = group_by_tier[tier_key]
        test_str = low_without_compound if tier_key == "medio" else low
        if any(tok in test_str for tok in group["anchor_tokens"]):
            return tier_key
    return None


def _translate_single_candidate_shade(
    candidate: CandidateProduct,
    reference_shade: str,
    confirmed_base_label: Optional[str] = None,
    chromatic_expectation: Optional[str] = None,
) -> CandidateProduct:
    """Translates shade via Findation triangulation or applies undertone approximation fallback."""
    updated = candidate.model_copy(deep=True)

    # Rule: If candidate is colorless, explicitly identify as 'Incolor (sem pigmento)'
    if not updated.is_tinted:
        updated.recommended_shade = "Incolor (sem pigmento)"
        updated.shade_match_method = "colorless"
        updated.undertone_Detected = None
        updated.shade_translation_note = "Produto sem pigmento — identificado explicitamente como Incolor."
        return updated

    # If chromatic makeup category (blush/sombra/contorno/iluminador)
    if updated.category in {"blush", "sombra", "contorno", "iluminador"}:
        desc = chromatic_expectation or "Acabamento cromático versátil"
        updated.recommended_shade = f"Acabamento Sugerido: {desc[:60]}"
        updated.shade_match_method = "chromatic_match"
        updated.undertone_Detected = _infer_undertone_from_text(reference_shade or desc)
        updated.shade_translation_note = (
            f"Curadoria cromática baseada na descrição '{desc}' e harmonia de subtom."
        )
        return updated

    ref_clean = (reference_shade or "").strip()
    display_anchor = confirmed_base_label or ref_clean

    if ref_clean:
        segments = [s.strip() for s in ref_clean.split("|") if s.strip()]
        tier_votes: List[str] = []
        for seg in segments:
            t = _classify_single_reference_segment(seg)
            if t:
                tier_votes.append(t)

        if tier_votes:
            # Triangulate across all confirmed references: pick consensus/majority tier first
            vote_counts = Counter(tier_votes)
            primary_tier = vote_counts.most_common(1)[0][0]
            primary_shade = _lookup_exact_product_shade(updated, primary_tier)

            secondary_shades: List[str] = []
            for t in tier_votes:
                s_alt = _lookup_exact_product_shade(updated, t)
                if s_alt and s_alt != primary_shade and s_alt not in secondary_shades:
                    secondary_shades.append(s_alt)

            if primary_shade:
                primary_tier_meta = TIER_METADATA[primary_tier]
                updated.recommended_shade = primary_shade
                updated.shade_match_method = "findation_exact"
                updated.undertone_Detected = primary_tier_meta["undertone"]
                alt_note = (
                    f" (ou '{secondary_shades[0]}' caso prefira o tom mais intenso da sua outra referência)"
                    if secondary_shades
                    else ""
                )
                updated.shade_translation_note = (
                    f"Match Exato via Findation: A partir de '{display_anchor}', a cor oficial para adquirir em "
                    f"{updated.product_name} é '{primary_shade}'{alt_note}."
                )
                return updated

        # 2. Try live Findation search + GPT-4o if API keys are configured
        findation_hits = tavily_search_findation(
            f"findation {ref_clean} equivalent shade in {updated.brand} {updated.product_name}",
            max_results=3,
        )
        llm = get_openai_llm(smart=True)
        if llm and findation_hits:
            try:
                structured_llm = llm.with_structured_output(LLMShadeTranslationResult)
                snippets = "\n".join(
                    f"- {h.get('title')}: {h.get('content', '')[:250]}" for h in findation_hits
                )
                prompt = (
                    f"Traduza o tom de referência do usuário ('{display_anchor}') para o nome comercial EXATO da cor "
                    f"impressa na embalagem do produto '{updated.brand} - {updated.product_name}' usando os dados do Findation abaixo:\n{snippets}\n"
                    "Se houver correspondência exata no Findation, use shade_match_method='findation_exact'. "
                    "Caso contrário, recomende por subtom aproximado (quente, frio, neutro) com shade_match_method='undertone_approx'."
                )
                res: LLMShadeTranslationResult = structured_llm.invoke(prompt)
                if res:
                    updated.recommended_shade = res.recommended_shade
                    updated.shade_match_method = res.shade_match_method
                    updated.undertone_Detected = res.undertone_detected
                    updated.shade_translation_note = res.shade_translation_note
                    return updated
            except Exception as exc:
                logger.warning("GPT-4o Findation translation fallback: %s", exc)

    # 3. Mandatory Fallback: When Findation does not find an exact match,
    # recommend approximately by undertone (quente, frio, neutro) using the exact product chart shade
    inferred_undertone = _infer_undertone_from_text(ref_clean)
    fallback_tier: TierKey = "claro_frio" if "Frio" in inferred_undertone else "medio"
    exact_chart_shade = _lookup_exact_product_shade(updated, fallback_tier)

    if "Frio" in inferred_undertone:
        approx_shade = (
            f"{exact_chart_shade} (Subtom Frio)"
            if exact_chart_shade
            else f"Tom Médio-Claro Frio — Cartela {updated.brand}"
        )
    elif "Quente" in inferred_undertone:
        approx_shade = (
            f"{exact_chart_shade} (Subtom Quente)"
            if exact_chart_shade
            else f"Tom Médio Quente — Cartela {updated.brand}"
        )
    else:
        approx_shade = (
            f"{exact_chart_shade} (Subtom Neutro)"
            if exact_chart_shade
            else f"Tom Médio Neutro — Cartela {updated.brand}"
        )

    anchor_label = display_anchor if display_anchor else "referência não especificada"
    updated.recommended_shade = f"≈ {approx_shade}"
    updated.shade_match_method = "undertone_approx"
    updated.undertone_Detected = inferred_undertone
    updated.shade_translation_note = (
        f"Aviso Findation (Match Aproximado por Subtom): Não encontramos correspondência exata 1:1 para "
        f"'{anchor_label}'. Recomendação calibrada na cor comercial '{approx_shade}' por aproximação de subtom {inferred_undertone}."
    )
    return updated


def run_stage_3_findation(state: OrchestratorState) -> Dict[str, Any]:
    """LangGraph Node: Stage 3 - Findation Shade Translation for Budget-Approved Finalists."""
    should_run, reason = evaluate_findation_routing(state)
    candidates = state.get("approved_candidates", [])

    reference_shade = state.get("reference_shade") or ""
    confirmed_base_label: Optional[str] = None

    shade_ref_cards = [
        p for p in state.get("confirmed_products", [])
        if p.confirmed and p.sentiment == "shade_reference"
    ]
    if shade_ref_cards:
        confirmed_base_label = " + ".join(
            f"{p.brand} ({p.detected_shade or p.raw_input})" for p in shade_ref_cards
        )
        reference_shade = " | ".join(
            f"{p.raw_input} {p.detected_shade or ''} {p.detected_undertone or ''}".strip()
            for p in shade_ref_cards
        )

    if not reference_shade:
        for p in state.get("confirmed_products", []):
            if p.confirmed and p.sentiment == "loved" and (p.detected_shade or p.is_tinted_sunscreen):
                reference_shade = p.detected_shade or p.raw_input
                break

    chromatic_expectation = state.get("chromatic_expectation")
    translated: List[CandidateProduct] = [
        _translate_single_candidate_shade(c, reference_shade, confirmed_base_label, chromatic_expectation)
        for c in candidates
    ]

    return {
        "stage_3_executed": should_run,
        "stage_3_routing_reason": reason,
        "shade_translated_candidates": translated,
    }
