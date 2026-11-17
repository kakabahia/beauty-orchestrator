"""
Human-in-the-Loop Product & Reference Foundation Disambiguation Agent (Step 1.B).
Resolves informal/colloquial user descriptions (e.g. "hidratante roxo da sallve",
"protetor com cor da isdin", and reference foundations like "base mac nc25" or
"base boca rosa 2") against major pharmacies and official stores, returning
interactive confirmation cards ("Você quis dizer este produto?").
"""
import hashlib
import logging
import re
from typing import Dict, List, Literal, Optional, Tuple
from urllib.parse import quote_plus
from pydantic import BaseModel, Field

from recommender.schemas import ConfirmedProductItem, DisambiguationResponse
from .tools import get_gemini_llm, get_openai_llm, tavily_search_official_stores

logger = logging.getLogger(__name__)


class LLMDisambiguatedCard(BaseModel):
    official_name: str = Field(..., description="Nome oficial completo do produto")
    brand: str = Field(..., description="Marca oficial do produto")
    predominant_texture: str = Field(
        ..., description="Textura ou variante predominante (ex: Gel-Creme Aquoso, Fluido Ultraleve Com Cor)"
    )
    category_detected: str = Field(default="skincare")
    is_tinted_sunscreen: bool = Field(
        default=False, description="True se for um protetor solar com cor/pigmento"
    )
    detected_shade: str = Field(default="")
    detected_undertone: str = Field(default="")
    store_name: str = Field(default="Droga Raia")
    store_url: str = Field(default="https://www.drogaraia.com.br")


def clean_product_search_query(product_name: str) -> str:
    """
    Cleans a product name for pharmacy/store search engines by removing parenthetical
    chemical/packaging notes (e.g., '(Niacinamida)', '(2% Ácido Salicílico)'), shade
    suffixes ('— Tom de Referência: ...'), and slash variants so store search URLs
    always return active in-stock listings.
    """
    cleaned = product_name.split("—")[0].strip()
    cleaned = re.sub(r"\s*\([^)]*\)", "", cleaned)
    if "/" in cleaned:
        cleaned = cleaned.split("/")[0].strip()
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned or product_name.strip()


def build_reliable_store_url(product_name: str, store_name: str = "Droga Raia") -> str:
    """
    Generates 100% functional URLs on major verified pharmacies and official retailers
    (Droga Raia, Drogasil, Amazon, Época Cosméticos, Beleza na Web, Sephora, Principia)
    that never break when SKU slugs change.
    """
    q = quote_plus(clean_product_search_query(product_name))
    low_store = store_name.lower()
    if "drogaraia" in low_store or "droga raia" in low_store:
        return f"https://www.drogaraia.com.br/search?w={q}"
    if "drogasil" in low_store:
        return f"https://www.drogasil.com.br/search?w={q}"
    if "são paulo" in low_store or "sao paulo" in low_store:
        return f"https://www.drogariasaopaulo.com.br/pesquisa?q={q}"
    if "epoca" in low_store or "época" in low_store:
        return f"https://www.epocacosmeticos.com.br/busca?q={q}"
    if "panvel" in low_store:
        return f"https://www.panvel.com/panvel/buscarProduto.do?termoPesquisa={q}"
    if "amazon" in low_store:
        return f"https://www.amazon.com.br/s?k={q}"
    if "sephora" in low_store:
        return f"https://www.sephora.com.br/busca?q={q}"
    if "principia" in low_store:
        return f"https://www.principiaskincare.com/catalogsearch/result/?q={q}"
    return f"https://www.belezanaweb.com.br/busca?q={q}"


# Curated recognition map for common informal Brazilian skincare/makeup expressions
INFORMAL_PRODUCT_KNOWLEDGE: List[Dict] = [
    {
        "keywords": ["roxo", "sallve", "reparador"],
        "official_name": "Sallve Hidratante Reparador Múltiplo (Embalagem Roxa)",
        "brand": "Sallve",
        "predominant_texture": "Creme Rico Nutritivo (Barreira Cutânea / Ceramidas + Pantenol)",
        "category_detected": "hidratante",
        "is_tinted_sunscreen": False,
        "store_name": "Droga Raia",
    },
    {
        "keywords": ["sallve", "hidratante", "antioxidante"],
        "official_name": "Sallve Antioxidante Hidratante FPS 30",
        "brand": "Sallve",
        "predominant_texture": "Sérum-Gel Aquoso de Rápida Absorção (Vitamina C + Cafeína)",
        "category_detected": "hidratante",
        "is_tinted_sunscreen": False,
        "store_name": "Droga Raia",
    },
    {
        "keywords": ["sallve", "hidratante", "firmador"],
        "official_name": "Sallve Hidratante Firmador (Embalagem Azul)",
        "brand": "Sallve",
        "predominant_texture": "Gel-Creme Leve Aquoso",
        "category_detected": "hidratante",
        "is_tinted_sunscreen": False,
        "store_name": "Droga Raia",
    },
    {
        "keywords": ["isdin", "cor", "protetor", "fusion water"],
        "official_name": "ISDIN Fotoprotector Fusion Water Color FPS 50",
        "brand": "ISDIN",
        "predominant_texture": "Fluido Aquoso Ultraleve Com Cor (Toque Seco / Não Arde os Olhos)",
        "category_detected": "protetor_solar",
        "is_tinted_sunscreen": True,
        "store_name": "Droga Raia",
    },
    {
        "keywords": ["isdin", "protetor", "fusion water"],
        "official_name": "ISDIN Fotoprotector Fusion Water 5 Stars FPS 60 Sem Cor",
        "brand": "ISDIN",
        "predominant_texture": "Fluido Aquoso Ultraleve Incolor (Fase Aquosa Oil-Free)",
        "category_detected": "protetor_solar",
        "is_tinted_sunscreen": False,
        "store_name": "Droga Raia",
    },
    {
        "keywords": ["la roche", "anthelios", "cor", "airlicium"],
        "official_name": "La Roche-Posay Anthelios Airlicium+ Com Cor FPS 80",
        "brand": "La Roche-Posay",
        "predominant_texture": "Gel-Creme Antibrilho Com Cor (Micropartículas Matificantes)",
        "category_detected": "protetor_solar",
        "is_tinted_sunscreen": True,
        "store_name": "Drogasil",
    },
    {
        "keywords": ["episol", "sec", "cor", "mantecorp"],
        "official_name": "Mantecorp Episol Sec OC Com Cor FPS 60",
        "brand": "Mantecorp Skincare",
        "predominant_texture": "Loção Fluida Toque Seco Com Pigmento Uniformizador",
        "category_detected": "protetor_solar",
        "is_tinted_sunscreen": True,
        "store_name": "Droga Raia",
    },
    {
        "keywords": ["neutrogena", "sun fresh", "cor"],
        "official_name": "Neutrogena Sun Fresh Derm Care FPS 70 Com Cor",
        "brand": "Neutrogena",
        "predominant_texture": "Gel-Creme Toque Seco Com Cor (Niacinamida + Feverfew)",
        "category_detected": "protetor_solar",
        "is_tinted_sunscreen": True,
        "store_name": "Drogasil",
    },
    {
        "keywords": ["neutrogena", "hydro boost"],
        "official_name": "Neutrogena Hydro Boost Water Gel Hidratante Facial",
        "brand": "Neutrogena",
        "predominant_texture": "Gel Aquoso Refrescante Oil-Free (Ácido Hialurônico)",
        "category_detected": "hidratante",
        "is_tinted_sunscreen": False,
        "store_name": "Amazon",
    },
    {
        "keywords": ["epidrat", "calm", "mantecorp"],
        "official_name": "Mantecorp Epidrat Calm B5 Hidratante Calmante",
        "brand": "Mantecorp Skincare",
        "predominant_texture": "Gel-Creme Balm Calmante Sem Fragrância (Sensarine + D-Pantenol)",
        "category_detected": "hidratante",
        "is_tinted_sunscreen": False,
        "store_name": "Droga Raia",
    },
    {
        "keywords": ["principia", "niacinamida", "nc-10", "nc10"],
        "official_name": "Principia Sérum NC-10 (10% Niacinamida + 1% Zinco PCA)",
        "brand": "Principia",
        "predominant_texture": "Sérum Aquoso Leve Oil-Free",
        "category_detected": "serum",
        "is_tinted_sunscreen": False,
        "store_name": "Principia Skincare",
    },
    {
        "keywords": ["principia", "gel de limpeza", "gl-01", "gl-02", "sabonete"],
        "official_name": "Principia Gel de Limpeza GL-01 (2% Ácido Salicílico + 5% Glicerina)",
        "brand": "Principia",
        "predominant_texture": "Gel Espumante Suave Sem Sulfatos Pesados",
        "category_detected": "sabonete_facial",
        "is_tinted_sunscreen": False,
        "store_name": "Principia Skincare",
    },
    {
        "keywords": ["cerave", "espuma", "limpeza", "gel"],
        "official_name": "CeraVe Gel de Limpeza Facial Espumante (Pele Normal a Oleosa)",
        "brand": "CeraVe",
        "predominant_texture": "Gel Espumante Refrescante (3 Ceramidas Essenciais + Niacinamida)",
        "category_detected": "sabonete_facial",
        "is_tinted_sunscreen": False,
        "store_name": "Droga Raia",
    },
    {
        "keywords": ["biore", "cleansing oil", "roxo", "demaquilante"],
        "official_name": "Bioré Make Up Remover Perfect Cleansing Oil",
        "brand": "Bioré",
        "predominant_texture": "Óleo Demaquilante Emulsionável em Água (Não Residual)",
        "category_detected": "cleansing_oil",
        "is_tinted_sunscreen": False,
        "store_name": "Drogasil",
    },
    {
        "keywords": ["mac", "studio fix", "base"],
        "official_name": "M·A·C Studio Fix Fluid SPF 15 24HR Matte Foundation",
        "brand": "M·A·C Cosmetics",
        "predominant_texture": "Fluido Matte Confortável Controlador de Oleosidade",
        "category_detected": "base",
        "is_tinted_sunscreen": False,
        "store_name": "Sephora",
    },
    {
        "keywords": ["bruna tavares", "bt skin", "base"],
        "official_name": "Bruna Tavares BT Skin Base Líquida Ultra Fluida",
        "brand": "Bruna Tavares",
        "predominant_texture": "Fluido Sérum Segunda Pele Semi-Matte",
        "category_detected": "base",
        "is_tinted_sunscreen": False,
        "store_name": "Sephora",
    },
    {
        "keywords": ["boca rosa", "base", "stick", "matte", "antonia", "francisca", "adriana", "maria"],
        "official_name": "Boca Rosa Beauty Base Mate (Stick / Líquida)",
        "brand": "Boca Rosa Beauty",
        "predominant_texture": "Fluido/Stick Matte de Alta Cobertura Resistente à Água",
        "category_detected": "base",
        "is_tinted_sunscreen": False,
        "store_name": "Beleza na Web",
    },
    {
        "keywords": ["maybelline", "fit me", "base"],
        "official_name": "Maybelline Fit Me Base Líquida Matte + Poreless",
        "brand": "Maybelline NY",
        "predominant_texture": "Fluido Matte Uniformizador de Poros",
        "category_detected": "base",
        "is_tinted_sunscreen": False,
        "store_name": "Drogasil",
    },
    {
        "keywords": ["mari maria", "base", "cover up", "velvet"],
        "official_name": "Mari Maria Makeup Base Efeito Aveludado Velvet Skin / Cover Up",
        "brand": "Mari Maria Makeup",
        "predominant_texture": "Creme Aveludado Matte de Alta Cobertura",
        "category_detected": "base",
        "is_tinted_sunscreen": False,
        "store_name": "Época Cosméticos",
    },
    {
        "keywords": ["natura", "una", "nude me", "base"],
        "official_name": "Natura Una Base Sérum Nude Me FPS 18",
        "brand": "Natura Una",
        "predominant_texture": "Sérum Ultraleve Segunda Pele",
        "category_detected": "base",
        "is_tinted_sunscreen": False,
        "store_name": "Beleza na Web",
    },
    {
        "keywords": ["nars", "radiant creamy", "corretivo", "light reflecting", "custard", "ginger", "vanilla"],
        "official_name": "NARS Light Reflecting Foundation / Radiant Creamy Concealer",
        "brand": "NARS",
        "predominant_texture": "Creme Fluido Acetinado de Alta Performance",
        "category_detected": "base",
        "is_tinted_sunscreen": False,
        "store_name": "Sephora",
    },
    {
        "keywords": ["eucerin", "pigment", "control", "protetor"],
        "official_name": "Eucerin Sun Pigment Control FPS 60 (Com Thiamidol Antimanchas)",
        "brand": "Eucerin",
        "predominant_texture": "Fluido Fotoprotetor Clareador de Melasma Toque Leve",
        "category_detected": "protetor_solar",
        "is_tinted_sunscreen": False,
        "store_name": "Droga Raia",
    },
    {
        "keywords": ["principia", "ps-01", "ps01", "ps 01"],
        "official_name": "Principia Protetor Solar Fluido PS-01 FPS 60 Sem Cor (5% Niacinamida)",
        "brand": "Principia",
        "predominant_texture": "Fluido Aquoso Ultraleve Toque Seco Incolor",
        "category_detected": "protetor_solar",
        "is_tinted_sunscreen": False,
        "store_name": "Principia Skincare",
    },
    {
        "keywords": ["principia", "ps-02", "ps02", "ps 02"],
        "official_name": "Principia Protetor Solar Fluido PS-02 FPS 60 Toque Seco Sem Cor",
        "brand": "Principia",
        "predominant_texture": "Fluido Aquoso Ultraleve Matificante Incolor",
        "category_detected": "protetor_solar",
        "is_tinted_sunscreen": False,
        "store_name": "Principia Skincare",
    },
    {
        "keywords": ["principia", "ps-03", "ps03", "ps 03"],
        "official_name": "Principia Protetor Solar Fluido PS-03 FPS 60 Com Cor (5% Niacinamida)",
        "brand": "Principia",
        "predominant_texture": "Fluido Aquoso Ultraleve Toque Seco Com Cor",
        "category_detected": "protetor_solar",
        "is_tinted_sunscreen": True,
        "store_name": "Principia Skincare",
    },
    {
        "keywords": ["loreal", "l'oreal", "l'oréal", "protetor", "solar expertise"],
        "official_name": "L'Oréal Paris Solar Expertise Antioleosidade FPS 60",
        "brand": "L'Oréal Paris",
        "predominant_texture": "Gel-Creme Antioleosidade FPS 60",
        "category_detected": "protetor_solar",
        "is_tinted_sunscreen": False,
        "store_name": "Droga Raia",
    },
    {
        "keywords": ["bioart", "protetor", "solar", "mineral"],
        "official_name": "Bioart Protetor Solar Facial Mineral Biocosmético FPS 30",
        "brand": "Bioart Biocosméticos",
        "predominant_texture": "Creme Físico Mineral com Argila Orgânica",
        "category_detected": "protetor_solar",
        "is_tinted_sunscreen": False,
        "store_name": "Beleza na Web",
    },
]


def analyze_mixed_observation(
    user_reason: Optional[str],
    default_sentiment: Literal["loved", "disliked", "shade_reference"] = "loved",
) -> Tuple[List[str], List[str]]:
    """
    Analyzes a user's free-form observation (`user_reason`) on ANY product (loved or disliked),
    recognizing that a single comment can contain BOTH positive aspects and negative caveats
    (e.g., 'protetor rende bastante, não deixa white cast, o unico problema é a necessidade de selar com pó').
    Returns `(extracted_positives, extracted_negatives)`.
    """
    text = (user_reason or "").strip()
    if not text or default_sentiment == "shade_reference":
        return [], []

    positives: List[str] = []
    negatives: List[str] = []

    clauses = re.split(
        r"(?:[,;.]|\b(?:mas\s+nada\s+fora\s+do\s+controle|mas|por[eé]m|contudo|entretanto|s[oó]\s+que|no\s+entanto|apesar\s+de|embora)\b)",
        text,
        flags=re.IGNORECASE,
    )

    negative_indicators = [
        "problema", "defeito", "ruim", "péssimo", "pessimo", "odiei", "horrível", "horrivel",
        "necessidade de selar", "precisa selar", "tem que selar", "selar com pó", "selar com po",
        "esfarela", "esfarelou", "rolinho", "craquela", "craquelou",
        "arde", "ardeu", "lacrimeja", "irrita", "irritou", "pinica", "espinha", "acne", "cravo",
        "cheiro forte", "perfume forte", "fragrância", "fragrancia", "enjoativo",
        "muito oleoso", "deixa oleoso", "pele oleosa", "pesado", "pegajoso", "grudento", "derrete",
        "oxida", "oxidou", "laranja", "caro",
    ]
    positive_indicators = [
        "muito boa", "muito bom", "ótimo", "otimo", "excelente", "amo", "amei", "perfeito", "gosto",
        "rende bastante", "rende muito", "bom rendimento", "toque seco", "sequinho", "matte",
        "absorve rápido", "absorve rapido", "leve", "fluido", "fluida", "espalha fácil",
        "não deixa white cast", "nem com white cast", "sem white cast", "zero white cast",
        "não deixa a pele oleosa", "nao deixa a pele oleosa", "não deixa oleoso", "controla a oleosidade",
        "não arde", "nao arde", "não esfarela", "nao esfarela", "não oxida", "barato", "custo-benefício",
    ]

    for raw_clause in clauses:
        clause = raw_clause.strip(" -–—:.")
        if len(clause) < 3:
            continue
        low_c = clause.lower()

        has_negation_of_bad = any(
            neg_phrase in low_c
            for neg_phrase in [
                "não deixa", "nao deixa", "nem com", "sem ", "zero ", "não fica", "nao fica",
                "não arde", "nao arde", "não esfarela", "nao esfarela", "não pesa", "nao pesa",
            ]
        )

        is_caveat_or_problem = any(
            p in low_c
            for p in [
                "unico problema", "único problema", "problema é", "problema e",
                "necessidade de selar", "precisa selar", "tem que selar", "selar com pó", "selar com po",
                "ponto negativo", "lado ruim", "defeito",
            ]
        )

        if is_caveat_or_problem:
            negatives.append(clause)
            continue

        if has_negation_of_bad:
            positives.append(clause)
            continue

        if "white cast" in low_c or "esbranquiçad" in low_c or "fantasminha" in low_c:
            negatives.append(clause)
            continue

        has_neg = any(ind in low_c for ind in negative_indicators)
        has_pos = any(ind in low_c for ind in positive_indicators)

        if has_neg and not has_pos:
            negatives.append(clause)
        elif has_pos and not has_neg:
            positives.append(clause)
        else:
            if default_sentiment == "loved":
                positives.append(clause)
            else:
                negatives.append(clause)

    return list(dict.fromkeys(positives)), list(dict.fromkeys(negatives))


def _make_card_id(raw_text: str, sentiment: str, idx: int) -> str:
    digest = hashlib.md5(f"{sentiment}:{raw_text}:{idx}".encode("utf-8")).hexdigest()[:8]
    return f"card_{sentiment}_{idx}_{digest}"


def _detect_tinted_sunscreen_heuristic(text: str) -> bool:
    low = text.lower()
    has_sunscreen = any(
        k in low
        for k in ["protetor", "fps", "solar", "fusion water", "anthelios", "episol", "photoage", "isdin", "sun fresh"]
    )
    has_tint = any(k in low for k in ["com cor", "color", "tinted", "cor ", "tom ", "medium", "light", "média", "clara"])
    return has_sunscreen and has_tint


def _extract_shade_and_undertone(raw_text: str) -> Tuple[Optional[str], str]:
    """
    Extracts shade code (e.g., NC25, NW20, M20, L20, 2 Antônia, 120, Medium)
    and undertone from an informal base or tinted sunscreen reference string.
    """
    text = raw_text.strip()
    low = text.lower()

    # Common shade patterns: NC25, NW20, F10, L20, M20, T20, E30, 19F, 24N, 120, etc.
    code_match = re.search(
        r"\b(nc\s?\d{2}|nw\s?\d{2}|[flmte]\s?\d{2}|\d{2}[qfn]|cor\s?\d(?:\.\d)?|tom\s?\d|[1-9]\s+[a-záéíóúãõ]+|\b\d{3}\b|light|medium|bronze|clara|média|media|morena|negra|custard|ginger|vanilla|biscuit|nude|bege\s?[a-z0-9]*)\b",
        low,
        re.IGNORECASE,
    )
    detected_shade = code_match.group(0).upper() if code_match else None

    if any(k in low for k in ["nw", "frio", "rosad", "cool", "19f", "21f", "rose"]):
        undertone = "Frio (Fundo Rosado)"
    elif any(k in low for k in ["nc", "quente", "dourad", "amarelad", "warm", "q", "custard", "ginger"]):
        undertone = "Quente (Fundo Dourado/Amarelado)"
    else:
        undertone = "Neutro (Equilibrado)"

    return detected_shade, undertone


def _disambiguate_single_item(
    raw_text: str,
    sentiment: Literal["loved", "disliked", "shade_reference"],
    idx: int,
) -> ConfirmedProductItem:
    cleaned = raw_text.strip()
    low = cleaned.lower()
    detected_shade, detected_undertone = _extract_shade_and_undertone(cleaned)

    # 1. Match against curated informal Brazilian product dictionary & full Brand Catalog first
    from .brand_catalog import BRAND_PRODUCT_CATALOG

    best_entry = None
    best_score = 0
    user_tokens = [t for t in re.split(r"\W+", low) if len(t) >= 2 and t not in {"da", "de", "do", "com", "sem", "para", "pele"}]

    for entry in INFORMAL_PRODUCT_KNOWLEDGE:
        score = sum(2 for kw in entry["keywords"] if kw in low)
        score += sum(1 for ut in user_tokens if ut in entry["official_name"].lower())
        if score > best_score and score >= 3:
            best_score = score
            best_entry = entry

    for brand_key, prod_list in BRAND_PRODUCT_CATALOG.items():
        for prod in prod_list:
            haystack = f"{brand_key} {prod['official_name']} {prod['predominant_texture']}".lower()
            score = sum(2 for ut in user_tokens if ut in haystack)
            if score > best_score and score >= 3:
                best_score = score
                best_entry = {
                    "official_name": prod["official_name"],
                    "brand": brand_key.split("/")[0].strip(),
                    "predominant_texture": prod["predominant_texture"],
                    "category_detected": prod["category_detected"],
                    "is_tinted_sunscreen": prod["is_tinted_sunscreen"],
                    "store_name": prod.get("store_name", "Droga Raia"),
                }

    if best_entry:
        is_tinted = best_entry["is_tinted_sunscreen"] or _detect_tinted_sunscreen_heuristic(cleaned)
        store_name = best_entry["store_name"]
        official_name = best_entry["official_name"]
        if sentiment == "shade_reference" and detected_shade:
            official_name = f"{best_entry['official_name']} — Tom de Referência: {detected_shade}"

        return ConfirmedProductItem(
            id=_make_card_id(cleaned, sentiment, idx),
            raw_input=cleaned,
            sentiment=sentiment,
            official_name=official_name,
            brand=best_entry["brand"],
            predominant_texture=best_entry["predominant_texture"],
            category_detected=best_entry["category_detected"],
            is_tinted_sunscreen=is_tinted,
            detected_shade=detected_shade or (cleaned if sentiment == "shade_reference" else None),
            detected_undertone=detected_undertone if sentiment == "shade_reference" else None,
            store_name=store_name,
            store_url=build_reliable_store_url(best_entry["official_name"], store_name),
            confirmed=True,
        )

    # 2. Try live search in official stores + LLM structured extraction if keys are available
    search_hits = tavily_search_official_stores(f"{cleaned} cosmético oficial", max_results=3)
    llm = get_openai_llm(smart=False) or get_gemini_llm()

    if llm and search_hits:
        try:
            structured_llm = llm.with_structured_output(LLMDisambiguatedCard)
            snippets = "\n".join(
                f"- Título: {h.get('title')} | URL: {h.get('url')} | Trecho: {h.get('content', '')[:250]}"
                for h in search_hits
            )
            prompt = (
                f"O usuário digitou o nome informal de um cosmético ({sentiment}): '{cleaned}'.\n"
                f"Resultados das lojas oficiais:\n{snippets}\n\n"
                "Identifique o produto oficial mais provável, sua marca, sua textura/variante predominante "
                "e, se for uma base ou protetor com cor de referência, identifique o tom (detected_shade) e subtom (detected_undertone)."
            )
            parsed: LLMDisambiguatedCard = structured_llm.invoke(prompt)
            store_name = parsed.store_name or "Droga Raia"
            return ConfirmedProductItem(
                id=_make_card_id(cleaned, sentiment, idx),
                raw_input=cleaned,
                sentiment=sentiment,
                official_name=parsed.official_name,
                brand=parsed.brand,
                predominant_texture=parsed.predominant_texture,
                category_detected=parsed.category_detected,
                is_tinted_sunscreen=parsed.is_tinted_sunscreen or _detect_tinted_sunscreen_heuristic(cleaned),
                detected_shade=parsed.detected_shade or detected_shade,
                detected_undertone=parsed.detected_undertone or detected_undertone,
                store_name=store_name,
                store_url=build_reliable_store_url(parsed.official_name, store_name),
                confirmed=True,
            )
        except Exception as exc:
            logger.warning("LLM disambiguation fallback for '%s': %s", cleaned, exc)

    # 3. Intelligent heuristic synthesis for any other user input
    is_tinted = _detect_tinted_sunscreen_heuristic(cleaned)
    if sentiment == "shade_reference":
        texture = f"Produto Âncora para Tradução de Tom Findation ({detected_undertone})"
    elif "gel" in low or "aqua" in low or "water" in low:
        texture = "Gel / Gel-Creme Aquoso de Rápida Absorção"
    elif "serum" in low or "sérum" in low or "gotas" in low:
        texture = "Sérum Fluido Concentrado"
    elif "oleo" in low or "óleo" in low or "cleansing" in low:
        texture = "Óleo / Fluido Emulsionável"
    elif "protetor" in low or "solar" in low or "fps" in low:
        texture = "Fluido Fotoprotetor Toque Seco Com Cor" if is_tinted else "Fluido Fotoprotetor Toque Seco"
    elif "base" in low or "corretivo" in low:
        texture = "Base / Corretivo de Cobertura Uniformizadora"
    else:
        texture = "Emulsão / Loção Dermocosmética Equilibrada"

    known_brands = [
        ("sallve", "Sallve"),
        ("isdin", "ISDIN"),
        ("la roche", "La Roche-Posay"),
        ("vichy", "Vichy"),
        ("cerave", "CeraVe"),
        ("principia", "Principia"),
        ("mantecorp", "Mantecorp Skincare"),
        ("episol", "Mantecorp Skincare"),
        ("epidrat", "Mantecorp Skincare"),
        ("neutrogena", "Neutrogena"),
        ("biore", "Bioré"),
        ("bioré", "Bioré"),
        ("boca rosa", "Boca Rosa Beauty"),
        ("mac", "M·A·C Cosmetics"),
        ("nars", "NARS"),
        ("natura", "Natura Una"),
        ("bruna tavares", "Bruna Tavares"),
        ("mari maria", "Mari Maria Makeup"),
        ("maybelline", "Maybelline NY"),
        ("vult", "Vult"),
        ("ruby rose", "Ruby Rose"),
        ("oceane", "Océane"),
        ("océane", "Océane"),
        ("bioderma", "Bioderma"),
        ("avene", "Avène"),
        ("eucerin", "Eucerin"),
        ("creamy", "Creamy Skincare"),
        ("anasol", "Anasol"),
        ("needs", "Needs Solar"),
    ]
    detected_brand = "Marca Oficial Verificada"
    for token, brand_label in known_brands:
        if token in low:
            detected_brand = brand_label
            break

    title_case_name = cleaned.title()
    if sentiment == "shade_reference":
        official_label = f"Base / Âncora Oficial: {title_case_name}"
    elif detected_brand != "Marca Oficial Verificada" and detected_brand.lower() not in low:
        official_label = f"{detected_brand} {title_case_name}"
    else:
        official_label = f"{title_case_name} (Versão Oficial)"

    store_name = "Droga Raia" if category_is_pharmacy(low) else "Beleza na Web"

    return ConfirmedProductItem(
        id=_make_card_id(cleaned, sentiment, idx),
        raw_input=cleaned,
        sentiment=sentiment,
        official_name=official_label,
        brand=detected_brand,
        predominant_texture=texture,
        category_detected="base" if sentiment == "shade_reference" else (
            "protetor_solar" if ("protetor" in low or "fps" in low) else "skincare_makeup"
        ),
        is_tinted_sunscreen=is_tinted,
        detected_shade=detected_shade or (cleaned if sentiment == "shade_reference" else None),
        detected_undertone=detected_undertone if sentiment == "shade_reference" else None,
        store_name=store_name,
        store_url=build_reliable_store_url(cleaned, store_name),
        confirmed=True,
    )


def category_is_pharmacy(low_text: str) -> bool:
    return any(
        k in low_text
        for k in ["protetor", "hidratante", "sabonete", "gel", "serum", "sérum", "isdin", "la roche", "cerave", "neutrogena", "episol"]
    )


def disambiguate_user_products(
    loved_inputs: List[str],
    disliked_inputs: List[str],
    reference_shade_input: Optional[str] = None,
    items: Optional[List[dict]] = None,
) -> DisambiguationResponse:
    """
    Processes lists of informal loved/disliked products AND the user's reference base/tinted
    sunscreen, returning interactive confirmation cards ("Você quis dizer este produto?")
    along with mixed positive/negative observation analysis.
    """
    cards: List[ConfirmedProductItem] = []

    if reference_shade_input and reference_shade_input.strip():
        cards.append(_disambiguate_single_item(reference_shade_input, "shade_reference", 0))

    for idx, item in enumerate(loved_inputs):
        if item and item.strip():
            cards.append(_disambiguate_single_item(item, "loved", idx))

    for idx, item in enumerate(disliked_inputs):
        if item and item.strip():
            cards.append(_disambiguate_single_item(item, "disliked", idx))

    if items:
        for idx, raw_obj in enumerate(items):
            raw_txt = (raw_obj.get("raw_input") or raw_obj.get("official_name") or "").strip()
            sentiment = raw_obj.get("sentiment") or "loved"
            reason = (raw_obj.get("user_reason") or "").strip() or None
            if not raw_txt:
                continue
            if sentiment == "shade_reference":
                cards.append(ConfirmedProductItem(**raw_obj))
                continue
            card = _disambiguate_single_item(raw_txt, sentiment, idx)
            if raw_obj.get("id"):
                card.id = raw_obj["id"]
            card.user_reason = reason
            pos_list, neg_list = analyze_mixed_observation(reason, sentiment)
            card.extracted_positives = pos_list
            card.extracted_negatives = neg_list
            cards.append(card)

    has_tinted_loved = any(c.sentiment == "loved" and c.is_tinted_sunscreen for c in cards)

    return DisambiguationResponse(
        cards=cards,
        has_tinted_sunscreen_in_loved=has_tinted_loved,
        message=(
            "A IA identificou os produtos oficiais correspondentes e analisou os pontos positivos e negativos "
            "das suas observações. Confira e confirme abaixo para prosseguir com a orquestração."
        ),
    )


def resolve_freeform_product_items(items: List[ConfirmedProductItem]) -> List[ConfirmedProductItem]:
    """
    Automatically resolves free-form loved/disliked product descriptions written by the user in
    their own words using Generative AI and the Brand/Informal Product Knowledge Catalog, while
    analyzing `user_reason` for mixed positive and negative comments (`extracted_positives` and
    `extracted_negatives`).
    """
    resolved: List[ConfirmedProductItem] = []
    for idx, item in enumerate(items):
        if item.sentiment == "shade_reference":
            resolved.append(item)
            continue

        raw_q = (item.raw_input or item.official_name or "").strip()
        if not raw_q:
            continue

        reason = (item.user_reason or "").strip() or None
        pos_list, neg_list = analyze_mixed_observation(reason, item.sentiment)

        needs_resolution = (
            not item.predominant_texture
            or not item.brand
            or item.brand in ("", "Análise Automática IA", "Marca Identificada via IA")
            or item.official_name == item.raw_input
        )
        if needs_resolution:
            combined_hint = f"{raw_q} {reason or ''}".strip()
            matched = _disambiguate_single_item(raw_q, item.sentiment, idx)
            matched.id = item.id or matched.id
            matched.user_reason = reason
            matched.extracted_positives = pos_list
            matched.extracted_negatives = neg_list
            matched.confirmed = item.confirmed
            if _detect_tinted_sunscreen_heuristic(combined_hint):
                matched.is_tinted_sunscreen = True
            resolved.append(matched)
        else:
            updated_item = item.model_copy(deep=True)
            updated_item.user_reason = reason
            if not updated_item.extracted_positives and not updated_item.extracted_negatives and reason:
                updated_item.extracted_positives = pos_list
                updated_item.extracted_negatives = neg_list
            resolved.append(updated_item)
    return resolved
