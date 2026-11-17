"""
Django Ninja Typed API (`recommender/api.py`) using Pydantic Discriminated Unions.
Exposes:
- POST /api/disambiguate -> Human-in-the-Loop product validation cards
- POST /api/recommend -> 4-stage parallel LangGraph multi-agent orchestration
- GET /api/health -> System & multi-model configuration telemetry
"""
from django.conf import settings
from django.http import HttpRequest
from ninja import Body, NinjaAPI
from ninja.errors import HttpError

from .agents import disambiguate_user_products, run_orchestrator_pipeline
from .agents.brand_catalog import get_catalog_for_frontend, search_products_by_brand
from .models import ConsultationLog, ProductReviewCache
from .schemas import (
    BrandSearchRequest,
    BrandSearchResponse,
    DisambiguationRequest,
    DisambiguationResponse,
    RecommendationRequestUnion,
    RecommendationResponse,
)

api = NinjaAPI(
    title="Beauty AI Orchestrator API",
    version="1.0.0",
    description=(
        "Sistema de recomendação hiperpersonalizada de Skincare e Beleza & Maquiagem "
        "com execução paralela (Fan-out / Fan-in), filtro de orçamento via Jina Reader, "
        "tradução de tons via Findation e Guardrail Dermatológico."
    ),
)


@api.get("/health", tags=["System"])
def health_check(request: HttpRequest):
    """Returns system status, configured LLM models, and official store domains."""
    return {
        "status": "ok",
        "architecture": "LangGraph Fan-out/Fan-in 4-Stage Computational Funnel",
        "models": {
            "stage_2_extractor": settings.GEMINI_MODEL,
            "agent_1b_formulation": settings.OPENAI_FAST_MODEL,
            "stage_3_findation": settings.OPENAI_SMART_MODEL,
            "stage_4_guardrail": settings.ANTHROPIC_MODEL,
        },
        "official_domains": settings.OFFICIAL_STORE_DOMAINS,
        "cached_products_count": ProductReviewCache.objects.count(),
        "consultations_count": ConsultationLog.objects.count(),
    }


@api.get(
    "/brand-catalog",
    tags=["Human-in-the-Loop"],
    summary="Etapa B: Catálogo Estruturado Marca -> Produtos -> Tons para confirmação 1 a 1",
)
def brand_catalog_endpoint(request: HttpRequest):
    """
    Returns brands and their official products (with textures, categories, and shade charts)
    so the user can select a brand first and immediately choose/confirm the exact product one by one.
    """
    return get_catalog_for_frontend()


@api.post(
    "/search-brand-products",
    response=BrandSearchResponse,
    tags=["Human-in-the-Loop"],
    summary="Etapa B: Usuário digita a marca -> Confirma a marca e retorna os produtos correspondentes",
)
def search_brand_products_endpoint(
    request: HttpRequest,
    payload: BrandSearchRequest,
) -> BrandSearchResponse:
    """
    Receives the brand typed by the user (e.g., 'sallve', 'la roche', 'isdin', 'mac'),
    confirms the official brand name, and returns all corresponding products of that brand.
    """
    return search_products_by_brand(
        brand_query=payload.brand_query,
        product_hint=payload.product_hint,
        only_shade_products=payload.only_shade_products,
    )


@api.post(
    "/disambiguate",
    response=DisambiguationResponse,
    tags=["Human-in-the-Loop"],
    summary="Etapa B: Desambiguação e Confirmação de Produtos",
)
def disambiguate_products_endpoint(
    request: HttpRequest,
    payload: DisambiguationRequest,
) -> DisambiguationResponse:
    """
    Receives informal loved & disliked product names (e.g., 'hidratante roxo da sallve',
    'protetor com cor da isdin') and returns interactive confirmation cards with official
    store name, brand, predominant texture, and tinted sunscreen detection.
    """
    return disambiguate_user_products(
        loved_inputs=payload.loved_inputs,
        disliked_inputs=payload.disliked_inputs,
        reference_shade_input=payload.reference_shade_input,
        items=payload.items,
    )


@api.post(
    "/recommend",
    response=RecommendationResponse,
    tags=["Multi-Agent Orchestrator"],
    summary="Etapa C & D: Orquestração Paralela em Funil (4 Estágios) + Webhook n8n",
)
def recommend_products_endpoint(
    request: HttpRequest,
    payload: RecommendationRequestUnion = Body(...),
) -> RecommendationResponse:
    """
    Executes the 4-stage LangGraph pipeline using a Pydantic Discriminated Union (`flow_type`):
    - `skincare_rotina`: Rotina Completa (Limpeza, Hidratação e Proteção Solar)
    - `skincare_especifico`: Produto Específico de Skincare (com campos condicionais para Tratamento e Protetor Solar)
    - `maquiagem`: Beleza & Maquiagem (com âncora Findation e expectativa cromática para blush/sombra/contorno/iluminador)
    """
    try:
        return run_orchestrator_pipeline(payload)
    except ValueError as exc:
        raise HttpError(422, str(exc)) from exc
