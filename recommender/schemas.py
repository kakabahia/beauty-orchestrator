"""
Pydantic Schemas with Discriminated Unions for the Beauty AI Orchestrator.
Covers:
1. Adaptive Dual-Flow User Inputs (Skincare Rotina Completa, Skincare Produto Específico, Beleza & Maquiagem)
2. Human-in-the-Loop Product Disambiguation Cards
3. Multi-Agent Structured Outputs (Agent 1A Social, Agent 1B Formulation, Stage 2 Budget, Stage 3 Findation, Stage 4 Derm Guardrail)
"""
from typing import Annotated, List, Literal, Optional, Union
from pydantic import BaseModel, Field, model_validator


# ==============================================================================
# Domain Literals & Enums
# ==============================================================================
SkinType = Literal["oleosa", "seca", "mista", "normal"]

SkincareProblem = Literal[
    "acne",
    "manchas_melasma",
    "rosacea",
    "poros_dilatados",
    "comedoes",
    "cravos",
]

SkincareSpecificCategory = Literal[
    "hidratante",
    "protetor_solar",
    "tonico",
    "sabonete_facial",
    "cleansing_oil",
    "limpeza_micelar",
    "serum",
    "tratamento",
]

MakeupCategory = Literal[
    "base",
    "corretivo",
    "blush",
    "sombra",
    "contorno",
    "iluminador",
    "po_facial",
    "batom",
]

CHROMATIC_FINISH_CATEGORIES = {"blush", "sombra", "contorno", "iluminador"}
SHADE_MATCH_MAKEUP_CATEGORIES = {"base", "corretivo"}


# ==============================================================================
# Step B: Human-in-the-Loop Product Disambiguation Schemas
# ==============================================================================
class RawProductMention(BaseModel):
    """Informal product query typed by the user (e.g., 'hidratante roxo da sallve')."""

    raw_text: str = Field(..., min_length=2, description="Nome informal digitado pelo usuário")
    sentiment: Literal["loved", "disliked"] = Field(
        ..., description="Se o usuário ama ('loved') ou não gosta ('disliked') do produto"
    )


class DisambiguationRequest(BaseModel):
    """Payload to disambiguate loved, disliked, and reference shade/base products before running the main graph."""

    loved_inputs: List[str] = Field(
        default_factory=list,
        description="Lista de nomes informais de produtos que o usuário ama",
    )
    disliked_inputs: List[str] = Field(
        default_factory=list,
        description="Lista de nomes informais de produtos que o usuário não gosta",
    )
    reference_shade_input: Optional[str] = Field(
        default=None,
        description="Base ou protetor solar com cor de referência digitado pelo usuário para confirmação de produto e tom",
    )
    items: List[dict] = Field(
        default_factory=list,
        description="Lista opcional de itens com raw_input, sentiment e user_reason para confirmação e análise mista de observações",
    )


class BrandSearchRequest(BaseModel):
    """Request payload when the user types a brand name to confirm and list its products."""

    brand_query: str = Field(..., min_length=1, description="Nome da marca digitada pelo usuário (ex: 'sallve', 'la roche', 'mac')")
    product_hint: Optional[str] = Field(
        default=None,
        description="Filtro opcional de produto ou categoria dentro da marca (ex: 'hidratante', 'protetor', 'base')",
    )
    only_shade_products: bool = Field(
        default=False,
        description="Se True, prioriza bases, corretivos e protetores solares com cor da marca",
    )


class BrandProductOption(BaseModel):
    """A single official product belonging to the confirmed brand."""

    official_name: str
    brand: str
    predominant_texture: str
    category_detected: str = "skincare"
    is_tinted_sunscreen: bool = False
    shades: List[str] = Field(default_factory=list)
    store_name: str = "Droga Raia"
    store_url: str = "https://www.drogaraia.com.br"


class BrandSearchResponse(BaseModel):
    """Response containing the confirmed brand and all corresponding products of that brand."""

    found: bool = True
    confirmed_brand: Optional[str] = None
    brand_aliases_matched: List[str] = Field(default_factory=list)
    products: List[BrandProductOption] = Field(default_factory=list)
    message: str


class ConfirmedProductItem(BaseModel):
    """Product item (free-form or disambiguated) with optional user reason for loving/disliking it."""

    id: str = Field(default="freeform_item", description="Identificador único do item")
    raw_input: str = Field(..., description="Texto original digitado pelo usuário")
    sentiment: Literal["loved", "disliked", "shade_reference"] = Field(
        ...,
        description="'loved' para favorito, 'disliked' para rejeitado, 'shade_reference' para base/protetor âncora de tom",
    )
    user_reason: Optional[str] = Field(
        default=None,
        description="Motivo/observação opcional informada pelo usuário sobre o produto (pode conter pontos positivos e negativos)",
    )
    extracted_positives: List[str] = Field(
        default_factory=list,
        description="Pontos positivos extraídos da observação do usuário sobre este item",
    )
    extracted_negatives: List[str] = Field(
        default_factory=list,
        description="Pontos negativos ou ressalvas extraídos da observação do usuário sobre este item",
    )
    official_name: str = Field(default="", description="Nome oficial completo do produto identificado pela IA")
    brand: str = Field(default="", description="Marca oficial do produto")
    predominant_texture: str = Field(
        default="",
        description="Textura ou variante predominante (ex: Gel-Creme Aquoso, Fluido Ultraleve Com Cor, Base Matte)",
    )
    category_detected: str = Field(
        default="skincare",
        description="Categoria detectada do produto (ex: hidratante, protetor_solar, base, serum)",
    )
    is_tinted_sunscreen: bool = Field(
        default=False,
        description="True se o produto confirmado for um protetor solar com cor/pigmento",
    )
    detected_shade: Optional[str] = Field(
        default=None,
        description="Tom detectado ou confirmado pelo usuário (ex: 'NC25', 'M20', '2 Antônia', 'Médio')",
    )
    detected_undertone: Optional[str] = Field(
        default=None,
        description="Subtom identificado da base/protetor de referência (ex: 'Quente Dourado', 'Neutro', 'Frio Rosado')",
    )
    store_name: str = Field(default="Busca Google / Loja Oficial", description="Loja/Farmácia oficial de referência")
    store_url: str = Field(default="https://www.google.com.br", description="URL oficial de referência")
    confirmed: bool = Field(default=True, description="Indica se o item está ativo para o grafo")


class DisambiguationResponse(BaseModel):
    """Response containing interactive product cards ('Você quis dizer este produto?')."""

    cards: List[ConfirmedProductItem]
    has_tinted_sunscreen_in_loved: bool = Field(
        default=False,
        description="Flag automática caso um protetor solar com cor seja detectado entre os favoritos",
    )
    message: str = Field(
        default="Confirme abaixo os produtos identificados nas lojas oficiais antes de iniciar a orquestração."
    )


# ==============================================================================
# Step A: Discriminated Union for Adaptive Dual-Flow Journey
# ==============================================================================
class BaseJourneyProfile(BaseModel):
    """Shared profile attributes across Skincare and Makeup journeys."""

    skin_type: SkinType = Field(..., description="Tipo de pele: oleosa, seca, mista ou normal")
    is_sensitive: bool = Field(default=False, description="Possui pele sensível ou reativa?")
    budget: float = Field(..., gt=0, description="Orçamento máximo total em R$ (Reais)")
    confirmed_products: List[ConfirmedProductItem] = Field(
        default_factory=list,
        description="Apenas os produtos confirmados na etapa Human-in-the-Loop",
    )


class SkincareRoutineJourney(BaseJourneyProfile):
    """
    Jornada Skincare -> Modo Rotina Completa:
    Coleta tipo de pele, sensibilidade, problemas ativos e orçamento total para
    estruturar uma cesta equilibrada entre etapas (limpeza, hidratação e proteção solar).
    """

    flow_type: Literal["skincare_rotina"] = "skincare_rotina"
    active_problems: List[SkincareProblem] = Field(
        default_factory=list,
        description="Problemas ativos: acne, manchas_melasma, rosacea, poros_dilatados, comedoes, cravos",
    )
    prefers_tinted_sunscreen: bool = Field(
        default=False,
        description="Se prefere que a etapa de proteção solar da rotina seja com cor",
    )
    reference_shade: Optional[str] = Field(
        default=None,
        description="Tom de referência de base ou protetor solar com cor que já dá certo",
    )


class SkincareSpecificJourney(BaseJourneyProfile):
    """
    Jornada Skincare -> Modo Produto Específico:
    Permite selecionar a categoria exata.
    - Ao selecionar Tratamento: campo condicional solicita o problema específico.
    - Ao selecionar Protetor Solar (ou citar protetor com cor nos favoritos): habilita
      preferência por versão com cor e solicita tom de referência.
    """

    flow_type: Literal["skincare_especifico"] = "skincare_especifico"
    category: SkincareSpecificCategory = Field(
        ...,
        description="Categoria exata de skincare desejada",
    )
    target_problem: Optional[str] = Field(
        default=None,
        description="Problema específico a ser combatido (obrigatório quando category='tratamento')",
    )
    prefers_tinted_sunscreen: bool = Field(
        default=False,
        description="Preferência por protetor solar com cor",
    )
    reference_shade: Optional[str] = Field(
        default=None,
        description="Tom de referência (de outro protetor com cor ou de uma base que dê certo)",
    )
    active_problems: List[SkincareProblem] = Field(
        default_factory=list,
        description="Problemas ativos adicionais opcionais",
    )

    @model_validator(mode="after")
    def validate_conditional_skincare_fields(self) -> "SkincareSpecificJourney":
        if self.category == "tratamento" and not (self.target_problem and self.target_problem.strip()):
            raise ValueError(
                "Ao selecionar a categoria 'tratamento', é obrigatório informar o problema específico a ser combatido (target_problem)."
            )
        return self


class MakeupJourney(BaseJourneyProfile):
    """
    Jornada Beleza & Maquiagem:
    - Coleta tipo de pele, sensibilidade e orçamento máximo.
    - Solicita cores/tons de produtos anteriores que deram certo (âncora Findation).
    - Para categorias cromáticas de acabamento (blush, sombra, contorno e iluminador),
      habilita caixa de texto livre para descrever cor desejada e efeito esperado.
    """

    flow_type: Literal["maquiagem"] = "maquiagem"
    category: MakeupCategory = Field(
        ...,
        description="Categoria de maquiagem desejada (base, corretivo, blush, sombra, contorno, iluminador, etc.)",
    )
    reference_shade: Optional[str] = Field(
        default=None,
        description="Produtos e cores/tons anteriores que deram certo (ex: 'Base MAC Studio Fix NC25 / Boca Rosa 2 Antônia')",
    )
    chromatic_expectation: Optional[str] = Field(
        default=None,
        description="Descrição livre da cor desejada e efeito esperado para categorias cromáticas (ex: 'blush pêssego queimado sem partículas de brilho')",
    )

    @model_validator(mode="after")
    def validate_conditional_makeup_fields(self) -> "MakeupJourney":
        if self.category in CHROMATIC_FINISH_CATEGORIES and not (
            self.chromatic_expectation and self.chromatic_expectation.strip()
        ):
            raise ValueError(
                f"Para a categoria cromática '{self.category}', descreva a cor desejada e o efeito esperado (chromatic_expectation)."
            )
        return self


# Pydantic Discriminated Union across the 3 adaptive user flows
RecommendationRequestUnion = Annotated[
    Union[SkincareRoutineJourney, SkincareSpecificJourney, MakeupJourney],
    Field(discriminator="flow_type"),
]


# ==============================================================================
# Multi-Agent Internal & Output Schemas
# ==============================================================================
class SocialConsensusOutput(BaseModel):
    """Output of Agent 1A (Consenso Social & Experiência de Uso)."""

    success_drivers: List[str] = Field(
        default_factory=list,
        description="Motivos práticos de sucesso extraídos das avaliações (ex: absorve rápido, não esfarela, não arde os olhos)",
    )
    recurring_complaints_to_avoid: List[str] = Field(
        default_factory=list,
        description="Reclamações recorrentes que devem ser evitadas (ex: oxida ao longo do dia, deixa a pele pegajosa)",
    )
    performance_search_queries: List[str] = Field(
        default_factory=list,
        description="Queries focadas em performance real geradas pelo Agente 1A",
    )
    summary_rationale: str = Field(
        default="",
        description="Síntese da análise de consenso social dos produtos favoritos e rejeitados",
    )


class FormulationAnalysisOutput(BaseModel):
    """Output of Agent 1B (Formulação, Textura & Ativos)."""

    enforced_texture_rule: str = Field(
        ...,
        description="Regra de textura inferida (ex: 'Textura Gel / Aquosa Oil-Free obrigatória')",
    )
    detected_hidden_patterns: List[str] = Field(
        default_factory=list,
        description="Padrões ocultos identificados nos produtos confirmados",
    )
    beneficial_actives: List[str] = Field(
        default_factory=list,
        description="Ativos benéficos recomendados para os objetivos e tipo de pele do usuário",
    )
    forbidden_ingredients: List[str] = Field(
        default_factory=list,
        description="Ingredientes proibidos conforme a sensibilidade da pele e rejeições anteriores",
    )
    auto_tinted_flag: bool = Field(
        default=False,
        description="True se o usuário pediu protetor com cor OU se listou protetor solar com cor entre os favoritos",
    )
    formulation_search_queries: List[str] = Field(
        default_factory=list,
        description="Queries técnicas geradas pelo Agente 1B com foco em ativos e veículos cosméticos",
    )


class CandidateProduct(BaseModel):
    """Product candidate extracted in Stage 2 (Official Stores + Jina Reader + Gemini 2.5 Flash) and enriched in Stages 3 & 4."""

    product_name: str
    brand: str
    routine_step: Optional[str] = Field(
        default=None,
        description="Etapa da rotina (Limpeza, Hidratação, Proteção Solar) ou categoria específica",
    )
    category: str
    texture: str
    price: float = Field(..., description="Preço atual vigente extraído via Jina Reader")
    store_name: str
    store_domain: str
    store_url: str
    store_reputation_badge: str = Field(
        default="🏆 Selo RA1000 / Ótimo no Reclame Aqui — Rede de Farmácia/Loja Oficial",
        description="Avaliação de confiabilidade da loja/farmácia segundo índice Reclame Aqui / Grandes Redes",
    )
    store_reputation_score: str = Field(
        default="Nota 8.7/10 (Verificada)",
        description="Nota média de reputação da loja",
    )
    price_verified_via_screenshot: bool = Field(
        default=True,
        description="True se o link e o preço foram validados via print/inspeção da página da loja",
    )
    price_verification_note: str = Field(
        default="📸 Preço e disponibilidade em estoque verificados na página do link.",
        description="Nota de verificação do preço e estoque no link retornado",
    )
    alternative_store_links: List[dict] = Field(
        default_factory=list,
        description="Links diretos funcionais em grandes farmácias e varejistas (Droga Raia, Drogasil, Amazon, Época, Beleza na Web)",
    )
    is_tinted: bool = False
    key_actives: List[str] = Field(default_factory=list)
    composition_notes: str = ""
    pros: List[str] = Field(default_factory=list, description="Prós segundo internautas")
    cons: List[str] = Field(default_factory=list, description="Contras segundo internautas")

    # Stage 3: Findation Shade Translation fields
    recommended_shade: str = Field(
        default="Incolor (sem pigmento)",
        description="Código exato da cor via Findation, recomendação por subtom ou 'Incolor (sem pigmento)'",
    )
    shade_match_method: Literal["findation_exact", "undertone_approx", "colorless", "chromatic_match"] = Field(
        default="colorless",
        description="Método utilizado para definição do tom",
    )
    undertone_Detected: Optional[str] = Field(
        default=None,
        description="Subtom estimado (Quente, Frio, Neutro, Oliva) quando usado fallback aproximado",
    )
    shade_translation_note: Optional[str] = Field(
        default=None,
        description="Explicação da tradução de tom via Findation ou alerta de aproximação por subtom",
    )

    # Stage 4: Dermatological Guardrail & Final Curation fields
    match_score: int = Field(
        default=92,
        ge=0,
        le=100,
        description="Score de Match hiperpersonalizado (0 a 100%)",
    )
    indicated_skin_types: List[str] = Field(
        default_factory=list,
        description="Tipos de pele para os quais o produto é dermatologicamente validado",
    )
    dermatological_badge: str = Field(
        default="Selo Dermatológico Seguro",
        description="Selo de segurança ou alerta específico para pele sensível",
    )
    dermatological_Safety_level: Literal["safe", "caution", "alert"] = Field(
        default="safe",
        description="Nível de segurança dermatológica (safe, caution, alert)",
    )
    irritant_warnings: List[str] = Field(
        default_factory=list,
        description="Lista de possíveis gatilhos irritantes detectados (ou vazia se 100% seguro)",
    )
    curator_verdict: str = Field(
        default="",
        description="Justificativa técnica e sensorial da curadoria final (Claude 3.7 Sonnet)",
    )


class OrchestrationTelemetry(BaseModel):
    """Detailed telemetry of the 4-stage multi-agent funnel execution for UI transparency."""

    flow_type: str
    models_used: dict
    stage_1a_social: SocialConsensusOutput
    stage_1b_formulation: FormulationAnalysisOutput
    stage_2_total_scraped: int
    stage_2_approved_count: int
    budget_limit: float
    total_recommended_price: float
    budget_exceeded_alert: bool = False
    budget_alert_message: Optional[str] = None
    stage_3_findation_executed: bool = False
    stage_3_routing_reason: str = ""
    n8n_webhook_queued: bool = True


class RecommendationResponse(BaseModel):
    """Final response returned by the `/api/recommend` endpoint."""

    recommendations: List[CandidateProduct]
    resolved_user_products: List[ConfirmedProductItem] = Field(
        default_factory=list,
        description="Produtos favoritos e não favoritos do usuário com a correspondência oficial encontrada pela IA e análise mista de observações",
    )
    telemetry: OrchestrationTelemetry
    curated_summary: str
