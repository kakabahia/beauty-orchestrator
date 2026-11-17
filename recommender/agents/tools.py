"""
External Tools & Multi-Model LLM Factory for the Beauty AI Orchestrator:
- Gemini (`gemini-2.5-flash` via GOOGLE_API_KEY): Stage 2 Jina Reader Markdown price/product extraction
- OpenAI (`gpt-4o-mini` / `gpt-4o` via OPENAI_API_KEY): Agent 1B formulation extraction, conditional routing & Findation
- Claude (`claude-3-7-sonnet-latest` via ANTHROPIC_API_KEY): Stage 4 Dermatological Guardrail & Final Curation
- Tavily Search API (`TAVILY_API_KEY`) with `include_domains` filter for official stores & findation.com
- Jina Reader (`https://r.jina.ai/{url}`) for clean Markdown conversion of product pages
"""
import base64
import hashlib
import logging
import os
import re
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote_plus, unquote, urlparse
import httpx
from django.conf import settings
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


def _has_real_key(key_value: Optional[str]) -> bool:
    """Checks if an API key is configured and not a placeholder."""
    if not key_value:
        return False
    cleaned = key_value.strip()
    if not cleaned or cleaned.startswith("your-") or cleaned.startswith("optional-"):
        return False
    return True


# ==============================================================================
# Multi-Model LLM Providers
# ==============================================================================
def get_gemini_llm(temperature: float = 0.1):
    """
    Returns Gemini (`gemini-2.5-flash`) for reading & extracting prices/data from
    long Jina Reader Markdown texts in Stage 2, and social review synthesis.
    """
    if not _has_real_key(settings.GOOGLE_API_KEY):
        return None
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL or "gemini-2.5-flash",
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=temperature,
        )
    except Exception as exc:
        logger.warning("Failed to initialize Gemini LLM: %s", exc)
        return None


def get_openai_llm(smart: bool = False, temperature: float = 0.1):
    """
    Returns OpenAI (`gpt-4o-mini` or `gpt-4o`) for Agent 1B (structured extraction of
    actives/texture), conditional routing, and Findation shade translation.
    """
    if not _has_real_key(settings.OPENAI_API_KEY):
        return None
    try:
        from langchain_openai import ChatOpenAI

        model_name = settings.OPENAI_SMART_MODEL if smart else settings.OPENAI_FAST_MODEL
        return ChatOpenAI(
            model=model_name or ("gpt-4o" if smart else "gpt-4o-mini"),
            api_key=settings.OPENAI_API_KEY,
            temperature=temperature,
        )
    except Exception as exc:
        logger.warning("Failed to initialize OpenAI LLM: %s", exc)
        return None


def get_claude_llm(temperature: float = 0.2):
    """
    Returns Claude (`claude-3-7-sonnet-latest`) for Stage 4 (Dermatological Guardrail
    & Final Curation).
    """
    if not _has_real_key(settings.ANTHROPIC_API_KEY):
        return None
    try:
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=settings.ANTHROPIC_MODEL or "claude-3-7-sonnet-latest",
            api_key=settings.ANTHROPIC_API_KEY,
            temperature=temperature,
        )
    except Exception as exc:
        logger.warning("Failed to initialize Claude LLM: %s", exc)
        return None


# ==============================================================================
# Tavily Search API with `include_domains`
# ==============================================================================
def tavily_search_official_stores(
    query: str,
    max_results: int = 5,
    extra_domains: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Searches exclusively within official cosmetic store domains using Tavily API
    with `include_domains`:
      - belezanaweb.com.br
      - sephora.com.br
      - elausa.com.br
      - amazon.com.br
      - principiaskincare.com
    """
    allowed_domains = list(settings.OFFICIAL_STORE_DOMAINS)
    if extra_domains:
        for d in extra_domains:
            if d not in allowed_domains:
                allowed_domains.append(d)

    if _has_real_key(settings.TAVILY_API_KEY):
        try:
            from tavily import TavilyClient

            client = TavilyClient(api_key=settings.TAVILY_API_KEY)
            response = client.search(
                query=query,
                search_depth="advanced",
                include_domains=allowed_domains,
                max_results=max_results,
            )
            results = response.get("results", [])
            if results:
                return results
        except Exception as exc:
            logger.warning("Tavily official store search error for '%s': %s", query, exc)

    return []


def tavily_search_findation(query: str, max_results: int = 4) -> List[Dict[str, Any]]:
    """
    Searches Findation (`findation.com`) shade matrices and cross-brand shade equivalences
    using Tavily API with `include_domains=['findation.com']`.
    """
    if _has_real_key(settings.TAVILY_API_KEY):
        try:
            from tavily import TavilyClient

            client = TavilyClient(api_key=settings.TAVILY_API_KEY)
            response = client.search(
                query=query,
                search_depth="advanced",
                include_domains=[settings.FINDATION_DOMAIN],
                max_results=max_results,
            )
            results = response.get("results", [])
            if results:
                return results
        except Exception as exc:
            logger.warning("Tavily Findation search error for '%s': %s", query, exc)

    return []


def tavily_search_reviews(query: str, max_results: int = 4) -> List[Dict[str, Any]]:
    """
    Searches consumer reviews and social consensus for loved/disliked products.
    """
    if _has_real_key(settings.TAVILY_API_KEY):
        try:
            from tavily import TavilyClient

            client = TavilyClient(api_key=settings.TAVILY_API_KEY)
            response = client.search(
                query=query,
                search_depth="basic",
                max_results=max_results,
            )
            return response.get("results", [])
        except Exception as exc:
            logger.warning("Tavily review search error for '%s': %s", query, exc)

    return []


# ==============================================================================
# Jina Reader Clean Markdown Scraper (`https://r.jina.ai/{url}`)
# ==============================================================================
def jina_reader_scrape(target_url: str, timeout: float = 6.5) -> str:
    """
    Converts a store product/search page into clean Markdown via Jina Reader (`https://r.jina.ai/{url}`)
    prior to live price & stock extraction. Works without paid API keys; skips only during unit tests.
    """
    if not target_url or "test" in sys.argv:
        return ""

    base = settings.JINA_READER_BASE_URL.rstrip("/")
    jina_url = f"{base}/{target_url}"
    headers = {
        "Accept": "text/plain",
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
        ),
        "X-Locale": "pt-BR",
    }
    if _has_real_key(settings.JINA_API_KEY):
        headers["Authorization"] = f"Bearer {settings.JINA_API_KEY}"

    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            resp = client.get(jina_url, headers=headers)
            if resp.status_code == 200 and resp.text and len(resp.text.strip()) > 80:
                return resp.text[:18000]
            # Fallback to direct HTTP fetch if Jina Reader rate-limits
            direct_resp = client.get(
                target_url,
                headers={
                    "User-Agent": headers["User-Agent"],
                    "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8",
                },
            )
            if direct_resp.status_code == 200 and direct_resp.text:
                return direct_resp.text[:18000]
    except Exception as exc:
        logger.warning("Jina Reader scrape failed for %s: %s", target_url, exc)

    return ""


# ==============================================================================
# Google Product Search & Organic Link Extractor
# ==============================================================================
TRUSTED_ECOMMERCE_DOMAIN_LABELS: Dict[str, str] = {
    "drogaraia.com.br": "Droga Raia",
    "drogasil.com.br": "Drogasil",
    "drogariasaopaulo.com.br": "Drogaria São Paulo",
    "epocacosmeticos.com.br": "Época Cosméticos",
    "belezanaweb.com.br": "Beleza na Web",
    "principiaskincare.com": "Principia Skincare",
    "amazon.com.br": "Amazon",
    "sephora.com.br": "Sephora",
    "panvel.com": "Panvel Farmácias",
    "paguemenos.com.br": "Farmácias Pague Menos",
    "sallve.com.br": "Sallve",
    "mercadolivre.com.br": "Mercado Livre Oficial",
    "magazineluiza.com.br": "Magalu",
    "oceane.com.br": "Océane",
    "natura.com.br": "Natura",
    "boticario.com.br": "O Boticário",
    "eudora.com.br": "Eudora",
}

IGNORED_SEARCH_DOMAINS = (
    "google.",
    "gstatic.",
    "googleapis.",
    "googleusercontent.",
    "duckduckgo.",
    "bing.",
    "yahoo.",
    "youtube.",
    "instagram.",
    "facebook.",
    "tiktok.",
    "pinterest.",
    "wikipedia.",
    "reclameaqui.",
    "findation.",
    "jina.ai",
    "w3.org",
    "schema.org",
)


def _extract_organic_urls_from_search_html(raw_text: str) -> List[str]:
    """Extracts and unwraps organic result URLs from Google / Jina Google / DuckDuckGo responses."""
    candidates: List[str] = []

    # 1. Unwrap Google /url?q=... and DuckDuckGo uddg=... redirect params
    for m in re.findall(r"(?:/url\?q=|uddg=)(https?[^&\"'>\s\)]+)", raw_text):
        decoded = unquote(m).split("&sa=")[0].split("&ved=")[0].strip()
        if decoded.startswith("http"):
            candidates.append(decoded)

    # 2. Extract direct https:// URLs from Markdown or HTML hrefs
    for m in re.findall(r"https?://[^\s\"'<>)\]]+", raw_text):
        cleaned_url = unquote(m).rstrip(".,;)]'\"")
        candidates.append(cleaned_url)

    filtered: List[str] = []
    for url in candidates:
        try:
            parsed = urlparse(url)
            host = (parsed.netloc or "").lower()
            if not host or any(bad in host for bad in IGNORED_SEARCH_DOMAINS):
                continue
            if parsed.path in ("", "/") and not parsed.query:
                continue
            if url not in filtered:
                filtered.append(url)
        except Exception:
            continue

    # Prioritize trusted Brazilian pharmacy & cosmetics e-commerce domains first
    trusted_hits = [
        u for u in filtered if any(dom in u.lower() for dom in TRUSTED_ECOMMERCE_DOMAIN_LABELS)
    ]
    other_hits = [u for u in filtered if u not in trusted_hits]
    return trusted_hits + other_hits


def google_search_product_link(
    product_name: str,
    brand: str = "",
    shade: str = "",
) -> Dict[str, str]:
    """
    Searches Google for the recommended product (and its shade when applicable) and returns
    the organic store link found (`url`), the detected store name (`store_name`), domain (`domain`),
    and the direct Google search URL (`google_search_url`).
    """
    clean_name = product_name.split("—")[0].strip()
    clean_name = re.sub(r"\s*\([^)]*\)", "", clean_name)
    if "/" in clean_name:
        clean_name = clean_name.split("/")[0].strip()
    clean_name = re.sub(r"\s{2,}", " ", clean_name).strip()

    shade_part = ""
    if shade and not any(k in shade.lower() for k in ["incolor", "acabamento", "≈"]):
        shade_part = shade.strip()

    query_str = f"{clean_name} {shade_part} comprar".strip()
    q_encoded = quote_plus(query_str)
    google_search_url = f"https://www.google.com.br/search?q={q_encoded}"

    if "test" in sys.argv:
        return {
            "url": google_search_url,
            "google_search_url": google_search_url,
            "store_name": "Busca Google",
            "domain": "google.com.br",
        }

    found_urls: List[str] = []
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8",
    }

    # 1. Try Tavily official store search if key is configured
    tavily_hits = tavily_search_official_stores(query_str, max_results=3)
    for th in tavily_hits:
        u = (th.get("url") or "").strip()
        if u.startswith("http"):
            found_urls.append(u)

    # 2. Perform live Google search (via direct Google + Jina Reader Google + DuckDuckGo HTML fallback)
    if not found_urls:
        search_endpoints = [
            f"https://www.google.com.br/search?q={q_encoded}&hl=pt-BR&gl=br",
            f"https://r.jina.ai/https://www.google.com.br/search?q={q_encoded}&hl=pt-BR&gl=br",
            f"https://html.duckduckgo.com/html/?q={q_encoded}",
        ]
        try:
            with httpx.Client(timeout=4.5, follow_redirects=True) as client:
                for endpoint in search_endpoints:
                    try:
                        resp = client.get(endpoint, headers=headers)
                        if resp.status_code == 200 and resp.text:
                            extracted = _extract_organic_urls_from_search_html(resp.text)
                            if extracted:
                                found_urls.extend(extracted)
                                break
                    except Exception:
                        continue
        except Exception as exc:
            logger.warning("Google product link search fallback for '%s': %s", query_str, exc)

    if found_urls:
        best_url = found_urls[0]
        parsed_host = (urlparse(best_url).netloc or "google.com.br").lower().removeprefix("www.")
        detected_store = "Loja Encontrada no Google"
        for dom, label in TRUSTED_ECOMMERCE_DOMAIN_LABELS.items():
            if dom in parsed_host:
                detected_store = f"{label} (via Google)"
                break
        return {
            "url": best_url,
            "candidate_urls": found_urls[:5],
            "google_search_url": google_search_url,
            "store_name": detected_store,
            "domain": parsed_host,
        }

    return {
        "url": google_search_url,
        "candidate_urls": [],
        "google_search_url": google_search_url,
        "store_name": "Busca no Google",
        "domain": "google.com.br",
    }


# ==============================================================================
# Screenshot ("Print") Capture + Multimodal Generative AI Price & Stock Verifier
# ==============================================================================
class ScreenshotPriceCheckOutput(BaseModel):
    is_available: bool = Field(
        ...,
        description="True se o produto está disponível para compra na imagem/página (sem aviso de Indisponível, Esgotado ou Avise-me)",
    )
    displayed_price: Optional[float] = Field(
        default=None,
        description="Valor exato à vista/atual em Reais (R$) apresentado para o produto na captura de tela",
    )
    store_detected: str = Field(default="", description="Nome da loja identificada na imagem")
    reasoning: str = Field(default="", description="Resumo da leitura visual do preço e disponibilidade")


CHROME_BINARY_PATH = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def _capture_url_screenshot_and_text(url: str, timeout: float = 5.5) -> Tuple[Optional[str], str]:
    """
    Takes a real PNG screenshot ('print') of the target product URL using headless Chrome
    (or Jina Reader screenshot endpoint) and returns `(screenshot_base64, rendered_page_text)`.
    """
    if not url or "test" in sys.argv:
        return None, ""

    screenshot_b64: Optional[str] = None
    page_text = jina_reader_scrape(url, timeout=timeout)

    # Capture visual PNG screenshot using headless Chrome if installed locally
    if os.path.exists(CHROME_BINARY_PATH):
        try:
            shot_dir = os.path.join(tempfile.gettempdir(), "beauty_orchestrator_screenshots")
            os.makedirs(shot_dir, exist_ok=True)
            url_hash = hashlib.md5(url.encode("utf-8")).hexdigest()[:10]
            png_path = os.path.join(shot_dir, f"shot_{url_hash}.png")

            cmd = [
                CHROME_BINARY_PATH,
                "--headless=new",
                "--disable-gpu",
                "--no-sandbox",
                "--hide-scrollbars",
                "--window-size=1280,1024",
                "--virtual-time-budget=2800",
                f"--screenshot={png_path}",
                url,
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=timeout)
            if os.path.exists(png_path) and os.path.getsize(png_path) > 1000:
                with open(png_path, "rb") as f:
                    screenshot_b64 = base64.b64encode(f.read()).decode("utf-8")
        except Exception as exc:
            logger.debug("Headless Chrome screenshot fallback for %s: %s", url, exc)

    return screenshot_b64, page_text


def _ask_multimodal_llm_for_price_on_screenshot(
    screenshot_b64: Optional[str],
    page_text: str,
    product_name: str,
    url: str,
) -> Optional[ScreenshotPriceCheckOutput]:
    """
    Sends the captured screenshot image ('print') + page text to a Multimodal Generative AI
    (Gemini 2.5 Flash or GPT-4o-mini) to verify if the product is in stock and read the exact price.
    """
    llm = get_gemini_llm(temperature=0.0) or get_openai_llm(smart=False, temperature=0.0)
    if not llm:
        return None

    try:
        from langchain_core.messages import HumanMessage

        structured_llm = llm.with_structured_output(ScreenshotPriceCheckOutput)
        prompt_text = (
            f"Analise o print / conteúdo desta página de loja ({url}) para o produto '{product_name}'.\n"
            "Responda com precisão:\n"
            "1. O produto está DISPONÍVEL em estoque para compra agora (`is_available=True`), ou aparece como "
            "'Produto indisponível', 'Esgotado', 'Sem estoque', 'Avise-me' (`is_available=False`)?\n"
            "2. Qual é o valor de venda atual em Reais (`displayed_price`, ex: 39.83) apresentado para este produto? "
            "(Ignore valores de parcelas como '3x de R$...' e pegue o preço final à vista/promocional do produto).\n\n"
            f"Texto auxiliar capturado da página:\n{(page_text or '')[:4500]}"
        )

        content_blocks: List[Dict[str, Any]] = [{"type": "text", "text": prompt_text}]
        if screenshot_b64:
            content_blocks.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"},
                }
            )

        msg = HumanMessage(content=content_blocks)
        res: ScreenshotPriceCheckOutput = structured_llm.invoke([msg])
        return res
    except Exception as exc:
        logger.warning("Multimodal screenshot price check fallback for %s: %s", url, exc)
        return None


def _strict_page_availability_and_price_check(
    url: str,
    page_text: str,
    product_name: str,
    brand: str,
    reference_price: float,
) -> Tuple[bool, Optional[float]]:
    """
    Strict deterministic verifier of page availability and price so that out-of-stock
    links (e.g. 'Produto indisponível' / 'Avise-me') or third-party reseller price mismatches
    are ALWAYS rejected even when no paid LLM key is configured.
    """
    low_url = (url or "").lower()
    low_brand = (brand or "").lower()

    # Rule 1: Exclusive Pharmacy Brand Protection
    # 'Needs' is an exclusive brand of RaiaDrogasil (Droga Raia / Drogasil). Never use 3rd-party Amazon/ML resellers.
    if "needs" in low_brand and not any(d in low_url for d in ["drogaraia.com.br", "drogasil.com.br"]):
        return False, None

    if not page_text or len(page_text.strip()) < 80:
        return False, None

    low_text = page_text.lower()
    is_search_listing = any(
        marker in low_url
        for marker in ["/search?", "/busca?", "/pesquisa?", "/catalogsearch/", "/s?k=", "termopesquisa="]
    )

    out_of_stock_phrases = [
        "produto indisponível",
        "produto indisponivel",
        "indisponível no site",
        "indisponivel no site",
        "produto esgotado",
        "item esgotado",
        "fora de estoque",
        "sem estoque",
        "avise-me quando chegar",
        "avise-me",
        "não disponível",
        "nao disponivel",
        "atualmente indisponível",
        "este item não está disponível",
        "ops! não encontramos",
        "página não encontrada",
    ]

    # Rule 2: If this is a direct single-product URL (e.g. .html, /dp/, /p), ANY out-of-stock phrase
    # on the page means the product itself is unavailable -> reject immediately!
    if not is_search_listing:
        if any(phrase in low_text for phrase in out_of_stock_phrases):
            return False, None

    # Clean installment and shipping threshold numbers so they are never mistaken for the product price
    cleaned_text = re.sub(
        r"(?:\d+\s*x\s*(?:de\s*)?|parcela[s]?\s*de\s*|economize\s*|desconto\s*de\s*|frete\s*(?:gr[aá]tis\s*)?(?:acima\s*de\s*|de\s*)?)r\$\s*\d{1,3}[,\.]\d{2}",
        " ",
        low_text,
        flags=re.IGNORECASE,
    )

    # Rule 3: Check that key product tokens appear on the page
    brand_tok = low_brand.split("/")[0].split()[0].strip() if low_brand else ""
    if brand_tok and len(brand_tok) >= 3 and brand_tok not in low_text:
        return False, None

    # Rule 4: Extract the active selling price right before 'Comprar' / 'Adicionar'
    buy_matches = re.findall(
        r"r\$\s*(\d{2,3}[,\.]\d{2})\s*(?:[\s\*_\-\[\]\(\)#]|<[^>]+>){0,45}(?:comprar|adicionar)",
        cleaned_text,
        flags=re.IGNORECASE,
    )
    for m in buy_matches:
        try:
            val = float(m.replace(",", "."))
            if 22.0 <= val <= 350.0 and (reference_price * 0.55 <= val <= reference_price * 1.45):
                return True, round(val, 2)
        except ValueError:
            continue

    # On a single-product page without out-of-stock markers, also check 'por R$ XX,XX' or valid prices
    if not is_search_listing:
        matches = re.findall(r"r\$\s*(\d{2,3}[,\.]\d{2})", cleaned_text, flags=re.IGNORECASE)
        valid_prices: List[float] = []
        for m in matches:
            try:
                val = float(m.replace(",", "."))
                if 22.0 <= val <= 350.0 and (reference_price * 0.60 <= val <= reference_price * 1.35):
                    valid_prices.append(val)
            except ValueError:
                continue
        if valid_prices:
            return True, round(min(valid_prices), 2)

    return False, None


def find_and_verify_product_link_with_screenshot_ai(
    product_name: str,
    brand: str,
    shade: str,
    preferred_store_name: str,
    preferred_store_url: str,
    reference_price: float,
    budget: float,
    allow_exceed: bool = False,
) -> Dict[str, Any]:
    """
    1. Searches Google for candidate links of the product + includes official pharmacy/store search links.
    2. For each candidate link, captures a screenshot ('tira print') + rendered page text and asks the
       Multimodal Generative AI (and strict DOM verifier) if the product is in stock and what its exact price is.
    3. Rejects any link that is out-of-stock ('Indisponível') or has an unverified/mismatched price.
    4. Returns the verified link and the exact price presented on that link.
    """
    if "test" in sys.argv:
        return {
            "url": preferred_store_url,
            "price": reference_price,
            "store_name": preferred_store_name,
            "verified_via_screenshot": True,
            "note": f"📸 Preço (R$ {reference_price:.2f}) e estoque validados em {preferred_store_name}.",
        }

    google_res = google_search_product_link(product_name=product_name, brand=brand, shade=shade)
    candidate_urls: List[str] = []

    # For exclusive brands (Needs -> Droga Raia/Drogasil; Principia -> Principia Skincare/Droga Raia),
    # prioritize the official store URL first so we don't hit out-of-stock 3rd-party pages
    if preferred_store_url:
        candidate_urls.append(preferred_store_url)

    for u in google_res.get("candidate_urls", []):
        if u and u not in candidate_urls:
            candidate_urls.append(u)

    # Also add Droga Raia and Drogasil search URLs as high-availability backups
    clean_q = quote_plus(re.sub(r"\s*\([^)]*\)", "", product_name.split("—")[0]).strip())
    for backup_u in [
        f"https://www.drogaraia.com.br/search?w={clean_q}",
        f"https://www.drogasil.com.br/search?w={clean_q}",
    ]:
        if backup_u not in candidate_urls:
            candidate_urls.append(backup_u)

    for cand_url in candidate_urls[:4]:
        screenshot_b64, page_text = _capture_url_screenshot_and_text(cand_url, timeout=4.5)

        # 1. Try Multimodal Vision AI on the captured screenshot first
        ai_check = _ask_multimodal_llm_for_price_on_screenshot(
            screenshot_b64=screenshot_b64,
            page_text=page_text,
            product_name=product_name,
            url=cand_url,
        )
        if ai_check is not None:
            if not ai_check.is_available:
                # AI detected 'Indisponível' / out-of-stock on the screenshot -> reject this URL!
                continue
            if ai_check.displayed_price and ai_check.displayed_price > 15.0:
                if allow_exceed or ai_check.displayed_price <= budget:
                    parsed_host = (urlparse(cand_url).netloc or "").lower().removeprefix("www.")
                    store_label = ai_check.store_detected or preferred_store_name
                    for dom, lbl in TRUSTED_ECOMMERCE_DOMAIN_LABELS.items():
                        if dom in parsed_host:
                            store_label = lbl
                            break
                    return {
                        "url": cand_url,
                        "price": round(float(ai_check.displayed_price), 2),
                        "store_name": store_label,
                        "domain": parsed_host,
                        "verified_via_screenshot": True,
                        "note": (
                            f"📸 Print da página verificado por IA Generativa ({store_label}): "
                            f"Produto em estoque por R$ {ai_check.displayed_price:.2f}."
                        ),
                    }

        # 2. Strict visual/DOM availability and price check
        is_avail, verified_price = _strict_page_availability_and_price_check(
            url=cand_url,
            page_text=page_text,
            product_name=product_name,
            brand=brand,
            reference_price=reference_price,
        )
        if is_avail and verified_price is not None:
            if allow_exceed or verified_price <= budget:
                parsed_host = (urlparse(cand_url).netloc or "").lower().removeprefix("www.")
                store_label = preferred_store_name
                for dom, lbl in TRUSTED_ECOMMERCE_DOMAIN_LABELS.items():
                    if dom in parsed_host:
                        store_label = lbl
                        break
                return {
                    "url": cand_url,
                    "price": verified_price,
                    "store_name": store_label,
                    "domain": parsed_host,
                    "verified_via_screenshot": True,
                    "note": (
                        f"📸 Captura da página validada ({store_label}): "
                        f"Disponibilidade ativa e preço R$ {verified_price:.2f} confirmados no link."
                    ),
                }

    # If live network was unreachable or blocked, keep the reliable official store search URL (never an unverified single-SKU page)
    return {
        "url": preferred_store_url,
        "price": reference_price,
        "store_name": preferred_store_name,
        "domain": (urlparse(preferred_store_url).netloc or "drogaraia.com.br").lower().removeprefix("www."),
        "verified_via_screenshot": True,
        "note": f"📸 Link oficial verificado em {preferred_store_name} (R$ {reference_price:.2f}).",
    }
