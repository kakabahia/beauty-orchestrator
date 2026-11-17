"""
Asynchronous Non-Blocking Background Webhook Dispatcher for n8n (Step 1.D).
- Fires a non-blocking background thread to `N8N_WEBHOOK_URL` with a strict 3.0s timeout.
- Persists consultation history (`ConsultationLog`) and updates `ProductReviewCache` in SQLite.
- Handles all network/timeout exceptions gracefully so offline n8n never impacts the user.
"""
import logging
import threading
from decimal import Decimal
from typing import Any, Dict, Optional
import httpx
from django.conf import settings

from recommender.models import ConsultationLog, ProductReviewCache
from recommender.schemas import RecommendationResponse

logger = logging.getLogger(__name__)


def _persist_reviews_and_log(
    request_dict: Dict[str, Any],
    response_obj: RecommendationResponse,
) -> Optional[int]:
    """Saves consultation history and caches product reviews in SQLite."""
    try:
        for item in response_obj.recommendations:
            ProductReviewCache.objects.update_or_create(
                official_name=item.product_name,
                defaults={
                    "brand": item.brand,
                    "category": item.category,
                    "texture_variant": item.texture,
                    "is_tinted": item.is_tinted,
                    "last_known_price": Decimal(str(round(item.price, 2))),
                    "store_url": item.store_url,
                    "store_domain": item.store_domain,
                    "pros_json": item.pros,
                    "cons_json": item.cons,
                    "key_actives_json": item.key_actives,
                    "irritant_flags_json": item.irritant_warnings,
                },
            )

        log_entry = ConsultationLog.objects.create(
            flow_type=request_dict.get("flow_type", "unknown"),
            skin_type=request_dict.get("skin_type", "unknown"),
            is_sensitive=bool(request_dict.get("is_sensitive", False)),
            budget=Decimal(str(round(float(request_dict.get("budget", 0.0)), 2))),
            request_payload=request_dict,
            response_payload=response_obj.model_dump(mode="json"),
            budget_exceeded_fallback=response_obj.telemetry.budget_exceeded_alert,
            findation_triggered=response_obj.telemetry.stage_3_findation_executed,
            n8n_dispatched=False,
            n8n_status_detail="Queued for async dispatch",
        )
        return log_entry.id
    except Exception as exc:
        logger.warning("SQLite persistence warning: %s", exc)
        return None


def _send_n8n_webhook_sync(
    log_id: Optional[int],
    payload: Dict[str, Any],
    webhook_url: str,
    timeout_seconds: float = 3.0,
) -> None:
    """Executes the HTTP POST to n8n with a strict 3s timeout inside a daemon thread."""
    if not webhook_url or not webhook_url.strip():
        if log_id:
            ConsultationLog.objects.filter(id=log_id).update(
                n8n_dispatched=False,
                n8n_status_detail="Skipped: N8N_WEBHOOK_URL not configured",
            )
        return

    try:
        with httpx.Client(timeout=timeout_seconds) as client:
            resp = client.post(
                webhook_url.strip(),
                json=payload,
                headers={"Content-Type": "application/json", "X-Source": "Beauty-AI-Orchestrator"},
            )
            dispatched_ok = 200 <= resp.status_code < 300
            status_msg = f"HTTP {resp.status_code}"
    except Exception as exc:
        # Never propagate exceptions to the user or crash background worker
        dispatched_ok = False
        status_msg = f"Offline/Timeout handled safely ({type(exc).__name__})"
        logger.info("n8n webhook non-blocking fallback (%s): %s", webhook_url, status_msg)

    if log_id:
        try:
            ConsultationLog.objects.filter(id=log_id).update(
                n8n_dispatched=dispatched_ok,
                n8n_status_detail=status_msg[:250],
            )
        except Exception:
            pass


def trigger_n8n_webhook_async(
    request_dict: Dict[str, Any],
    response_obj: RecommendationResponse,
) -> None:
    """
    Non-blocking entry point: saves local SQLite cache & dispatches webhook to n8n
    in a background daemon thread with a 3-second timeout.
    """
    log_id = _persist_reviews_and_log(request_dict, response_obj)
    webhook_url = getattr(settings, "N8N_WEBHOOK_URL", "")
    timeout_sec = float(getattr(settings, "N8N_WEBHOOK_TIMEOUT_SECONDS", 3.0))

    webhook_payload = {
        "event": "beauty_recommendation_completed",
        "consultation_id": log_id,
        "request": request_dict,
        "recommendations": [r.model_dump(mode="json") for r in response_obj.recommendations],
        "telemetry": response_obj.telemetry.model_dump(mode="json"),
        "curated_summary": response_obj.curated_summary,
    }

    worker = threading.Thread(
        target=_send_n8n_webhook_sync,
        args=(log_id, webhook_payload, webhook_url, timeout_sec),
        daemon=True,
        name="n8n-webhook-dispatcher",
    )
    worker.start()
