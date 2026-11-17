from django.db import models


class ProductReviewCache(models.Model):
    """
    Local SQLite cache of product metadata, official store prices, and social consensus
    reviews to speed up repeated queries and store n8n-synchronized intelligence.
    """

    official_name = models.CharField(max_length=255, unique=True, db_index=True)
    brand = models.CharField(max_length=120, db_index=True)
    category = models.CharField(max_length=80, db_index=True)
    texture_variant = models.CharField(max_length=120, blank=True, default="")
    is_tinted = models.BooleanField(default=False)
    last_known_price = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    store_url = models.URLField(max_length=500, blank=True, default="")
    store_domain = models.CharField(max_length=120, blank=True, default="")
    pros_json = models.JSONField(default=list, blank=True)
    cons_json = models.JSONField(default=list, blank=True)
    key_actives_json = models.JSONField(default=list, blank=True)
    irritant_flags_json = models.JSONField(default=list, blank=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at"]

    def __str__(self) -> str:
        return f"{self.brand} - {self.official_name}"


class ConsultationLog(models.Model):
    """
    Stores completed multi-agent recommendations and tracks asynchronous n8n webhook status.
    """

    flow_type = models.CharField(max_length=40, db_index=True)
    skin_type = models.CharField(max_length=40)
    is_sensitive = models.BooleanField(default=False)
    budget = models.DecimalField(max_digits=10, decimal_places=2)
    request_payload = models.JSONField(default=dict)
    response_payload = models.JSONField(default=dict)
    budget_exceeded_fallback = models.BooleanField(default=False)
    findation_triggered = models.BooleanField(default=False)
    n8n_dispatched = models.BooleanField(default=False)
    n8n_status_detail = models.CharField(max_length=255, blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"[{self.flow_type}] {self.skin_type} - R$ {self.budget} ({self.created_at:%Y-%m-%d %H:%M})"
