from django.contrib import admin
from .models import ConsultationLog, ProductReviewCache


@admin.register(ProductReviewCache)
class ProductReviewCacheAdmin(admin.ModelAdmin):
    list_display = ("official_name", "brand", "category", "texture_variant", "last_known_price", "is_tinted", "updated_at")
    search_fields = ("official_name", "brand", "category")
    list_filter = ("category", "is_tinted", "store_domain")


@admin.register(ConsultationLog)
class ConsultationLogAdmin(admin.ModelAdmin):
    list_display = ("id", "flow_type", "skin_type", "is_sensitive", "budget", "findation_triggered", "n8n_dispatched", "created_at")
    list_filter = ("flow_type", "skin_type", "is_sensitive", "findation_triggered", "n8n_dispatched")
    readonly_fields = ("created_at",)
