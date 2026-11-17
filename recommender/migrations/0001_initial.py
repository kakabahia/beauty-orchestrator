# Generated initial migration for Beauty AI Orchestrator
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="ConsultationLog",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("flow_type", models.CharField(db_index=True, max_length=40)),
                ("skin_type", models.CharField(max_length=40)),
                ("is_sensitive", models.BooleanField(default=False)),
                ("budget", models.DecimalField(decimal_places=2, max_digits=10)),
                ("request_payload", models.JSONField(default=dict)),
                ("response_payload", models.JSONField(default=dict)),
                ("budget_exceeded_fallback", models.BooleanField(default=False)),
                ("findation_triggered", models.BooleanField(default=False)),
                ("n8n_dispatched", models.BooleanField(default=False)),
                ("n8n_status_detail", models.CharField(blank=True, default="", max_length=255)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
            ],
            options={
                "ordering": ["-created_at"],
            },
        ),
        migrations.CreateModel(
            name="ProductReviewCache",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("official_name", models.CharField(db_index=True, max_length=255, unique=True)),
                ("brand", models.CharField(db_index=True, max_length=120)),
                ("category", models.CharField(db_index=True, max_length=80)),
                ("texture_variant", models.CharField(blank=True, default="", max_length=120)),
                ("is_tinted", models.BooleanField(default=False)),
                ("last_known_price", models.DecimalField(blank=True, decimal_places=2, max_digits=10, null=True)),
                ("store_url", models.URLField(blank=True, default="", max_length=500)),
                ("store_domain", models.CharField(blank=True, default="", max_length=120)),
                ("pros_json", models.JSONField(blank=True, default=list)),
                ("cons_json", models.JSONField(blank=True, default=list)),
                ("key_actives_json", models.JSONField(blank=True, default=list)),
                ("irritant_flags_json", models.JSONField(blank=True, default=list)),
                ("updated_at", models.DateTimeField(auto_now=True)),
            ],
            options={
                "ordering": ["-updated_at"],
            },
        ),
    ]
