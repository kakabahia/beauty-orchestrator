"""
Automated tests for Beauty AI Orchestrator covering:
- Step 1.A: Adaptive Dual-Flow Pydantic Discriminated Unions & conditional validations
- Step 1.B: Human-in-the-Loop Product & Reference Base Disambiguation
- Step 1.C: 4-Stage LangGraph Parallel Fan-out / Fan-in Funnel, Price-Aware Sunscreen Selection,
            Budget Fallback, Findation Routing, and Reclame Aqui Pharmacy Links
- Step 1.D: Non-blocking n8n Webhook fault tolerance
"""
import json
from django.test import Client, TestCase


class BeautyOrchestratorTests(TestCase):
    def setUp(self):
        self.client = Client()

    def test_index_view_renders_adaptive_interface(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "Beauty AI Orchestrator")
        self.assertContains(resp, "Jornada Skincare")
        self.assertContains(resp, "Jornada Beleza &amp; Maquiagem")

    def test_brand_first_catalog_endpoint_for_one_by_one_confirmation(self):
        resp = self.client.get("/api/brand-catalog")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("brands", data)
        self.assertIn("products_by_brand", data)
        self.assertIn("ISDIN", data["brands"])
        self.assertIn("La Roche-Posay", data["brands"])
        self.assertIn("M·A·C Cosmetics", data["brands"])
        self.assertGreaterEqual(len(data["products_by_brand"]["Principia"]), 5)

    def test_type_brand_and_return_corresponding_brand_products(self):
        # 1. User types "sallve" -> confirms brand "Sallve" and lists Sallve products
        resp1 = self.client.post(
            "/api/search-brand-products",
            data=json.dumps({"brand_query": "sallve", "product_hint": "hidratante"}),
            content_type="application/json",
        )
        self.assertEqual(resp1.status_code, 200)
        d1 = resp1.json()
        self.assertEqual(d1["confirmed_brand"], "Sallve")
        self.assertGreaterEqual(len(d1["products"]), 5)
        self.assertIn("Hidratante", d1["products"][0]["official_name"])

        # 2. User types "mac" for shade reference -> confirms "M·A·C Cosmetics" and lists bases with shades
        resp2 = self.client.post(
            "/api/search-brand-products",
            data=json.dumps({"brand_query": "mac", "only_shade_products": True}),
            content_type="application/json",
        )
        self.assertEqual(resp2.status_code, 200)
        d2 = resp2.json()
        self.assertTrue(d2["found"])
        self.assertEqual(d2["confirmed_brand"], "M·A·C Cosmetics")
        self.assertIn("NC25", d2["products"][0]["shades"])

        # 3. User types North American / US Findation brands ("Dragun Beauty", "Fenty", "Rare Beauty") -> found=True
        for us_query, expected_brand in [
            ("Dragun Beauty", "Dragun Beauty"),
            ("sheglam", "SHEGLAM"),
            ("fenty", "Fenty Beauty"),
            ("rare beauty", "Rare Beauty"),
            ("niina secrets", "Eudora (Niina Secrets / Glam)"),
            ("make b", "O Boticário (Make B.)"),
            ("vizzela", "Vizzela"),
        ]:
            resp_us = self.client.post(
                "/api/search-brand-products",
                data=json.dumps({"brand_query": us_query, "only_shade_products": True}),
                content_type="application/json",
            )
            self.assertEqual(resp_us.status_code, 200)
            d_us = resp_us.json()
            self.assertTrue(d_us["found"], f"Expected {us_query} to be found")
            self.assertEqual(d_us["confirmed_brand"], expected_brand)
            self.assertGreaterEqual(len(d_us["products"]), 1)

        # 4. Verify SHEGLAM returns all concealer lines (Complexion Boost, 12HR, Magic Touch Cushion, Out Of Sight, Under Eye, Stick, Correct)
        resp_sheglam = self.client.post(
            "/api/search-brand-products",
            data=json.dumps({"brand_query": "sheglam", "only_shade_products": True}),
            content_type="application/json",
        ).json()
        sheglam_concealers = [
            p for p in resp_sheglam["products"] if p["category_detected"] == "corretivo"
        ]
        self.assertGreaterEqual(len(sheglam_concealers), 7)

        # 5. User types an unrecognized/nonexistent brand ("MarcaInexistenteXYZ999") -> returns found=False and empty products list
        resp3 = self.client.post(
            "/api/search-brand-products",
            data=json.dumps({"brand_query": "MarcaInexistenteXYZ999"}),
            content_type="application/json",
        )
        self.assertEqual(resp3.status_code, 200)
        d3 = resp3.json()
        self.assertFalse(d3["found"])
        self.assertEqual(len(d3["products"]), 0)
        self.assertIn("Marca não encontrada", d3["message"])

    def test_human_in_the_loop_disambiguation_with_reference_base(self):
        payload = {
            "loved_inputs": ["protetor com cor da isdin", "neutrogena hydro boost"],
            "disliked_inputs": ["hidratante roxo da sallve"],
            "reference_shade_input": "Base MAC Studio Fix NC25",
        }
        resp = self.client.post(
            "/api/disambiguate",
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["has_tinted_sunscreen_in_loved"])
        self.assertEqual(len(data["cards"]), 4)

        shade_cards = [c for c in data["cards"] if c["sentiment"] == "shade_reference"]
        self.assertEqual(len(shade_cards), 1)
        self.assertEqual(shade_cards[0]["brand"], "M·A·C Cosmetics")
        self.assertEqual(shade_cards[0]["detected_shade"], "NC25")
        self.assertIn("Quente", shade_cards[0]["detected_undertone"])

    def test_reasonable_sunscreen_budget_finds_products_and_reclame_aqui_links(self):
        payload = {
            "flow_type": "skincare_especifico",
            "skin_type": "oleosa",
            "is_sensitive": True,
            "budget": 65.0,  # Reasonable mid-range sunscreen budget
            "category": "protetor_solar",
            "prefers_tinted_sunscreen": True,
            "reference_shade": "MAC Studio Fix NC25",
            "confirmed_products": [],
        }
        resp = self.client.post(
            "/api/recommend",
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertFalse(data["telemetry"]["budget_exceeded_alert"])
        self.assertGreaterEqual(len(data["recommendations"]), 2)
        for rec in data["recommendations"]:
            self.assertLessEqual(rec["price"], 65.0)
            self.assertIn("Reclame Aqui", rec["store_reputation_badge"])
            self.assertTrue(rec["store_url"].startswith("https://"))
            self.assertGreaterEqual(len(rec["alternative_store_links"]), 3)

    def test_skincare_routine_parallel_funnel_and_auto_tint_flag(self):
        disambig = self.client.post(
            "/api/disambiguate",
            data=json.dumps(
                {
                    "loved_inputs": ["protetor com cor da isdin", "neutrogena hydro boost"],
                    "disliked_inputs": ["hidratante roxo da sallve"],
                    "reference_shade_input": "MAC Studio Fix NC25",
                }
            ),
            content_type="application/json",
        ).json()

        payload = {
            "flow_type": "skincare_rotina",
            "skin_type": "oleosa",
            "is_sensitive": True,
            "budget": 200.0,
            "active_problems": ["acne", "manchas_melasma"],
            "prefers_tinted_sunscreen": False,  # Auto-enabled by Agent 1B due to loved ISDIN Color!
            "reference_shade": "MAC Studio Fix NC25",
            "confirmed_products": disambig["cards"],
        }
        resp = self.client.post(
            "/api/recommend",
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(len(data["recommendations"]), 3)
        self.assertFalse(data["telemetry"]["budget_exceeded_alert"])
        self.assertTrue(data["telemetry"]["stage_1b_formulation"]["auto_tinted_flag"])
        self.assertTrue(data["telemetry"]["stage_3_findation_executed"])

        # Check colorless vs tinted shade identification
        shades = [r["recommended_shade"] for r in data["recommendations"]]
        self.assertIn("Incolor (sem pigmento)", shades)
        self.assertTrue(any(k in s for s in shades for k in ("Média", "Médio", "Bege Médio", "Medium", "Light")))

    def test_strict_budget_fallback_returns_two_closest_never_empty(self):
        payload = {
            "flow_type": "skincare_especifico",
            "skin_type": "mista",
            "is_sensitive": False,
            "budget": 15.0,  # Intentionally below any official store price to trigger fallback
            "category": "protetor_solar",
            "prefers_tinted_sunscreen": False,
            "confirmed_products": [],
        }
        resp = self.client.post(
            "/api/recommend",
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["telemetry"]["budget_exceeded_alert"])
        self.assertEqual(len(data["recommendations"]), 2)
        self.assertIsNotNone(data["telemetry"]["budget_alert_message"])

    def test_makeup_findation_undertone_fallback_when_no_exact_match(self):
        payload = {
            "flow_type": "maquiagem",
            "skin_type": "oleosa",
            "is_sensitive": True,
            "budget": 120.0,
            "category": "base",
            "reference_shade": "Base Experimental Subtom Quente Dourado X99",
            "confirmed_products": [],
        }
        resp = self.client.post(
            "/api/recommend",
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["telemetry"]["stage_3_findation_executed"])
        first = data["recommendations"][0]
        self.assertEqual(first["shade_match_method"], "undertone_approx")
        self.assertIn("Quente", first["undertone_Detected"])

    def test_conditional_validation_for_tratamento_and_chromatic_makeup(self):
        bad_skincare = {
            "flow_type": "skincare_especifico",
            "skin_type": "oleosa",
            "is_sensitive": False,
            "budget": 100.0,
            "category": "tratamento",
            "target_problem": "",
        }
        resp1 = self.client.post(
            "/api/recommend",
            data=json.dumps(bad_skincare),
            content_type="application/json",
        )
        self.assertEqual(resp1.status_code, 422)

        bad_makeup = {
            "flow_type": "maquiagem",
            "skin_type": "mista",
            "is_sensitive": False,
            "budget": 100.0,
            "category": "blush",
            "chromatic_expectation": "",
        }
        resp2 = self.client.post(
            "/api/recommend",
            data=json.dumps(bad_makeup),
            content_type="application/json",
        )
        self.assertEqual(resp2.status_code, 422)

    def test_exact_product_shade_names_and_clean_store_urls(self):
        payload = {
            "flow_type": "skincare_especifico",
            "skin_type": "mista",
            "is_sensitive": True,
            "budget": 65.0,
            "category": "protetor_solar",
            "prefers_tinted_sunscreen": True,
            "confirmed_products": [
                {
                    "id": "ref_1",
                    "raw_input": "Bruna Tavares BT Skin — Tom: M20",
                    "sentiment": "shade_reference",
                    "official_name": "Bruna Tavares BT Skin Base Líquida Ultra Fluida — Tom: M20",
                    "brand": "Bruna Tavares",
                    "predominant_texture": "Fluido Sérum Segunda Pele",
                    "category_detected": "base",
                    "is_tinted_sunscreen": False,
                    "detected_shade": "M20",
                    "detected_undertone": "Quente",
                    "store_name": "Época Cosméticos",
                    "store_url": "https://www.epocacosmeticos.com.br",
                    "confirmed": True,
                },
                {
                    "id": "ref_2",
                    "raw_input": "SHEGLAM 12HR Concealer — Tom: Sand",
                    "sentiment": "shade_reference",
                    "official_name": "SHEGLAM 12HR Full Coverage Concealer — Tom: Sand",
                    "brand": "SHEGLAM",
                    "predominant_texture": "Corretivo Líquido 12h",
                    "category_detected": "corretivo",
                    "is_tinted_sunscreen": False,
                    "detected_shade": "Sand",
                    "detected_undertone": "Quente",
                    "store_name": "Amazon",
                    "store_url": "https://www.amazon.com.br",
                    "confirmed": True,
                },
                {
                    "id": "ref_3",
                    "raw_input": "SHEGLAM Complexion Pro — Tom: Golden (Médio-Escuro Dourado)",
                    "sentiment": "shade_reference",
                    "official_name": "SHEGLAM Complexion Pro Foundation — Tom: Golden (Médio-Escuro Dourado)",
                    "brand": "SHEGLAM",
                    "predominant_texture": "Fluido Matte Respirável",
                    "category_detected": "base",
                    "is_tinted_sunscreen": False,
                    "detected_shade": "Golden (Médio-Escuro Dourado)",
                    "detected_undertone": "Quente",
                    "store_name": "Amazon",
                    "store_url": "https://www.amazon.com.br",
                    "confirmed": True,
                },
            ],
        }
        resp = self.client.post(
            "/api/recommend",
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        recs = data["recommendations"]
        self.assertEqual(len(recs), 3)
        for rec in recs:
            self.assertNotIn("%28", rec["store_url"])
            self.assertNotIn("%29", rec["store_url"])
            self.assertNotEqual(rec["brand"], "Anasol")
            if "Needs" in rec["brand"]:
                self.assertEqual(rec["recommended_shade"], "Bege Médio")
                self.assertAlmostEqual(rec["price"], 39.83, places=2)
            if "Neutrogena" in rec["brand"]:
                self.assertEqual(rec["recommended_shade"], "Cor 2.0 Médio Claro")
            if "Principia" in rec["brand"]:
                self.assertEqual(rec["recommended_shade"], "Cor 2 (Média)")

    def test_freeform_loved_disliked_with_optional_reasons_and_google_links(self):
        payload = {
            "flow_type": "skincare_especifico",
            "skin_type": "oleosa",
            "is_sensitive": True,
            "budget": 60.0,
            "category": "protetor_solar",
            "prefers_tinted_sunscreen": False,
            "confirmed_products": [
                {
                    "id": "freeform_1",
                    "raw_input": "protetor solar da principia ps-02",
                    "sentiment": "loved",
                    "user_reason": "Gosto muito porque é fluido, toque seco e não arde o olho",
                    "confirmed": True,
                },
                {
                    "id": "freeform_2",
                    "raw_input": "neutrogena sun fresh sem cor",
                    "sentiment": "disliked",
                    "user_reason": "Não gostei por causa do cheiro forte de fragrância e deixou oleoso",
                    "confirmed": True,
                },
            ],
        }
        resp = self.client.post(
            "/api/recommend",
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        recs = data["recommendations"]
        self.assertGreaterEqual(len(recs), 2)
        # Verify user's reasons were incorporated into Agent 1A and Agent 1B telemetry
        drivers = " ".join(data["telemetry"]["stage_1a_social"]["success_drivers"])
        complaints = " ".join(data["telemetry"]["stage_1a_social"]["recurring_complaints_to_avoid"])
        forbidden = " ".join(data["telemetry"]["stage_1b_formulation"]["forbidden_ingredients"])
        self.assertIn("toque seco", drivers.lower())
        self.assertIn("fragrância", complaints.lower())
        self.assertIn("fragrância", forbidden.lower())
        # Verify budget limit and Google link presence in alternative_store_links
        for rec in recs:
            self.assertLessEqual(rec["price"], 60.0)
            self.assertNotIn("Neutrogena", rec["product_name"])
            self.assertTrue(any("google.com" in alt["url"] for alt in rec["alternative_store_links"]))

    def test_mixed_observation_analysis_and_strict_screenshot_link_verification(self):
        from recommender.agents.tools import _strict_page_availability_and_price_check

        # 1. Test mixed positive + negative observation splitting via /api/disambiguate
        dis_resp = self.client.post(
            "/api/disambiguate",
            data=json.dumps(
                {
                    "loved_inputs": [],
                    "disliked_inputs": [],
                    "items": [
                        {
                            "id": "card_loved_ps01",
                            "raw_input": "protetor ps-01 principia",
                            "sentiment": "loved",
                            "user_reason": "protetor rende bastante, não deixa white cast, o unico problema é a necessidade de selar com pó mas nada fora do controle.",
                        },
                        {
                            "id": "card_disliked_loreal",
                            "raw_input": "protetor loreal",
                            "sentiment": "disliked",
                            "user_reason": "o preço é ótimo, mas é muito oleoso e deixa white cast",
                        },
                    ],
                }
            ),
            content_type="application/json",
        )
        self.assertEqual(dis_resp.status_code, 200)
        cards = dis_resp.json()["cards"]
        self.assertEqual(len(cards), 2)

        ps01_card = cards[0]
        self.assertIn("Principia Protetor Solar Fluido PS-01", ps01_card["official_name"])
        self.assertTrue(any("rende bastante" in p.lower() for p in ps01_card["extracted_positives"]))
        self.assertTrue(any("white cast" in p.lower() for p in ps01_card["extracted_positives"]))
        self.assertTrue(any("selar com pó" in n.lower() for n in ps01_card["extracted_negatives"]))

        loreal_card = cards[1]
        self.assertIn("L'Oréal", loreal_card["official_name"])
        self.assertTrue(any("ótimo" in p.lower() for p in loreal_card["extracted_positives"]))
        self.assertTrue(any("oleoso" in n.lower() for n in loreal_card["extracted_negatives"]))

        # 2. Test strict page availability & price check (rejecting out-of-stock & 3rd-party reseller mismatches)
        out_of_stock_html = (
            "Minha Sacola (0) | Protetor Solar Principia PS-02 FPS 60 40ml "
            "R$ 49,00 Produto indisponível Avise-me quando chegar"
        )
        avail1, price1 = _strict_page_availability_and_price_check(
            url="https://www.drogaraia.com.br/protetor-solar-principia-ps-02-fps-60-40ml-1493234.html",
            page_text=out_of_stock_html,
            product_name="Principia Protetor Solar Fluido PS-02 FPS 60 Toque Seco Sem Cor",
            brand="Principia",
            reference_price=49.0,
        )
        self.assertFalse(avail1)
        self.assertIsNone(price1)

        # Exclusive brand Needs on Amazon must be rejected
        avail2, price2 = _strict_page_availability_and_price_check(
            url="https://www.amazon.com.br/Protetor-Controle-Oleosidade-Niacinamida-Vitamina/dp/B0FHFPMTVZ",
            page_text="Needs Solar Controle de Oleosidade R$ 65,90 Comprar agora",
            product_name="Needs Solar Protetor Facial Gel-Creme Toque Seco FPS 70 Sem Cor",
            brand="Needs Solar",
            reference_price=39.83,
        )
        self.assertFalse(avail2)
        self.assertIsNone(price2)

        # In-stock Droga Raia page with strikethrough R$ 56,90 and active R$ 39,83 before Comprar must succeed
        in_stock_raia = (
            "Protetor Solar Facial Needs Controle de Oleosidade FPS 70 40g "
            "R$ 56,90 R$ 39,83 Comprar"
        )
        avail3, price3 = _strict_page_availability_and_price_check(
            url="https://www.drogaraia.com.br/search?w=Needs+Solar+Protetor+Facial",
            page_text=in_stock_raia,
            product_name="Needs Solar Protetor Facial Gel-Creme Toque Seco FPS 70 Sem Cor",
            brand="Needs Solar",
            reference_price=39.83,
        )
        self.assertTrue(avail3)
        self.assertAlmostEqual(price3, 39.83, places=2)
