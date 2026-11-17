"""
Comprehensive Findation Brand & Product Catalog (`recommender/agents/brand_catalog.py`).
Covers Brazilian, North American (US), and International brands listed on Findation.com,
plus Dermocosmetic/Skincare brands sold in official pharmacies and stores.
"""
import logging
import unicodedata
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from recommender.schemas import BrandProductOption, BrandSearchResponse
from .disambiguator import build_reliable_store_url
from .tools import get_gemini_llm, get_openai_llm, tavily_search_official_stores

logger = logging.getLogger(__name__)


class LLMDiscoveredBrandProducts(BaseModel):
    confirmed_brand: str
    products: List[BrandProductOption] = Field(default_factory=list)


def _normalize(text: str) -> str:
    if not text:
        return ""
    nfkd = unicodedata.normalize("NFKD", text.lower().strip())
    return "".join(c for c in nfkd if not unicodedata.combining(c))


# ==============================================================================
# Aliases mapping informal queries to Canonical Brand Names (BR + US / Findation)
# ==============================================================================
BRAND_ALIASES: Dict[str, List[str]] = {
    # --- Dermocosméticos & Skincare ---
    "La Roche-Posay": ["la roche", "laroche", "la roche-posay", "lrp", "posay", "anthelios", "effaclar", "cicaplast", "hyalu b5"],
    "ISDIN": ["isdin", "fusion water", "fotoprotector isdin", "acniben", "isdinceutics", "fotoultra"],
    "Principia": ["principia", "princípia", "principia skincare", "nc-10", "gl-01", "gl-02", "gh-01", "ps-01", "ps-02"],
    "Sallve": ["sallve", "salve", "hidratante roxo", "hidratante reparador sallve"],
    "Mantecorp Skincare": ["mantecorp", "episol", "epidrat", "glycare", "ivy c", "episol sec"],
    "CeraVe": ["cerave", "cera ve"],
    "Neutrogena": ["neutrogena", "hydro boost", "sun fresh", "deep clean", "purified skin"],
    "Eucerin": ["eucerin", "oil control eucerin", "anti-pigment", "thiamidol", "dermopure"],
    "Vichy": ["vichy", "mineral 89", "minéral 89", "normaderm", "capital soleil", "liftactiv"],
    "Bioderma": ["bioderma", "sensibio", "sebium", "sébium", "photoderm"],
    "Avène": ["avene", "avène", "cleanance", "cicalfate", "mat perfect"],
    "Creamy Skincare": ["creamy", "creamy skincare", "calming cream"],
    "Bioré": ["biore", "bioré", "aqua rich", "cleansing oil biore"],
    "Needs Solar": ["needs", "needs solar", "needs beauty"],
    "Anasol": ["anasol"],
    "Adcos": ["adcos", "protetor stick adcos", "derma complex"],
    "Dermage": ["dermage", "photoage", "secatriz"],
    "Darrow (Actine)": ["darrow", "actine", "doctar", "nutriol"],
    "Cetaphil": ["cetaphil", "cetafil", "pro ac", "optimal hydration"],
    "The Ordinary": ["the ordinary", "ordinary", "niacinamide 10% + zinc 1%", "glycolic acid 7%"],
    "Nivea": ["nivea", "nivea sun", "nivea facial", "cellular luminous 630", "luminous 630"],
    "Garnier": ["garnier", "uniform & matte", "uniform e matte", "vitamina c garnier"],
    "Profuse (Aché)": ["profuse", "encefabol", "puriance", "hidratadeeep", "clariage"],
    "Hada Labo": ["hada labo", "hadalabo", "gokujyun", "shirojyun"],
    "COSRX": ["cosrx", "snail mucin", "bha blackhead", "good morning gel"],
    "SkinCeuticals": ["skinceuticals", "skin ceuticals", "blemish + age", "ce ferulic", "ha intensifier", "phyto corrective"],
    "Caudalie": ["caudalie", "vinoperfect", "vinopure", "beauty elixir"],
    "Granado / Phebo": ["granado", "phebo", "granado dermocosmeticos", "enxofre granado"],

    # --- Marcas Brasileiras no Findation ---
    "Bruna Tavares": ["bruna tavares", "bt", "bt skin", "bt plush", "bt skinplush", "bt glowtion"],
    "Boca Rosa Beauty": ["boca rosa", "boca rosa beauty", "bianca andrade"],
    "Natura Una": ["natura", "natura una", "una", "nude me", "natura faces", "natura aquarela", "chronos"],
    "Mari Maria Makeup": ["mari maria", "marimaria", "velvet skin", "cover up"],
    "O Boticário (Make B.)": ["boticario", "o boticario", "o boticário", "make b", "make b.", "intense boticario"],
    "Quem Disse, Berenice?": ["quem disse berenice", "qdb", "berenice"],
    "Eudora (Niina Secrets / Glam)": ["eudora", "niina secrets", "niina", "hidra glow", "daily tint", "eudora glam"],
    "Fran by Franciny Ehlke": ["fran", "franciny", "franciny ehlke", "real filter"],
    "Karen Bachini Beauty": ["karen bachini", "kah", "karen"],
    "Vizzela": ["vizzela", "vizela", "base vizzela"],
    "Tracta": ["tracta", "base tracta"],
    "Dailus": ["dailus", "base dailus", "ultra cobertura"],
    "Vult": ["vult", "hidraluronic", "nano hd"],
    "Ruby Rose": ["ruby rose", "feels mood", "natural look", "soft matte"],
    "Océane": ["oceane", "océane", "oceane edition"],
    "Payot": ["payot", "retinol payot", "base payot"],
    "Catharine Hill": ["catharine hill", "catherine hill", "angel wings"],
    "Alice Salazar": ["alice salazar"],
    "LFpro (Lu Ferraes)": ["lfpro", "lf pro", "lu ferraes"],
    "Bauny Cosméticos": ["bauny", "bauny cosmeticos"],
    "BM Beauty (Bruna Malheiros)": ["bm beauty", "bruna malheiros"],
    "Vic Beauté": ["vic beaute", "vic beauté"],
    "Simple Organic": ["simple organic"],
    "Care Natural Beauty": ["care natural", "care natural beauty"],
    "Contém 1g": ["contem 1g", "contém 1g", "velvet contem 1g"],
    "Avon": ["avon", "power stay", "avon tratamake"],

    # --- Marcas Norte-Americanas (EUA) & Internacionais no Findation ---
    "M·A·C Cosmetics": ["mac", "m.a.c", "m·a·c", "studio fix", "studio radiance", "face and body"],
    "Fenty Beauty": ["fenty", "fenty beauty", "rihanna", "pro filtr", "pro filt'r", "eaze drop"],
    "Rare Beauty": ["rare", "rare beauty", "selena gomez", "liquid touch"],
    "NARS": ["nars", "radiant creamy", "light reflecting", "sheer glow", "natural radiant"],
    "Dragun Beauty": ["dragun", "dragun beauty", "nikita dragun", "dragunheart", "transformation powder"],
    "Haus Labs by Lady Gaga": ["haus labs", "haus laboratories", "lady gaga", "triclone"],
    "Huda Beauty": ["huda", "huda beauty", "fauxfilter", "faux filter", "easy blur"],
    "Too Faced": ["too faced", "born this way"],
    "Estée Lauder": ["estee lauder", "estée lauder", "double wear"],
    "Tarte Cosmetics": ["tarte", "shape tape", "amazonian clay", "maracuja"],
    "Charlotte Tilbury": ["charlotte tilbury", "airbrush flawless", "hollywood flawless filter", "beautiful skin"],
    "Maybelline NY": ["maybelline", "fit me", "superstay", "age rewind", "instant age rewind", "sky high"],
    "L'Oréal Paris": ["loreal", "l'oreal", "l'oréal", "infallible", "true match", "solar expertise", "revitalift"],
    "Revlon": ["revlon", "colorstay"],
    "e.l.f. Cosmetics": ["elf", "e.l.f", "e.l.f.", "halo glow", "camo concealer", "flawless satin"],
    "NYX Professional Makeup": ["nyx", "can't stop won't stop", "cant stop wont stop", "bare with me", "born to glow"],
    "Milani": ["milani", "conceal + perfect", "conceal and perfect"],
    "CoverGirl": ["covergirl", "cover girl", "clean fresh", "trublend"],
    "Wet n Wild": ["wet n wild", "photofocus", "photo focus"],
    "ColourPop": ["colourpop", "colorpop", "pretty fresh"],
    "Anastasia Beverly Hills": ["anastasia", "abh", "anastasia beverly hills", "luminous foundation", "magic touch"],
    "IT Cosmetics": ["it cosmetics", "cc+ cream", "your skin but better"],
    "Urban Decay": ["urban decay", "stay naked", "all nighter", "face bond"],
    "Clinique": ["clinique", "even better", "beyond perfecting"],
    "Laura Mercier": ["laura mercier", "real flawless", "tinted moisturizer"],
    "Bobbi Brown": ["bobbi brown", "skin long-wear", "weightless"],
    "Make Up For Ever": ["make up for ever", "mufe", "hd skin", "ultra hd"],
    "KVD Beauty": ["kvd", "kat von d", "good apple", "lock-it"],
    "Kosas": ["kosas", "revealer"],
    "Ilia Beauty": ["ilia", "super serum skin tint", "true skin"],
    "Tower 28": ["tower 28", "tower28", "sunnydays", "swipe concealer"],
    "Saie": ["saie", "slip tint", "glowy super skin"],
    "Milk Makeup": ["milk", "milk makeup", "flex foundation", "hydro grip"],
    "Glossier": ["glossier", "stretch fluid", "perfecting skin tint"],
    "Patrick Ta": ["patrick ta", "major skin"],
    "ONE/SIZE by Patrick Starrr": ["one size", "one/size", "patrick starrr", "turn up the base"],
    "Morphe": ["morphe", "filter effect", "hint hint skin tint"],
    "Juvia's Place": ["juvias place", "juvia's place", "i am magic"],
    "Kylie Cosmetics": ["kylie", "kylie cosmetics", "power plush"],
    "r.e.m. beauty": ["rem beauty", "r.e.m. beauty", "ariana grande", "sweetener foundation"],
    "Smashbox": ["smashbox", "studio skin", "always on"],
    "bareMinerals": ["bareminerals", "bare minerals", "complexion rescue", "original liquid"],
    "Mary Kay": ["mary kay", "timewise", "timewise 3d"],
    "Dior": ["dior", "dior forever", "backstage", "dior backstage"],
    "Lancôme": ["lancome", "lancôme", "teint idole"],
    "Giorgio Armani": ["armani", "giorgio armani", "luminous silk", "power fabric"],
    "Shiseido": ["shiseido", "synchro skin"],
    "SHEGLAM": ["sheglam", "she glam", "shein", "complexion pro", "skinfinite", "skin influencer", "color bloom", "sun sculpt"],
}


def _make_base_entry(
    official_name: str,
    texture: str,
    shades: List[str],
    category: str = "base",
    store_name: str = "Sephora",
    is_tinted_sunscreen: bool = False,
) -> Dict[str, Any]:
    return {
        "official_name": official_name,
        "predominant_texture": texture,
        "category_detected": category,
        "is_tinted_sunscreen": is_tinted_sunscreen,
        "store_name": store_name,
        "shades": shades,
    }


BRAND_PRODUCT_CATALOG: Dict[str, List[Dict[str, Any]]] = {
    # ==========================================================================
    # DERMOCOSMÉTICOS & SKINCARE
    # ==========================================================================
    "La Roche-Posay": [
        _make_base_entry(
            "La Roche-Posay Anthelios Airlicium+ Com Cor FPS 80",
            "Gel-Creme Antibrilho Alta Cobertura Com Cor",
            ["1.0 Muito Clara", "2.0 Clara", "3.0 Média", "4.0 Morena", "5.0 Morena Mais", "6.0 Negra"],
            category="protetor_solar",
            store_name="Drogasil",
            is_tinted_sunscreen=True,
        ),
        _make_base_entry("La Roche-Posay Anthelios Airlicium Sem Cor FPS 70 Gel-Creme", "Gel-Creme Toque Seco Antibrilho Incolor", [], "protetor_solar", "Droga Raia"),
        _make_base_entry("La Roche-Posay Anthelios UVMune 400 Fluido Oil Control FPS 60", "Fluido Ultraleve Oil Control Incolor", [], "protetor_solar", "Droga Raia"),
        _make_base_entry("La Roche-Posay Anthelios UVMune 400 Fluido Com Cor FPS 60", "Fluido Ultraleve Com Cor Uniformizadora", ["Cor Única Adaptável"], "protetor_solar", "Droga Raia", True),
        _make_base_entry("La Roche-Posay Anthelios Hydraox FPS 60 Fluido Antioxidante", "Fluido Ultraleve Invisível com Água Termal", [], "protetor_solar", "Drogasil"),
        _make_base_entry("La Roche-Posay Anthelios AE-Pigmentation FPS 50 Com Cor", "Gel-Creme Clareador Anti-Manchas Com Cor (Ácido Tranexâmico)", ["Clara", "Morena"], "protetor_solar", "Droga Raia", True),
        _make_base_entry("La Roche-Posay Effaclar Gel Concentrado de Limpeza Antiacne", "Gel Espumante Purificante (Ácido Salicílico + LHA + Zinco)", [], "sabonete_facial", "Droga Raia"),
        _make_base_entry("La Roche-Posay Effaclar Gel Alta Tolerância (Pele Sensível)", "Gel Espumante Suave Sem Ácidos Fortes e Sem Sabão", [], "sabonete_facial", "Drogasil"),
        _make_base_entry("La Roche-Posay Effaclar Sabonete em Barra Concentrado / Alta Tolerância", "Sabonete Dermatológico Antioleosidade", [], "sabonete_facial", "Droga Raia"),
        _make_base_entry("La Roche-Posay Água Micelar Ultra Pele Sensível / Reativa", "Solução Micelar Calmante com Glicerina e Água Termal", [], "limpeza_micelar", "Droga Raia"),
        _make_base_entry("La Roche-Posay Água Micelar Ultra Pele Oleosa (Effaclar)", "Solução Micelar Purificante com Zinco", [], "limpeza_micelar", "Drogasil"),
        _make_base_entry("La Roche-Posay Cicaplast Baume B5+ Multirreparador Calmante", "Bálsamo Creme Reparador de Barreira (Pantenol 5% + Madecassoside)", [], "hidratante", "Droga Raia"),
        _make_base_entry("La Roche-Posay Toleriane Sensitive Hidratante Prebiótico Calmante", "Creme-Gel Hidratante Calmante Sem Fragrância (Niacinamida + Ceramida)", [], "hidratante", "Droga Raia"),
        _make_base_entry("La Roche-Posay Effaclar Mat Hidratante Matificante Antiporos", "Loção Oil-Free Matificante Controladora de Brilho (Sebulyse)", [], "hidratante", "Drogasil"),
        _make_base_entry("La Roche-Posay Effaclar Duo+M Tratamento Antiacne Tripla Correção", "Gel-Creme Hidratante Matificante Anti-Imperfeições", [], "tratamento", "Droga Raia"),
        _make_base_entry("La Roche-Posay Effaclar Ultra Concentrado Sérum Antiacne", "Sérum Peeling Diário (Ácido Salicílico + Glicólico + LHA + Niacinamida)", [], "serum", "Droga Raia"),
        _make_base_entry("La Roche-Posay Hyalu B5 Sérum Reparador Preenchedor", "Sérum Aquoso Hidratante (Duplo Ácido Hialurônico + Vitamina B5)", [], "serum", "Drogasil"),
        _make_base_entry("La Roche-Posay Pure Vitamin C10 Sérum Antioxidante Renovador", "Sérum Fluido Iluminador (10% Vitamina C Pura + Ácido Salicílico + Neurosensine)", [], "serum", "Droga Raia"),
        _make_base_entry("La Roche-Posay Mela B3 Sérum Concentrado Antimanchas", "Sérum Clareador Intensivo (Melasyl + 10% Niacinamida)", [], "tratamento", "Droga Raia"),
        _make_base_entry("La Roche-Posay Retinol B3 Sérum Antirrugas Renovador", "Sérum Fluido Noturno (Retinol Puro + Vitamina B3)", [], "tratamento", "Droga Raia"),
    ],
    "ISDIN": [
        _make_base_entry(
            "ISDIN Fotoprotector Fusion Water Color FPS 50",
            "Fluido Aquoso Ultraleve Com Cor (Toque Seco / Não Arde os Olhos)",
            ["Light (Clara)", "Medium (Média)", "Bronze (Morena/Negra)"],
            category="protetor_solar",
            store_name="Droga Raia",
            is_tinted_sunscreen=True,
        ),
        _make_base_entry("ISDIN Fotoprotector Fusion Water 5 Stars FPS 60 Sem Cor", "Fluido Fase Aquosa Oil-Free Incolor (Safe-Eye Tech)", [], "protetor_solar", "Droga Raia"),
        _make_base_entry("ISDIN FotoUltra Active Unify Color FPS 99 Clareador Com Cor", "Fluido Fotoprotetor Clareador de Manchas/Melasma Com Cor", ["Tom Único Adaptável"], "protetor_solar", "Drogasil", True),
        _make_base_entry("ISDIN FotoUltra Active Unify Sem Cor FPS 99 Clareador", "Fluido Fotoprotetor Clareador Tripla Ação Despigmentante Incolor", [], "protetor_solar", "Droga Raia"),
        _make_base_entry("ISDIN FotoUltra Spot Prevent FPS 99 Prevenção de Manchas", "Fluido Fotoprotetor de Altíssima Proteção UVA", [], "protetor_solar", "Droga Raia"),
        _make_base_entry("ISDIN FotoUltra Age Repair FPS 50 / Color", "Fluido Fotoprotetor Anti-Fotoenvelhecimento", ["Light", "Medium"], "protetor_solar", "Droga Raia", True),
        _make_base_entry("ISDIN Acniben Limpiador Purificante Espuma/Gel Pele Oleosa", "Gel-Espuma Purificante Desobstrutor de Poros (Niacinamida + Zinco)", [], "sabonete_facial", "Droga Raia"),
        _make_base_entry("ISDIN Acniben Control de Brillos y Granos Gel-Creme", "Gel-Creme Facial Matificante para Pele com Tendência Acneica", [], "hidratante", "Droga Raia"),
        _make_base_entry("ISDIN Acniben Night Concentrate Sérum Noturno Anti-Imperfeições", "Sérum Noturno com Retinaldeído + Zinco + Niacinamida", [], "tratamento", "Drogasil"),
        _make_base_entry("ISDIN Essential Cleansing Oil Óleo de Limpeza Facial", "Cleansing Oil com Azeite de Oliva Virgem Extra e Girassol", [], "cleansing_oil", "Droga Raia"),
        _make_base_entry("ISDIN Micellar Solution 4 in 1 Água Micelar Hidratante", "Solução Micelar Demaquilante Sem Álcool e Sem Perfume", [], "limpeza_micelar", "Droga Raia"),
        _make_base_entry("ISDIN Isdinceutics Hyaluronic Concentrate Sérum", "Sérum Aqua-Gel Ultraleve Preenchedor", [], "serum", "Droga Raia"),
        _make_base_entry("ISDIN Isdinceutics Hyaluronic Moisture Oily & Combination Skin", "Gel-Creme Hidratante Matificante com Ácido Hialurônico", [], "hidratante", "Droga Raia"),
        _make_base_entry("ISDIN Isdinceutics Hyaluronic Moisture Sensitive Skin", "Creme Hidratante Calmante Anti-Vermelhidão", [], "hidratante", "Drogasil"),
        _make_base_entry("ISDIN Isdinceutics Melaclear Advanced Sérum Antimanchas", "Sérum Corretor de Hiperpigmentação (Ácido Tranexâmico + Niacinamida)", [], "tratamento", "Droga Raia"),
        _make_base_entry("ISDIN Isdinceutics Retinal Intense Sérum Bifásico Noturno", "Sérum Bifásico Renovador com Retinaldeído + Bakuchiol", [], "tratamento", "Droga Raia"),
    ],
    "Principia": [
        _make_base_entry("Principia Gel de Limpeza GL-01 (2% Ácido Salicílico + 5% Glicerina)", "Gel Espumante Suave Oil-Free", [], "sabonete_facial", "Principia Skincare"),
        _make_base_entry("Principia Gel de Limpeza Suave GL-02 (5% Glicerina + 1% Pantenol)", "Gel Aquoso Ultra-Suave Hipoalergênico (Pele Sensível)", [], "sabonete_facial", "Principia Skincare"),
        _make_base_entry("PrincipiaGel de Limpeza GL-03 (10% Glicerina + 1% Tensoativo)", "Loção Limpadora Hidratante Sem Espuma", [], "sabonete_facial", "Principia Skincare"),
        _make_base_entry("Principia Água Micelar AM-01 (Glicerina + Pantenol)", "Solução Micelar Demaquilante Sem Perfume", [], "limpeza_micelar", "Principia Skincare"),
        _make_base_entry("Principia Tônico AL-07 (7% Ácido Lático + 1% Ácido Salicílico)", "Loção Aquosa Renovadora de Rápida Absorção", [], "tonico", "Principia Skincare"),
        _make_base_entry("Principia Tônico AG-10 (9,8% Ácido Glicólico + Propanediol)", "Loção Esfoliante Química Renovadora", [], "tonico", "Principia Skincare"),
        _make_base_entry("Principia Gel Hidratante GH-01 (2% Ácido Hialurônico + 5% Glicerina)", "Gel Aquoso Transparente Oil-Free", [], "hidratante", "Principia Skincare"),
        _make_base_entry("Principia Creme Hidratante Calmante CM-01 (5% Glicerina + Bisabolol)", "Creme-Loção Reparadora de Barreira", [], "hidratante", "Principia Skincare"),
        _make_base_entry("Principia Emulsão Hidratante Reparadora EH-01 (Ceramidas + Pantenol)", "Loção Fluida Fortalecedora de Barreira Cutânea", [], "hidratante", "Principia Skincare"),
        _make_base_entry("Principia Protetor Solar Fluido PS-01 FPS 60 Sem Cor (5% Niacinamida)", "Fluido Aquoso Ultraleve Toque Seco Incolor", [], "protetor_solar", "Principia Skincare"),
        _make_base_entry("Principia Protetor Solar Fluido PS-02 FPS 60 Toque Seco Sem Cor", "Fluido Aquoso Ultraleve Matificante Incolor", [], "protetor_solar", "Principia Skincare"),
        _make_base_entry("Principia Protetor Solar PS-03 FPS 60 Com Cor", "Fluido Toque Seco Com Cor Uniformizadora", ["Claro", "Médio", "Moreno", "Escuro"], "protetor_solar", "Principia Skincare", True),
        _make_base_entry("Principia Sérum NC-10 (10% Niacinamida + 1% Zinco PCA)", "Sérum Aquoso Leve Oil-Free", [], "serum", "Principia Skincare"),
        _make_base_entry("Principia Sérum AH-2 (2% Ácidos Hialurônicos + Vitamina B5)", "Sérum Gel Hidratante Preenchedor", [], "serum", "Principia Skincare"),
        _make_base_entry("Principia Sérum VC-10 (10% Vitamina C + Ácido Ferúlico)", "Sérum Fluido Antioxidante Iluminador", [], "serum", "Principia Skincare"),
        _make_base_entry("Principia SérumVc-05 / Peptídeos Matrixyl (10% Peptídeos)", "Sérum Pró-Idade Firmador", [], "serum", "Principia Skincare"),
        _make_base_entry("Principia Sérum AM-10 (10% Ácido Mandélico + 1% Alfa-Arbutin)", "Sérum Fluido Renovador Anti-Manchas e Anti-Acne", [], "tratamento", "Principia Skincare"),
        _make_base_entry("Principia Sérum AA-2 (2% Alfa-Arbutin + Ácido Tranexâmico / Niacinamida)", "Sérum Clareador de Manchas e Melasma", [], "tratamento", "Principia Skincare"),
        _make_base_entry("Principia Sérum AT-05 (5% Ácido Tranexâmico + Niacinamida)", "Sérum Despigmentante Anti-Melasma", [], "tratamento", "Principia Skincare"),
        _make_base_entry("Principia Sérum AS-2 (2% Ácido Salicílico + Ácido Lático)", "Sérum Secativo Antiacne e Desobstrutor de Cravos", [], "tratamento", "Principia Skincare"),
        _make_base_entry("Principia Sérum RN-0,3 (0,3% Retinol Puro + Vitamina E)", "Sérum Noturno Renovador Anti-Idade", [], "tratamento", "Principia Skincare"),
        _make_base_entry("Principia Sérum Mix-01 (Niacinamida + Glicólico + Salicílico + Tranexâmico)", "Sérum Renovador Noturno Multi-Ácidos", [], "tratamento", "Principia Skincare"),
        _make_base_entry("Principia Sérum Mix-02 (Cafeína + Ácido Tranexâmico + Ferúlico Olheiras)", "Sérum Concentrado para Área dos Olhos e Bolsas", [], "tratamento", "Principia Skincare"),
    ],
    "Sallve": [
        _make_base_entry("Sallve Hidratante Reparador Múltiplo (Embalagem Roxa)", "Creme Rico Nutritivo (Ceramidas + Pantenol)", [], "hidratante", "Droga Raia"),
        _make_base_entry("Sallve Antioxidante Hidratante FPS 30 (Embalagem Laranja)", "Sérum-Gel Aquoso de Rápida Absorção (Vitamina C + Cafeína)", [], "hidratante", "Droga Raia"),
        _make_base_entry("Sallve Antioxidante Hidratante Sem FPS (Bisnaga Laranja)", "Sérum-Gel Hidratante Antioxidante", [], "hidratante", "Droga Raia"),
        _make_base_entry("Sallve Hidratante Facial Firmador (Embalagem Azul)", "Gel-Creme Leve Aquoso (Pantenol + Cogumelo Fu Ling)", [], "hidratante", "Droga Raia"),
        _make_base_entry("Sallve Hidratante Antiatrito / Calmante", "Creme Suave Protetor de Barreira", [], "hidratante", "Droga Raia"),
        _make_base_entry("Sallve Limpador Facial (Embalagem Amarela)", "Gel Espumante Suave Sem Sulfatos (Niacinamida + Moringa)", [], "sabonete_facial", "Droga Raia"),
        _make_base_entry("Sallve Limpador Antiacne (Ácido Salicílico + Melaleuca)", "Gel de Limpeza Controlador de Oleosidade", [], "sabonete_facial", "Drogasil"),
        _make_base_entry("Sallve Limpador Enzimático em Pó", "Pó Esfoliante Enzimático Suave (Papaína + Argila Branca)", [], "sabonete_facial", "Droga Raia"),
        _make_base_entry("Sallve Bálsamo Demaquilante (Cleansing Balm)", "Bálsamo Óleo-Gel Demaquilante com Óleo de Macadâmia", [], "cleansing_oil", "Droga Raia"),
        _make_base_entry("Sallve Água Micelar (Ácido Lático + Centella Asiática)", "Solução Micelar Limpadora e Calmante", [], "limpeza_micelar", "Droga Raia"),
        _make_base_entry("Sallve Tônico Renovador (AHA 7% - Glicólico + Lático + Málico)", "Tônico Esfoliante Químico Uniformizador", [], "tonico", "Droga Raia"),
        _make_base_entry("Sallve Tônico Adstringente Antiacne (Ácido Salicílico + Centella)", "Tônico Suave Desobstrutor de Poros Sem Álcool", [], "tonico", "Droga Raia"),
        _make_base_entry("Sallve Protetor Solar Toque Seco FPS 60 Sem Cor", "Gel-Creme Toque Seco Matificante (Carnosina + Niacinamida)", [], "protetor_solar", "Droga Raia"),
        _make_base_entry("Sallve Protetor Solar Hidratante FPS 60 Sem Cor", "Loção Fluida Hidratante Luminosa Incolor", [], "protetor_solar", "Droga Raia"),
        _make_base_entry("Sallve Protetor Solar Com Cor FPS 60", "Fluido Uniformizador Com Cor e Carnosina", ["Tom 1", "Tom 2", "Tom 3", "Tom 4", "Tom 5", "Tom 6", "Tom 7", "Tom 8"], "protetor_solar", "Droga Raia", True),
        _make_base_entry("Sallve Protetor Solar Bastão Com Cor FPS 60", "Stick Fotoprotetor Resistente à Água Com Cor", ["Tom 1", "Tom 2", "Tom 3", "Tom 4", "Tom 5", "Tom 6"], "protetor_solar", "Droga Raia", True),
        _make_base_entry("Sallve Sérum Antiacne (2% Ácido Salicílico + Niacinamida)", "Sérum-Gel Secativo e Desobstrutor de Poros", [], "tratamento", "Droga Raia"),
        _make_base_entry("Sallve Sérum Uniformizador (Retinol Biomimético + Ácido Tranexâmico)", "Sérum Fluido Clareador de Manchas e Olheiras", [], "tratamento", "Droga Raia"),
        _make_base_entry("Sallve Super Niacinamida 20% Sérum", "Sérum Aquoso Multi-Corretor de Poros e Manchas", [], "serum", "Droga Raia"),
    ],
    "Mantecorp Skincare": [
        _make_base_entry(
            "Mantecorp Episol Sec OC Com Cor FPS 60",
            "Loção Fluida Toque Seco Com Pigmento Uniformizador",
            ["Muito Claro", "Claro", "Médio", "Médio Escuro", "Escuro", "Muito Escuro"],
            category="protetor_solar",
            store_name="Droga Raia",
            is_tinted_sunscreen=True,
        ),
        _make_base_entry("Mantecorp Episol Color Base Pó Compacto FPS 70", "Base Compacta Fotoprotetora Matificante", ["Pele Muito Clara", "Pele Clara", "Pele Morena", "Pele Morena Mais", "Pele Negra"], "protetor_solar", "Droga Raia", True),
        _make_base_entry("Mantecorp Episol Color Fluido FPS 70 Alta Cobertura", "Fluido Fotoprotetor Base de Alta Cobertura", ["Muito Claro", "Claro", "Médio", "Médio Escuro", "Escuro"], "protetor_solar", "Droga Raia", True),
        _make_base_entry("Mantecorp Episol Sec OC Sem Cor FPS 60", "Loção Fluida Toque Seco Incolor Controle 12h", [], "protetor_solar", "Droga Raia"),
        _make_base_entry("Mantecorp Episol Acqua Gel FPS 60 Sem Cor", "Gel Aquoso Invisível de Rápida Absorção", [], "protetor_solar", "Drogasil"),
        _make_base_entry("Mantecorp Epidrat Calm B5 Hidratante Facial Calmante", "Gel-Creme Balm Calmante Sem Fragrância (Sensarine + D-Pantenol)", [], "hidratante", "Droga Raia"),
        _make_base_entry("Mantecorp Epidrat Calm Loção Hidratante Pele Sensível", "Loção Calmante Hipoalergênica", [], "hidratante", "Droga Raia"),
        _make_base_entry("Mantecorp Epidrat Acqua Hidratante Gel Aquoso", "Gel Aquoso Refrescante Oil-Free (Ácido Hialurônico)", [], "hidratante", "Drogasil"),
        _make_base_entry("Mantecorp Epidrat Mat FPS 30 Hidratante Matificante", "Loção Hidratante Oil-Free com Controle de Brilho", [], "hidratante", "Droga Raia"),
        _make_base_entry("Mantecorp Glycare Control Gel de Limpeza Pele Oleosa", "Gel Espumante Purificante (Ácido Glicólico)", [], "sabonete_facial", "Droga Raia"),
        _make_base_entry("Mantecorp Glycare Sabonete Líquido Facial (Pele Normal a Mista)", "Gel Limpador Renovador com Ácido Glicólico", [], "sabonete_facial", "Drogasil"),
        _make_base_entry("Mantecorp Glycare Sérum / Duo Anti-Idade e Antiporos", "Sérum Renovador com Ácido Glicólico + Niacinamida", [], "serum", "Droga Raia"),
        _make_base_entry("Mantecorp Ivy C Acqua Gel / Aox Vitamina C", "Gel Hidratante Antioxidante com Vitamina C Nanoencapsulada", [], "serum", "Droga Raia"),
        _make_base_entry("Mantecorp Blancy TX Gel Creme Clareador (Ácido Tranexâmico)", "Gel-Creme Despigmentante para Manchas e Melasma", [], "tratamento", "Droga Raia"),
    ],
    "CeraVe": [
        _make_base_entry("CeraVe Gel de Limpeza Facial Espumante (Pele Normal a Oleosa)", "Gel Espumante Refrescante Sem Perfume (3 Ceramidas + Niacinamida)", [], "sabonete_facial", "Droga Raia"),
        _make_base_entry("CeraVe Loção de Limpeza Hidratante (Pele Normal a Seca / Sensível)", "Emulsão de Limpeza Sem Espuma e Sem Perfume", [], "sabonete_facial", "Drogasil"),
        _make_base_entry("CeraVe Gel de Limpeza Acne Control (2% Ácido Salicílico + Argila)", "Gel Espumante Antiacne com Ceramidas", [], "sabonete_facial", "Droga Raia"),
        _make_base_entry("CeraVe Espuma Cremosa de Limpeza Hidratante (Aminoácidos)", "Creme que vira Espuma Demaquilante e Limpadora", [], "sabonete_facial", "Droga Raia"),
        _make_base_entry("CeraVe Óleo de Limpeza Espumante Hidratante (Esqualano)", "Óleo-Gel Limpador para Pele Seca a Muito Seca e Atópica", [], "cleansing_oil", "Droga Raia"),
        _make_base_entry("CeraVe Água Micelar Hidratante (Niacinamida + 3 Ceramidas)", "Solução Micelar Sem Perfume e Oftalmologicamente Testada", [], "limpeza_micelar", "Droga Raia"),
        _make_base_entry("CeraVe Loção Facial Hidratante Ultraleve PM (Niacinamida + Ceramidas)", "Loção Gel-Creme Ultraleve Oil-Free Sem Perfume", [], "hidratante", "Droga Raia"),
        _make_base_entry("CeraVe Loção Facial Hidratante AM FPS 30", "Loção Hidratante Diurna com Fotoproteção e Ceramidas", [], "hidratante", "Drogasil"),
        _make_base_entry("CeraVe Creme Hidratante Reparador de Barreira", "Creme Rico Não-Oleoso (3 Ceramidas + Ácido Hialurônico)", [], "hidratante", "Droga Raia"),
        _make_base_entry("CeraVe Gel Controle de Imperfeições Acne Control (AHA + BHA)", "Gel Noturno Antiacne com Ácido Salicílico, Glicólico, Lático e Ceramidas", [], "tratamento", "Droga Raia"),
        _make_base_entry("CeraVe Sérum Retinol Antimarcas Pós-Acne", "Sérum Encapsulado Uniformizador de Textura e Poros", [], "serum", "Droga Raia"),
        _make_base_entry("CeraVe Sérum Hidratante Ácido Hialurônico + Vitamina B5", "Sérum-Creme Preenchedor Sem Fragrância", [], "serum", "Droga Raia"),
    ],
    "Neutrogena": [
        _make_base_entry("Neutrogena Hydro Boost Water Gel Hidratante Facial", "Water Gel Aquoso Refrescante Oil-Free (Ácido Hialurônico)", [], "hidratante", "Drogasil"),
        _make_base_entry("Neutrogena Hydro Boost Water Gel FPS 25 Hidratante Diurno", "Gel Aquoso Hidratante com Proteção Solar", [], "hidratante", "Droga Raia"),
        _make_base_entry("Neutrogena Face Care Intensive Hidratante Matte 3 em 1", "Creme-Gel Primer Matificante (D-Pantenol + Niacinamida)", [], "hidratante", "Droga Raia"),
        _make_base_entry("Neutrogena Face Care Intensive Antissinais / Reparador", "Creme Hidratante Facial Nutritivo Sem Óleo", [], "hidratante", "Drogasil"),
        _make_base_entry(
            "Neutrogena Sun Fresh Derm Care FPS 70 Com Cor",
            "Gel-Creme Fluido Toque Seco Com Cor (Niacinamida)",
            ["Cor Clara", "Cor Média-Clara", "Cor Média-Escura (Morena)", "Cor Escura (Negra)"],
            category="protetor_solar",
            store_name="Drogasil",
            is_tinted_sunscreen=True,
        ),
        _make_base_entry("Neutrogena Sun Fresh Derm Care Facial Antioleosidade FPS 70 Sem Cor", "Gel-Creme Fluido Matte Invisível", [], "protetor_solar", "Droga Raia"),
        _make_base_entry("Neutrogena Sun Fresh Facial FPS 60 Pele Seca a Normal", "Loção Fluida Hidratante Invisível", [], "protetor_solar", "Droga Raia"),
        _make_base_entry("Neutrogena Purified Skin Gel de Limpeza Profunda (Ácido Glicólico)", "Gel Espumante Purificante Antioleosidade", [], "sabonete_facial", "Droga Raia"),
        _make_base_entry("Neutrogena Deep Clean Intensive Gel de Limpeza (Ácido Salicílico)", "Gel Refrescante Antiacne e Antioleosidade", [], "sabonete_facial", "Drogasil"),
        _make_base_entry("Neutrogena Deep Clean Grapefruit Gel de Limpeza Suave", "Gel Limpador Diário", [], "sabonete_facial", "Droga Raia"),
        _make_base_entry("Neutrogena Hydro Boost Água Micelar 7 em 1", "Solução Micelar com Ácido Hialurônico", [], "limpeza_micelar", "Droga Raia"),
        _make_base_entry("Neutrogena Purified Skin Água Micelar 7 em 1 Antioleosidade", "Solução Micelar Purificante", [], "limpeza_micelar", "Drogasil"),
    ],
    "Eucerin": [
        _make_base_entry("Eucerin Sun Oil Control Tinted FPS 60 Toque Seco Com Cor", "Gel-Creme Ultraleve Oil Control Com Cor Sem Perfume", ["Claro", "Médio"], "protetor_solar", "Droga Raia", True),
        _make_base_entry("Eucerin Sun Oil Control Gel-Creme FPS 60 Toque Seco Sem Cor", "Gel-Creme Ultraleve Matificante Incolor Sem Perfume", [], "protetor_solar", "Droga Raia"),
        _make_base_entry("Eucerin Sun Pigment Control FPS 60 (Com Thiamidol Antimanchas)", "Fluido Fotoprotetor Clareador de Melasma", [], "protetor_solar", "Droga Raia"),
        _make_base_entry("Eucerin Sun Hydro Fluid FPS 60 Ultraleve Invisível", "Fluido Aquoso de Rápida Absorção", [], "protetor_solar", "Drogasil"),
        _make_base_entry("Eucerin Anti-Pigment Dual Sérum Clareador (Thiamidol)", "Sérum Emulsão Dupla Câmara Anti-Manchas e Melasma", [], "tratamento", "Droga Raia"),
        _make_base_entry("Eucerin Anti-Pigment Skin Perfecting Sérum Ultraleve", "Sérum Aquoso Clareador e Iluminador (Thiamidol + Ácido Hialurônico)", [], "serum", "Droga Raia"),
        _make_base_entry("Eucerin Anti-Pigment Creme Dia FPS 30 / Noite", "Creme Hidratante Despigmentante", [], "hidratante", "Droga Raia"),
        _make_base_entry("Eucerin DermoPure Oil Control Gel de Limpeza Facial", "Gel Espumante Sem Sabão e Sem Fragrância", [], "sabonete_facial", "Drogasil"),
        _make_base_entry("Eucerin DermoPure Gel de Limpeza Concentrado (2% Ácido Salicílico)", "Gel Esfoliante Antiacne e Antimarcas", [], "sabonete_facial", "Droga Raia"),
        _make_base_entry("Eucerin DermoPure Sérum Efeito Triplo Antimarcas Pós-Acne", "Sérum Fluido Matificante (Thiamidol + Ácido Salicílico)", [], "tratamento", "Droga Raia"),
        _make_base_entry("Eucerin Hyaluron-Filler Moisture Booster Sérum", "Gel Aquoso Transparente Ultra-Hidratante", [], "hidratante", "Droga Raia"),
    ],
    "Vichy": [
        _make_base_entry("Vichy Capital Soleil UV-Age Daily Com Cor FPS 60", "Fluido Ultraleve Aquoso Com Cor Anti-Fotoenvelhecimento", ["Claro", "Médio", "Moreno", "Escuro"], "protetor_solar", "Droga Raia", True),
        _make_base_entry("Vichy Capital Soleil UV-Age Daily Sem Cor FPS 60", "Fluido Aquoso Invisível com Peptídeos e Niacinamida", [], "protetor_solar", "Droga Raia"),
        _make_base_entry("Vichy Capital Soleil UV-Clear FPS 60 Sem Cor", "Fluido Aquoso Invisível Antioleosidade (Ácido Salicílico + Niacinamida)", [], "protetor_solar", "Droga Raia"),
        _make_base_entry("Vichy Idéal Soleil Purify FPS 70 Com Cor / Sem Cor", "Gel-Creme Toque Seco Purificante com Argila Verde", ["Clara", "Média", "Morena"], "protetor_solar", "Drogasil", True),
        _make_base_entry("Vichy Minéral 89 Sérum Fortalecedor Hidratante", "Gel Aquoso Puro (89% Água Vulcânica + Ácido Hialurônico)", [], "serum", "Droga Raia"),
        _make_base_entry("Vichy Minéral 89 Creme Hidratante Booster 72h", "Gel-Creme Hidratante Reparador de Barreira (Esqualano + Niacinamida)", [], "hidratante", "Droga Raia"),
        _make_base_entry("Vichy Minéral 89 Probiotic Fractions Sérum Regenerador", "Sérum Calmante para Pele Estressada e Sensível", [], "serum", "Droga Raia"),
        _make_base_entry("Vichy Normaderm Phytosolution Gel de Limpeza Intensivo", "Gel Purificante (Ácido Salicílico + Zinco + Cobre)", [], "sabonete_facial", "Drogasil"),
        _make_base_entry("Vichy Normaderm Phytosolution Hidratante Dupla Correção Antiacne", "Loção Aquosa Hidratante e Secativa (2% Ácido Salicílico + Ácido Hialurônico)", [], "hidratante", "Droga Raia"),
        _make_base_entry("Vichy Normaderm Probio-BHA Sérum Antiacne", "Sérum Renovador (5% Ácidos Salicílico e Glicólico + Frações Probióticas)", [], "tratamento", "Droga Raia"),
        _make_base_entry("Vichy Liftactiv B3 Sérum Antimanchas e Melasma", "Sérum Aquoso (Niacinamida + Ácido Tranexâmico + Ácido Glicólico)", [], "tratamento", "Droga Raia"),
        _make_base_entry("Vichy Liftactiv Supreme Vitamina C 15% Sérum Antioxidante", "Sérum Fluido Iluminador Sem Perfume", [], "serum", "Droga Raia"),
    ],
    "Bioderma": [
        _make_base_entry("Bioderma Photoderm Cover Touch Mineral FPS 50+ Com Cor", "Fluido Mineral Alta Cobertura Matte Com Cor", ["Muito Claro", "Claro", "Dourado (Médio)", "Bronze (Morena)", "Escuro"], "protetor_solar", "Droga Raia", True),
        _make_base_entry("Bioderma Photoderm Nude Touch Mineral FPS 50+ Com Cor", "Fluido Bifásico Toque Seco Aveludado Com Cor", ["Muito Claro", "Claro", "Dourado"], "protetor_solar", "Droga Raia", True),
        _make_base_entry("Bioderma Photoderm Aquafluide FPS 50+ Sem Cor / Com Cor", "Fluido Base Aquosa Toque Seco", ["Incolor", "Claro", "Dourado"], "protetor_solar", "Drogasil", True),
        _make_base_entry("Bioderma Sensibio H2O Solução Micelar Calmante", "Água Micelar Biomimética Ultra-Pura Sem Fragrância", [], "limpeza_micelar", "Droga Raia"),
        _make_base_entry("Bioderma Sébium H2O Solução Micelar Pele Mista a Oleosa", "Água Micelar Purificante (Zinco + Cobre)", [], "limpeza_micelar", "Droga Raia"),
        _make_base_entry("Bioderma Sensibio Gel Moussant Limpeza Calmante", "Gel Espumante Suave Hidratante Sem Sabão (Pele Sensível/Rosácea)", [], "sabonete_facial", "Droga Raia"),
        _make_base_entry("Bioderma Sébium Gel Moussant Limpeza Purificante", "Gel Espumante Suave Sem Sabão (Zinco + Cobre)", [], "sabonete_facial", "Drogasil"),
        _make_base_entry("Bioderma Sébium Gel Moussant Actif (Ácido Salicílico + Glicólico)", "Gel Limpador Desobstrutor de Poros e Cravos", [], "sabonete_facial", "Droga Raia"),
        _make_base_entry("Bioderma Sensibio Micellar Cleansing Oil", "Óleo Demaquilante Calmante com Aminoácidos", [], "cleansing_oil", "Droga Raia"),
        _make_base_entry("Bioderma Sensibio Defensive / AR+ Creme Calmante Anti-Vermelhidão", "Creme Fluido Calmante para Pele Sensível e Rosácea", [], "hidratante", "Droga Raia"),
        _make_base_entry("Bioderma Sébium Hydra Hidratante Compensador (Pele Acneica Ressecada)", "Creme Hidratante Calmante Pós-Ácidos/Isotretinoína", [], "hidratante", "Droga Raia"),
        _make_base_entry("Bioderma Sébium Pore Refiner / Kerato+ Gel-Creme", "Concentrado Corretor de Poros Dilatados e Acne", [], "tratamento", "Droga Raia"),
        _make_base_entry("Bioderma Pigmentbio C-Concentrate / Night Renewer Clareador", "Sérum/Gel Clareador de Manchas e Melasma", [], "tratamento", "Droga Raia"),
    ],
    "Avène": [
        _make_base_entry("Avène Mat Perfect Fluido Antiacne Com Cor FPS 60", "Fluido Toque Seco Matificante Com Cor", ["Tom Uniformizador"], "protetor_solar", "Drogasil", True),
        _make_base_entry("Avène Mat Perfect Fluido Antiacne Sem Cor FPS 60", "Fluido Toque Seco Incolor Controlador de Oleosidade", [], "protetor_solar", "Droga Raia"),
        _make_base_entry("Avène Ultra Fluid Oil Control FPS 50 Sem Cor", "Fluido Ultra-Aquoso Absorção em 1 Segundo", [], "protetor_solar", "Droga Raia"),
        _make_base_entry("Avène Cleanance Gel de Limpeza Pele Oleosa e Acneica", "Gel Purificante Suave com Água Termal Avène", [], "sabonete_facial", "Droga Raia"),
        _make_base_entry("Avène Cleanance Comedomed Concentrado Anti-Imperfeições", "Emulsão Gel Secativa Anti-Acne e Anti-Cravos Recorrentes (Comedoclastin)", [], "tratamento", "Droga Raia"),
        _make_base_entry("Avène Cicalfate+ Creme Reparador Calmante", "Creme Reparador de Barreira Sem Perfume", [], "hidratante", "Droga Raia"),
        _make_base_entry("Avène Hydrance Aqua-Gel Hidratante Facial", "Gel-Creme Aquoso Refrescante 3 em 1", [], "hidratante", "Droga Raia"),
        _make_base_entry("Avène Hyaluron Activ B3 Sérum Preenchedor (Niacinamida 6%)", "Sérum Concentrado Firmador Sem Fragrância", [], "serum", "Droga Raia"),
    ],
    "Creamy Skincare": [
        _make_base_entry("Creamy Ácido Tranexâmico + Alfa-Arbutin Sérum Clareador", "Sérum-Gel Aquoso de Rápida Absorção (Embalagem Rosa)", [], "tratamento", "Droga Raia"),
        _make_base_entry("Creamy Ácido Mandélico + Alfa-Arbutin Gel-Creme Noturno", "Gel-Creme Renovador Anti-Manchas e Acne (Embalagem Roxa)", [], "tratamento", "Droga Raia"),
        _make_base_entry("Creamy Ácido Glicólico 10% + Niacinamida Creme Renovador", "Creme Noturno Uniformizador de Textura e Poros (Embalagem Azul)", [], "tratamento", "Droga Raia"),
        _make_base_entry("Creamy Ácido Lático 5% + Niacinamida 5% Sérum", "Sérum Esfoliante Suave Hidratante (Embalagem Amarela)", [], "tratamento", "Droga Raia"),
        _make_base_entry("Creamy Ácido Salicílico 2% Solução Tônica Antiacne", "Tônico Líquido Desobstrutor de Poros e Cravos", [], "tonico", "Droga Raia"),
        _make_base_entry("Creamy Tônico Calmante (Centella Asiática + Pantenol)", "Loção Tônica Hidratante Anti-Vermelhidão", [], "tonico", "Droga Raia"),
        _make_base_entry("Creamy Calming Cream Hidratante Facial Calmante", "Gel-Creme Ultraleve Sem Fragrância para Pele Sensível", [], "hidratante", "Droga Raia"),
        _make_base_entry("Creamy Intensive Repair Cream Hidratante Reparador (Ceramidas)", "Creme Rico Calmante para Pele Seca ou Sensibilizada", [], "hidratante", "Droga Raia"),
        _make_base_entry("Creamy Gel de Limpeza Facial (Pele Normal a Seca / Sensível)", "Gel Suave Sem Sulfatos com Glicerina e Aloe Vera", [], "sabonete_facial", "Droga Raia"),
        _make_base_entry("Creamy Emulsão de Limpeza Calmante", "Loção Cremosa de Limpeza Sem Espuma", [], "sabonete_facial", "Droga Raia"),
        _make_base_entry("Creamy Protetor Solar Watery Lotion FPS 60 Sem Cor", "Fluido Aquoso Ultraleve Toque Seco Incolor", [], "protetor_solar", "Droga Raia"),
        _make_base_entry("Creamy Vitamina C Gold / Sérum Antioxidante", "Sérum Fluido Antioxidante Iluminador", [], "serum", "Droga Raia"),
        _make_base_entry("Creamy Retinol 0,3% Nanoencapsulado Creme Noturno", "Creme Anti-Idade Renovador", [], "tratamento", "Droga Raia"),
    ],
    "Bioré": [
        _make_base_entry("Bioré Make Up Remover Perfect Cleansing Oil (Embalagem Roxa)", "Óleo Demaquilante Emulsionável em Água", [], "cleansing_oil", "Drogasil"),
        _make_base_entry("Bioré The Cleanse Instant Foaming Cleansing Oil", "Óleo Demaquilante de Ação Rápida", [], "cleansing_oil", "Droga Raia"),
        _make_base_entry("Bioré Marshmallow Whip Facial Wash Espuma de Limpeza", "Espuma Densa Ultra-Suave com Tecnologia SPT", [], "sabonete_facial", "Droga Raia"),
        _make_base_entry("Bioré Facial Wash Moisture / Acne Care Sabonete Facial", "Creme Espumante Limpador", [], "sabonete_facial", "Droga Raia"),
        _make_base_entry("Bioré UV Aqua Rich Watery Essence FPS 50 Sem Cor", "Gel Aquoso Microencapsulado Invisível", [], "protetor_solar", "Droga Raia"),
        _make_base_entry("Bioré UV Aqua Rich Watery Gel FPS 50 Sem Cor", "Loção-Gel Aquosa Refrescante", [], "protetor_solar", "Droga Raia"),
        _make_base_entry("Bioré UV Perfect Milk FPS 50 (Embalagem Azul)", "Fluido Leitoso Ultra Matificante Resistente ao Suor", [], "protetor_solar", "Droga Raia"),
        _make_base_entry("Bioré UV Bright Milk FPS 50 (Embalagem Rosa)", "Fluido Leitoso Primer Iluminador Disfarce de Poros", [], "protetor_solar", "Droga Raia"),
        _make_base_entry("Bioré UV Athlizm Essence / Milk FPS 50", "Fluido Fotoprotetor Resistência Extrema ao Calor e Suor", [], "protetor_solar", "Droga Raia"),
    ],
    "Needs Solar": [
        _make_base_entry("Needs Solar Protetor Facial Controle de Oleosidade Com Cor FPS 70", "Gel-Creme Fluido Toque Seco Com Cor", ["Clara", "Média", "Morena", "Negra"], "protetor_solar", "Droga Raia", True),
        _make_base_entry("Needs Solar Protetor Facial Gel-Creme Toque Seco FPS 70 Sem Cor", "Gel-Creme Toque Seco Oil-Free Incolor", [], "protetor_solar", "Droga Raia"),
        _make_base_entry("Needs Solar Protetor Facial Hidratação Equilibrada FPS 70 Sem Cor", "Loção Fluida Hidratante com Ácido Hialurônico", [], "protetor_solar", "Droga Raia"),
        _make_base_entry("Needs Gel Hidratante Facial Ácido Hialurônico + Pepino Oil-Free", "Gel Aquoso Refrescante Oil-Free", [], "hidratante", "Droga Raia"),
        _make_base_entry("Needs Hidratante Facial Controle de Oleosidade (Niacinamida + Chá Verde)", "Gel-Creme Matificante", [], "hidratante", "Droga Raia"),
        _make_base_entry("Needs Gel de Limpeza Facial Controle de Oleosidade (Ácido Salicílico)", "Gel Espumante Refrescante Oil-Free", [], "sabonete_facial", "Droga Raia"),
        _make_base_entry("Needs Sabonete Líquido Facial Ácido Glicólico Renovador", "Gel Limpador Uniformizador", [], "sabonete_facial", "Droga Raia"),
        _make_base_entry("Needs Água Micelar Hidratante / Controle de Oleosidade", "Solução Micelar Demaquilante", [], "limpeza_micelar", "Droga Raia"),
        _make_base_entry("Needs Sérum Facial Niacinamida / Vitamina C", "Sérum Fluido de Rápida Absorção", [], "serum", "Droga Raia"),
    ],
    "Anasol": [
        _make_base_entry("Anasol Protetor Solar Facial Toque Seco Com Cor FPS 50", "Loção Gel-Creme Toque Seco Oil-Free Com Cor", ["Clara", "Média", "Morena", "Escura"], "protetor_solar", "Drogasil", True),
        _make_base_entry("Anasol Protetor Solar Facial Oil-Free Toque Seco Sem Cor FPS 50", "Loção Oil-Free Toque Seco Incolor", [], "protetor_solar", "Drogasil"),
        _make_base_entry("Anasol Protetor Solar Facial Clareador FPS 70 (Niacinamida)", "Loção Fluida Clareadora Anti-Manchas", [], "protetor_solar", "Drogasil"),
        _make_base_entry("Anasol Sabonete Líquido Facial Antioleosidade / Hidratante", "Gel Limpador Facial", [], "sabonete_facial", "Drogasil"),
        _make_base_entry("Anasol Sérum Facial Vitamina C / Ácido Hialurônico", "Sérum Antioxidante e Hidratante", [], "serum", "Drogasil"),
    ],
    "Adcos": [
        _make_base_entry("Adcos Protetor Solar Stick Tonalizante FPS 80", "Bastão Matte de Alta Cobertura Resistente à Água", ["Ivory", "Peach", "Beige", "Bronze", "Sienna", "Ebony"], "protetor_solar", "Droga Raia", True),
        _make_base_entry("Adcos Protetor Solar Fluid Tonalizante FPS 70 Shield Protection", "Fluido Toque Seco Com Cor", ["Ivory", "Peach", "Beige", "Bronze", "Sienna"], "protetor_solar", "Droga Raia", True),
        _make_base_entry("Adcos Protetor Solar Fluid Máxima Proteção FPS 99 Sem Cor / Com Cor", "Fluido Ultra Protetor para Peles Sensíveis e Melasma", ["Incolor", "Beige", "Bronze"], "protetor_solar", "Droga Raia", True),
        _make_base_entry("Adcos Protetor Solar Aqua Fluid FPS 50 Toque Seco", "Fluido Aquoso Ultraleve Controle de Oleosidade", ["Incolor", "Tonalizante"], "protetor_solar", "Droga Raia", True),
        _make_base_entry("Adcos Derma Complex Vitamina C 20 Sérum", "Sérum Antioxidante Clareador e Anti-Idade", [], "serum", "Droga Raia"),
        _make_base_entry("Adcos Hyalu Water Gel Ácido Hialurônico Hidratante", "Gel Aquoso Oil-Free para Pele Oleosa", [], "hidratante", "Droga Raia"),
        _make_base_entry("Adcos Oily Solution Sabonete Glico-Ativo / Secativo", "Gel de Limpeza Profunda com Ácido Glicólico e Salicílico", [], "sabonete_facial", "Droga Raia"),
        _make_base_entry("Adcos Melan-Off Concentrado Clareador", "Sérum Despigmentante Intensivo para Melasma", [], "tratamento", "Droga Raia"),
    ],
    "Dermage": [
        _make_base_entry("Dermage Photoage Water Color FPS 50 Com Cor", "Fluido Ultraleve Toque Seco Com Cor", ["Claro", "Médio", "Escuro"], "protetor_solar", "Droga Raia", True),
        _make_base_entry("Dermage Photoage Water FPS 50 Sem Cor", "Fluido Aquoso Invisível Não-Comedogênico", [], "protetor_solar", "Droga Raia"),
        _make_base_entry("Dermage Photoage Stick Color FPS 99", "Bastão Fotoprotetor de Alta Cobertura", ["Claro", "Médio", "Moreno", "Escuro"], "protetor_solar", "Droga Raia", True),
        _make_base_entry("Dermage Secatriz Sabonete Líquido Pele Acneica", "Gel Limpador com Ácido Salicílico", [], "sabonete_facial", "Droga Raia"),
        _make_base_entry("Dermage Hyaluage Sérum Ácido Hialurônico", "Sérum Hidratante Preenchedor", [], "serum", "Droga Raia"),
        _make_base_entry("Dermage Clarité TX Sérum Clareador (Ácido Tranexâmico)", "Sérum Clareador de Manchas e Melasma", [], "tratamento", "Droga Raia"),
    ],
    "Darrow (Actine)": [
        _make_base_entry("Darrow Actine Gel de Limpeza Antiacne (Ácido Salicílico 2% + Vitamina C)", "Gel Espumante Refrescante Controlador de Oleosidade", [], "sabonete_facial", "Droga Raia"),
        _make_base_entry("Darrow Actine Sabonete em Barra Pele Acneica", "Barra Limpadora Antioleosidade", [], "sabonete_facial", "Drogasil"),
        _make_base_entry("Darrow Actine Água Micelar Dermatológica Peles Oleosas", "Solução Micelar Purificante Oil-Free", [], "limpeza_micelar", "Droga Raia"),
        _make_base_entry("Darrow Actine Aquaforce Gel Hidratante Matificante (Vitamina C + Ácido Hialurônico)", "Gel Aquoso Refrescante Oil-Free", [], "hidratante", "Droga Raia"),
        _make_base_entry("Darrow Actine Tratamento Gel-Creme Antiacne (Ácido Salicílico + Niacinamida)", "Gel-Creme Corretor de Acne e Poros", [], "tratamento", "Droga Raia"),
        _make_base_entry("Darrow Actine Renov Sérum Antiacne e Antimarcas (Ácido Glicólico + Niacinamida)", "Sérum Renovador Noturno", [], "serum", "Droga Raia"),
        _make_base_entry("Darrow Actine Protetor Solar Fluido Sem Cor FPS 60 Toque Seco", "Fluido Ultraleve Oil Control Incolor", [], "protetor_solar", "Droga Raia"),
        _make_base_entry("Darrow Actine Protetor Solar Colors FPS 70 Com Cor", "Fluido Toque Seco Com Cor Uniformizadora", ["Pele Clara", "Pele Morena", "Pele Morena Mais", "Pele Negra"], "protetor_solar", "Droga Raia", True),
    ],
    "Cetaphil": [
        _make_base_entry("Cetaphil Loção de Limpeza Facial Suave (Pele Sensível e Seca)", "Loção Cremosa Sem Espuma e Sem Sabão (Niacinamida + Pantenol)", [], "sabonete_facial", "Droga Raia"),
        _make_base_entry("Cetaphil Gel de Limpeza Facial Pele Oleosa e Sensível", "Gel Espumante Suave Desobstrutor de Poros", [], "sabonete_facial", "Drogasil"),
        _make_base_entry("Cetaphil Pro AC Control Espuma de Limpeza Pele Acneica", "Espuma Suave com Zinco Calmante", [], "sabonete_facial", "Droga Raia"),
        _make_base_entry("Cetaphil Optimal Hydration Sérum Facial Hidratante 48h", "Sérum Aquoso Calmante com Ácido Hialurônico", [], "serum", "Droga Raia"),
        _make_base_entry("Cetaphil Optimal Hydration Gel-Creme Facial Refrescante", "Gel-Creme Hidratante Sem Perfume", [], "hidratante", "Droga Raia"),
        _make_base_entry("Cetaphil Creme Hidratante Pele Extremamente Seca e Sensível", "Creme Reparador Intensivo de Barreira", [], "hidratante", "Droga Raia"),
        _make_base_entry("Cetaphil Sun Protetor Solar Lipossomal Sem Cor / Com Cor FPS 50+", "Loção Lipossomal Oil Control", ["Incolor", "Cor Universal"], "protetor_solar", "Droga Raia", True),
    ],
    "The Ordinary": [
        _make_base_entry("The Ordinary Niacinamide 10% + Zinc 1% Sérum", "Sérum Aquoso Controlador de Oleosidade e Poros", [], "serum", "Sephora"),
        _make_base_entry("The Ordinary Hyaluronic Acid 2% + B5 (With Ceramides)", "Sérum Hidratante Reparador de Barreira", [], "serum", "Sephora"),
        _make_base_entry("The Ordinary Glycolic Acid 7% Exfoliating Toner", "Tônico Esfoliante Químico Renovador", [], "tonico", "Sephora"),
        _make_base_entry("The Ordinary Alpha Arbutin 2% + HA Sérum Clareador", "Sérum Despigmentante Anti-Manchas", [], "tratamento", "Sephora"),
        _make_base_entry("The Ordinary Azelaic Acid Suspension 10%", "Creme-Gel Uniformizador para Rosácea, Manchas e Acne", [], "tratamento", "Sephora"),
        _make_base_entry("The Ordinary Salicylic Acid 2% Solution", "Sérum Secativo Desobstrutor de Poros", [], "tratamento", "Sephora"),
        _make_base_entry("The Ordinary Natural Moisturizing Factors + HA", "Creme Hidratante Biomimético Sem Fragrância", [], "hidratante", "Sephora"),
        _make_base_entry("The Ordinary Squalane Cleanser Demaquilante e Limpador", "Bálsamo-Creme Limpador com Esqualano", [], "cleansing_oil", "Sephora"),
    ],
    "Nivea": [
        _make_base_entry("Nivea Hidratante Facial em Gel Pepino e Ácido Hialurônico", "Gel Aquoso Transparente 100% Oil-Free", [], "hidratante", "Droga Raia"),
        _make_base_entry("Nivea Creme Facial Antissinais / Nutritivo", "Creme Hidratante Leve de Rápida Absorção", [], "hidratante", "Drogasil"),
        _make_base_entry("Nivea Sabonete em Gel Controle do Brilho / Acne Control", "Gel Espumante Facial Purificante", [], "sabonete_facial", "Droga Raia"),
        _make_base_entry("Nivea Água Micelar Solução de Limpeza 7 em 1 Efeito Matte", "Solução Micelar Demaquilante Sem Enxágue", [], "limpeza_micelar", "Droga Raia"),
        _make_base_entry("Nivea Sun Beauty Expert Facial Toque Seco FPS 50 Sem Cor / Com Cor", "Fluido Fotoprotetor Matificante", ["Incolor", "Médio"], "protetor_solar", "Droga Raia", True),
        _make_base_entry("Nivea Cellular Luminous 630 Antispot Sérum Clareador de Manchas", "Sérum Emulsão Despigmentante Patenteado", [], "tratamento", "Droga Raia"),
        _make_base_entry("Nivea Cellular Luminous 630 Fluido Preventivo FPS 50", "Protetor Hidratante Diurno Antimanchas", [], "protetor_solar", "Droga Raia"),
    ],
    "Garnier": [
        _make_base_entry("Garnier Uniform & Matte Sabonete Limpeza Facial Vitamina C", "Gel Limpador Purificante Antioleosidade", [], "sabonete_facial", "Droga Raia"),
        _make_base_entry("Garnier Água Micelar Tudo em 1 / Vitamina C", "Solução Micelar Demaquilante e Iluminadora", [], "limpeza_micelar", "Drogasil"),
        _make_base_entry("Garnier Uniform & Matte Sérum Booster Vitamina C + Niacinamida + Salicílico", "Sérum Fluido Antimanchas Toque Seco", [], "serum", "Droga Raia"),
        _make_base_entry("Garnier Protetor Hidratante Uniform & Matte FPS 50 Sem Cor / Com Cor", "Loção Fotoprotetora Matificante com Vitamina C", ["Incolor", "Claro", "Médio", "Moreno", "Negra"], "protetor_solar", "Droga Raia", True),
    ],
    "Profuse (Aché)": [
        _make_base_entry("Profuse Puriance Sabonete Líquido Pele Oleosa e Acneica", "Gel Limpador com Ácido Salicílico e Zinco", [], "sabonete_facial", "Droga Raia"),
        _make_base_entry("Profuse Hidradeep Calm Loção Hidratante Pele Sensível", "Loção Calmante com Niacinamida e Manteiga de Karité", [], "hidratante", "Droga Raia"),
        _make_base_entry("Profuse Hidradeep Gel Creme Facial Pele Oleosa", "Gel-Creme Oil-Free Hidratante", [], "hidratante", "Drogasil"),
        _make_base_entry("Profuse Ensolei Extreme Comfort / Aox FPS 60 Sem Cor e Com Cor", "Fluido Fotoprotetor Toque Seco", ["Incolor", "Translucid", "Bronze"], "protetor_solar", "Droga Raia", True),
        _make_base_entry("Profuse Clair Concentré Sérum Clareador", "Sérum Despigmentante para Manchas e Melasma", [], "tratamento", "Droga Raia"),
    ],
    "Hada Labo": [
        _make_base_entry("Hada Labo Gokujyun Oil Cleansing Demaquilante (Ácido Hialurônico)", "Cleansing Oil Japonês com Azeite de Oliva e Ácido Hialurônico", [], "cleansing_oil", "Droga Raia"),
        _make_base_entry("Hada Labo Gokujyun Face Wash Sabonete Facial Hidratante", "Espuma Cremosa Limpadora Sem Fragrância", [], "sabonete_facial", "Droga Raia"),
        _make_base_entry("Hada Labo Gokujyun Lotion Hidratante Aquosa (4 Tipos de Ácido Hialurônico)", "Loção Tônica/Essência Hidratante Japonesa Sem Perfume", [], "hidratante", "Droga Raia"),
        _make_base_entry("Hada Labo Shirojyun Whitening Lotion (Ácido Tranexâmico)", "Loção Aquosa Clareadora de Manchas", [], "tratamento", "Droga Raia"),
    ],
    "COSRX": [
        _make_base_entry("COSRX Low pH Good Morning Gel Cleanser", "Gel de Limpeza Suave com pH Baixo e Melaleuca", [], "sabonete_facial", "Amazon"),
        _make_base_entry("COSRX Advanced Snail 96 Mucin Power Essence", "Essência Reparadora e Hidratante (96% Mucina de Caracol)", [], "serum", "Amazon"),
        _make_base_entry("COSRX BHA Blackhead Power Liquid", "Líquido Desobstrutor de Cravos e Poros (4% Betaína Salicilato)", [], "tratamento", "Amazon"),
        _make_base_entry("COSRX Aloe Soothing Sun Cream SPF 50+", "Creme Fotoprotetor Calmante com Aloe Vera", [], "protetor_solar", "Amazon"),
    ],
    "SkinCeuticals": [
        _make_base_entry("SkinCeuticals Blemish + Age Cleansing Gel Limpeza Intensiva", "Gel Limpador com Ácido Salicílico, Glicólico e LHA", [], "sabonete_facial", "Droga Raia"),
        _make_base_entry("SkinCeuticals Blemish + Age Defense Sérum Antiacne e Anti-Idade", "Sérum Oil-Free com Ácido Dioico, Salicílico e Glicólico", [], "tratamento", "Droga Raia"),
        _make_base_entry("SkinCeuticals C E Ferulic / Silymarin CF Sérum Vitamina C Pura", "Sérum Antioxidante de Alta Potência", [], "serum", "Droga Raia"),
        _make_base_entry("SkinCeuticals H.A. Intensifier / Phyto Corrective Gel", "Sérum Preenchedor e Calmante Botânico Anti-Vermelhidão", [], "serum", "Droga Raia"),
        _make_base_entry("SkinCeuticals Discoloration Defense Sérum Antimanchas (Tranexâmico)", "Sérum Clareador de Melasma e Hiperpigmentação", [], "tratamento", "Droga Raia"),
        _make_base_entry("SkinCeuticals UV Oil Defense FPS 80 Sem Cor / Com Cor", "Gel-Creme Fotoprotetor Antioleosidade", ["Incolor", "Com Cor Adaptável"], "protetor_solar", "Droga Raia", True),
    ],
    "Caudalie": [
        _make_base_entry("Caudalie Vinoperfect Sérum Radiance Antimanchas", "Sérum Leitoso Clareador com Viniferina (Alternativa à Vitamina C)", [], "tratamento", "Sephora"),
        _make_base_entry("Caudalie Vinopure Gel de Limpeza Purificante (Ácido Salicílico Natural)", "Gel Limpador para Pele Acneica e Oleosa", [], "sabonete_facial", "Sephora"),
        _make_base_entry("Caudalie Vinopure Sérum Infusão Anti-Imperfeições", "Sérum Desobstrutor de Poros com Niacinamida e Polifenóis", [], "serum", "Sephora"),
        _make_base_entry("Caudalie VinoHydra Gel Hidratante Ácido Hialurônico & Aloe Vera", "Gel Fresco Hidratante", [], "hidratante", "Sephora"),
    ],
    "Granado / Phebo": [
        _make_base_entry("Granado Sabonete Líquido Facial Enxofre Antioleosidade", "Gel Espumante Regulador de Seborreia e Acne", [], "sabonete_facial", "Droga Raia"),
        _make_base_entry("Granado Gel Secativo Antiacne Enxofre + Ácido Salicílico", "Gel Secativo Rápido de Espinhas", [], "tratamento", "Droga Raia"),
        _make_base_entry("Granado Sabonete Líquido Glicerina Tradicional / Bebê Camomila", "Gel Ultra-Suave Calmante para Pele Sensível", [], "sabonete_facial", "Droga Raia"),
    ],

    # ==========================================================================
    # MARCAS BRASILEIRAS (LINHA COMPLETA: MAQUIAGEM + SKINCARE)
    # ==========================================================================
    "Bruna Tavares": [
        _make_base_entry("Bruna Tavares BT Skin Base Líquida Ultra Fluida", "Fluido Sérum Segunda Pele Acetinado/Semi-Matte", ["F10", "F20", "F30", "L10", "L20", "L30", "M10", "M20", "M30", "T10", "T20", "T30", "E10", "E20", "E30"], "base", "Época Cosméticos"),
        _make_base_entry("Bruna Tavares BT Ton Base Hidratante / Cushion", "Base Cremosa Hidratante", ["F10", "L20", "M20", "T20", "E20"], "base", "Época Cosméticos"),
        _make_base_entry("Bruna Tavares BT Skinplush Corretivo Líquido", "Creme Fluido Elástico Não-Craquelante", ["F1020", "L1020", "M1020", "T1020", "E1020"], "corretivo", "Época Cosméticos"),
        _make_base_entry("Bruna Tavares BT Multicover Corretivo Líquido Alta Cobertura", "Creme Matte Alta Cobertura Longa Duração", ["F10", "L20", "M20", "T20", "E20"], "corretivo", "Época Cosméticos"),
        _make_base_entry("Bruna Tavares BT Plush Blush Cremoso/Veludo Multifuncional", "Mousse Aveludado Matte Sem Partículas de Brilho", [], "blush", "Época Cosméticos"),
        _make_base_entry("Bruna Tavares BT Shimmer Blush Iluminado / Color", "Compacto Acetinado Micronizado", [], "blush", "Época Cosméticos"),
        _make_base_entry("Bruna Tavares BT Contorno Taupe / Choco / BrownSugar", "Pó Compacto Aveludado Esfumável", [], "contorno", "Época Cosméticos"),
        _make_base_entry("Bruna Tavares BT Glowtion Iluminador Compacto/Gel", "Pó-Gel Micronizado Efeito Espelhado Sem Glitter Grosso", [], "iluminador", "Época Cosméticos"),
        _make_base_entry("Bruna Tavares BT Skinpowder Pó Solto Facial Ultrafino", "Pó Solto Micronizado Resistente à Água", [], "maquiagem", "Época Cosméticos"),
        _make_base_entry("Bruna Tavares BTClerb Cleansing Oil Óleo Demaquilante", "Óleo de Limpeza Emulsionável (Cherry Blossom)", [], "cleansing_oil", "Época Cosméticos"),
        _make_base_entry("Bruna Tavares BT Cleanskin Gel de Limpeza Facial", "Gel Limpador Hidratante Suave", [], "sabonete_facial", "Época Cosméticos"),
        _make_base_entry("Bruna Tavares BT Cream / BT Hydra Primer Hidratante", "Creme Hidratante Preparador de Pele com Esqualano", [], "hidratante", "Época Cosméticos"),
    ],
    "Boca Rosa Beauty": [
        _make_base_entry("Boca Rosa Beauty Base Mate Stick Multifuncional FPS 30", "Stick Cremoso de Acabamento Matte Aveludado", ["1 Maria", "2 Antônia", "3 Francisca", "4 Adriana", "5 Ana", "6 Juliana", "7 Márcia", "8 Camila", "9 Fernanda"], "base", "Beleza na Web"),
        _make_base_entry("Boca Rosa Beauty Base Mate Líquida Alta Cobertura", "Fluido Matte Resistente à Água", ["1 Maria", "2 Antônia", "3 Francisca", "4 Adriana", "5 Ana", "6 Juliana", "7 Márcia", "8 Camila", "9 Fernanda"], "base", "Beleza na Web"),
        _make_base_entry("Boca Rosa Beauty Corretivo Líquido Máxima Cobertura", "Creme Matte Alta Cobertura", ["1 Jasmim", "2 Peônia", "3 Orquídea", "4 Lírio", "5 Acácia"], "corretivo", "Beleza na Web"),
        _make_base_entry("Boca Rosa Beauty Stick Cor (Blush Cremoso Multifuncional)", "Bastão Cremoso Esfumável para Bochechas e Lábios", [], "blush", "Beleza na Web"),
        _make_base_entry("Boca Rosa Beauty Paleta de Contorno Copacabana", "Pó Compacto Aveludado de Contorno e Bronzer", [], "contorno", "Beleza na Web"),
        _make_base_entry("Boca Rosa Beauty Paleta de Iluminador OMG", "Pó Iluminador Micronizado Espelhado", [], "iluminador", "Beleza na Web"),
        _make_base_entry("Boca Rosa Beauty Pó Solto Facial / Pó Compacto Blindado", "Pó Selador Ultrafino à Prova D'Água", [], "maquiagem", "Beleza na Web"),
    ],
    "Natura Una": [
        _make_base_entry("Natura Una Base Sérum Nude Me FPS 18", "Sérum Ultraleve Toque Aveludado Segunda Pele", ["10N", "15Q (Claro Quente)", "19F (Claro-Médio Frio)", "20N (Claro-Médio Neutro)", "24N (Médio Neutro)", "32Q (Castanho Quente)", "37Q", "40N (Escuro Neutro)"], "base", "Beleza na Web"),
        _make_base_entry("Natura Una Base Matte Fluida HD", "Fluido Matte de Longa Duração", ["Claro 20", "Claro 24", "Médio 22", "Médio 26", "Castanho 20", "Escuro 35"], "base", "Beleza na Web"),
        _make_base_entry("Natura Una Base Radiante / CC Cream Nude Me FPS 25", "Creme Uniformizador Hidratante", ["Claro 20", "Médio 22", "Castanho 20", "Escuro 35"], "base", "Beleza na Web"),
        _make_base_entry("Natura Una Corretivo Cobertura Extrema 24h", "Fluido Matte Confortável Alta Cobertura", ["Claro 20", "Claro 24", "Médio 22", "Castanho 20", "Escuro 35"], "corretivo", "Beleza na Web"),
        _make_base_entry("Natura Una Blush Intense Me (Matte e Acetinado)", "Pó Compacto Ultrafino Alta Fixação", [], "blush", "Beleza na Web"),
        _make_base_entry("Natura Una Iluminador Marmorizado / Rosé", "Pó Compacto Iluminador Sofisticado", [], "iluminador", "Beleza na Web"),
        _make_base_entry("Natura Chronos Acqua Biohidratante Renovador", "Gel-Acqua Prebiótico com Ácido Hialurônico", [], "hidratante", "Beleza na Web"),
        _make_base_entry("Natura Chronos Protetor Antissinais FPS 70 Toque Seco Com Cor / Sem Cor", "Fluido Multiprotetor Antioleosidade", ["Incolor", "Claro Médio", "Médio Escuro"], "protetor_solar", "Beleza na Web", True),
        _make_base_entry("Natura Chronos Sabonete Gel Limpeza Purificante Antioleosidade", "Gel Limpador com Ácido Salicílico", [], "sabonete_facial", "Beleza na Web"),
        _make_base_entry("Natura Chronos Sérum Intensivo Multiclareador", "Sérum Despigmentante Anti-Manchas", [], "tratamento", "Beleza na Web"),
    ],
    "Mari Maria Makeup": [
        _make_base_entry("Mari Maria Makeup Base Efeito Aveludado Velvet Skin", "Creme Fluido Aveludado Matte Alta Cobertura", ["Nude", "Bege Claro 1", "Bege Claro 2", "Bege Médio", "Amêndoa", "Caramelo", "Canela", "Cacau", "Chocolate"], "base", "Época Cosméticos"),
        _make_base_entry("Mari Maria Makeup Base e Corretivo Cover Up", "Creme Matte Resistente à Água (24 Tons)", ["MM01", "MM02", "MM04", "MM06", "MM07", "MM10", "MM12", "MM14", "MM18", "MM22"], "base", "Época Cosméticos"),
        _make_base_entry("Mari Maria Makeup Corretivo Cover Up Stick / Líquido", "Corretivo Cremoso Alta Cobertura", ["MM01", "MM04", "MM07", "MM12", "MM18"], "corretivo", "Época Cosméticos"),
        _make_base_entry("Mari Maria Makeup Blush Compacto Sunny Cheeks", "Pó Ultrafino Aveludado", [], "blush", "Época Cosméticos"),
        _make_base_entry("Mari Maria Makeup Iluminador em Pó Stella / Solto", "Pó Iluminador Espelhado", [], "iluminador", "Época Cosméticos"),
        _make_base_entry("Mari Maria Makeup Pó Solto Invisible Silk", "Pó Translúcido Selador", [], "maquiagem", "Época Cosméticos"),
        _make_base_entry("Mari Maria Makeup Sérum Facial Hidratante / Primer Ice Glow", "Gel-Sérum Refrescante Preparador de Pele", [], "hidratante", "Época Cosméticos"),
    ],
    "O Boticário (Make B.)": [
        _make_base_entry("Make B. Base Líquida Glycolic TX / Hyaluronic FPS 70", "Fluido Hidratante Protetor Anti-Manchas", ["Cor 110", "Cor 120", "Cor 140", "Cor 160", "Cor 180", "Cor 210", "Cor 240", "Cor 270", "Cor 310"], "base", "Beleza na Web"),
        _make_base_entry("Make B. Base Líquida Mate Salicylic", "Fluido Matte Controlador de Oleosidade (Ácido Salicílico)", ["Cor 110", "Cor 140", "Cor 170", "Cor 200", "Cor 240", "Cor 280"], "base", "Beleza na Web"),
        _make_base_entry("Make B. Base Líquida Retinol H+", "Sérum-Base Renovador Luminoso FPS 80", ["Cor 120", "Cor 160", "Cor 200", "Cor 240", "Cor 280"], "base", "Beleza na Web"),
        _make_base_entry("Make B. Corretivo Líquido Effect / Hyaluronic", "Fluido Elástico Alta Cobertura", ["Cor 120", "Cor 160", "Cor 210", "Cor 260"], "corretivo", "Beleza na Web"),
        _make_base_entry("Make B. Blush Compacto / Líquido", "Textura Aveludada de Longa Duração", [], "blush", "Beleza na Web"),
        _make_base_entry("Botik Ácido Hialurônico Gel Creme Hidratante Facial (O Boticário)", "Gel-Creme Refrescante para Pele Mista a Oleosa", [], "hidratante", "Beleza na Web"),
        _make_base_entry("Botik Vitamina C / Ácido Mandélico Sérum de Alta Potência", "Sérum Facial Concentrado", [], "serum", "Beleza na Web"),
        _make_base_entry("Boti.Sun Protetor Solar Facial Acqua Fluido FPS 70 Sem Cor / Com Cor", "Fluido Ultraleve Invisível", ["Incolor", "Claro/Médio", "Médio/Escuro"], "protetor_solar", "Beleza na Web", True),
    ],
    "Quem Disse, Berenice?": [
        _make_base_entry("Quem Disse, Berenice? Base Líquida Tint Alta Cobertura", "Fluido Matte Confortável", ["01F", "02Q", "03N", "05Q", "07F", "10N", "14Q", "17N", "20Q"], "base", "Beleza na Web"),
        _make_base_entry("Quem Disse, Berenice? Corretivo Líquido Tint Alta Cobertura", "Fluido Matte Longa Duração", ["01", "03", "05", "10", "14"], "corretivo", "Beleza na Web"),
        _make_base_entry("Quem Disse, Berenice? Cleansing Oil Óleo Demaquilante", "Óleo Limpador Facial Emulsionável", [], "cleansing_oil", "Beleza na Web"),
        _make_base_entry("Quem Disse, Berenice? Hidratante Facial Gel / Primer", "Gel HidratantePreparador de Pele", [], "hidratante", "Beleza na Web"),
    ],
    "Eudora (Niina Secrets / Glam)": [
        _make_base_entry("Niina Secrets by Eudora Base Líquida Hidra Glow", "Fluido Hidratante Luminoso (Ácido Hialurônico)", ["Cor 00", "Cor 05", "Cor 10", "Cor 13", "Cor 25", "Cor 35", "Cor 45", "Cor 60", "Cor 70", "Cor 85", "Cor 95"], "base", "Beleza na Web"),
        _make_base_entry("Niina Secrets by Eudora Base Daily Tint Cream FPS 30", "Tint Matte Confortável Tratamento Diário", ["Cor 05", "Cor 10", "Cor 25", "Cor 35", "Cor 50", "Cor 70", "Cor 90"], "base", "Beleza na Web"),
        _make_base_entry("Eudora Glam Base Líquida Skin Perfection Alta Cobertura", "Fluido Matte Fixação Extrema", ["Claro 10", "Bege Claro 20", "Bege Médio 35", "Moreno 55", "Escuro 80"], "base", "Beleza na Web"),
        _make_base_entry("Niina Secrets Corretivo Perfect Match 24h", "Corretivo Cremoso Hidratante", ["Cor 05", "Cor 10", "Cor 25", "Cor 35", "Cor 60"], "corretivo", "Beleza na Web"),
        _make_base_entry("Niina Secrets Blush Cremoso / Compacto Luminous", "Blush Aveludado", [], "blush", "Beleza na Web"),
        _make_base_entry("Niina Secrets Skin Gel de Limpeza / Hidratante Antioleosidade", "Gel Limpador e Hidratante com Niacinamida", [], "sabonete_facial", "Beleza na Web"),
        _make_base_entry("Niina Secrets Skin Cleansing Oil Demaquilante", "Óleo Demaquilante que Não Arde os Olhos", [], "cleansing_oil", "Beleza na Web"),
    ],
    "Fran by Franciny Ehlke": [
        _make_base_entry("Fran by Franciny Ehlke Base Mate Real Filter", "Fluido Matte de Longa Duração Resistente à Água", ["C-01", "C-02", "C-03", "M-01", "M-02", "M-03", "E-01", "E-02", "E-03"], "base", "Beleza na Web"),
        _make_base_entry("Fran by Franciny Ehlke Corretivo Líquido Stick / Real Filter", "Creme Fluido Alta Cobertura", ["C-01", "C-02", "M-01", "M-02", "E-01"], "corretivo", "Beleza na Web"),
        _make_base_entry("Fran by Franciny Ehlke Blush Compacto / Stick Cheek", "Blush Aveludado", [], "blush", "Beleza na Web"),
        _make_base_entry("Fran by Franciny Ehlke Pó Facial Solto Plush", "Pó Ultrafino Selador", [], "maquiagem", "Beleza na Web"),
    ],
    "Karen Bachini Beauty": [
        _make_base_entry("Karen Bachini Base Líquida Hidratante / Matte", "Fluido Segunda Pele Vegano", ["01C", "04N", "07Q", "10N", "14Q", "18N", "22Q", "26N"], "base", "Época Cosméticos"),
        _make_base_entry("Karen Bachini Corretivo Líquido Hidratante", "Corretivo Fluido Vegano", ["01C", "04N", "10N", "18N"], "corretivo", "Época Cosméticos"),
    ],
    "Vizzela": [
        _make_base_entry("Vizzela Base Líquida Matte Vegana Oil-Free", "Fluido Matte Aveludado Não-Comedogênico", ["Cor 01", "Cor 02", "Cor 03", "Cor 05", "Cor 06", "Cor 07", "Cor 10", "Cor 12", "Cor 15", "Cor 18"], "base", "Época Cosméticos"),
        _make_base_entry("Vizzela Base Sérum Gotas Vegana", "Sérum Fluido Leve Acetinado", ["Cor 02", "Cor 05", "Cor 07", "Cor 10", "Cor 14"], "base", "Época Cosméticos"),
        _make_base_entry("Vizzela Corretivo Líquido Vegano / Fix", "Creme Fluido Alta Cobertura Sem Craquelar", ["Cor 01", "Cor 02", "Cor 03", "Cor 05", "Cor 07", "Cor 10"], "corretivo", "Época Cosméticos"),
        _make_base_entry("Vizzela Blush Compacto / Stick Oh My Blush", "Textura Aveludada Vegana", [], "blush", "Época Cosméticos"),
        _make_base_entry("Vizzela Pó Solto Banana / Translúcido", "Pó Micronizado Selador", [], "maquiagem", "Época Cosméticos"),
        _make_base_entry("Vizzela Gel de Limpeza Facial Hialurônico / Vitamina C", "Gel Limpador Suave Vegano", [], "sabonete_facial", "Época Cosméticos"),
        _make_base_entry("Vizzela Creme Hidratante / Sérum Facial", "Hidratante Facial Leve", [], "hidratante", "Época Cosméticos"),
    ],
    "Tracta": [
        _make_base_entry("Tracta Base Líquida Matte Alta / Média Cobertura Oil-Free", "Creme Matte Secagem Rápida", ["01C", "02C", "03", "04", "05", "06", "08", "10"], "base", "Droga Raia"),
        _make_base_entry("Tracta Corretivo Líquido Matte / Radiance", "Fluido Corretivo", [" Claro", "Médio", "Escuro"], "corretivo", "Droga Raia"),
        _make_base_entry("Tracta Sabonete Gel Facial / Água Micelar / Vitamina C", "Gel Limpador e Skincare Facial", [], "sabonete_facial", "Droga Raia"),
    ],
    "Dailus": [
        _make_base_entry("Dailus Base Líquida Ultra Cobertura Matte", "Fluido Matte Hidratante com Ácido Hialurônico", ["D1 (Claro)", "D2 (Claro)", "D4 (Médio)", "D6 (Médio)", "D8 (Moreno)", "D10 (Escuro)"], "base", "Drogasil"),
        _make_base_entry("Dailus Corretivo Líquido Ultra Cobertura", "Fluido Matte Elástico", ["D1", "D2", "D4", "D6", "D8"], "corretivo", "Drogasil"),
        _make_base_entry("Dailus Blush Meteorito / Stick Chubby", "Blush Compacto e Cremoso", [], "blush", "Drogasil"),
    ],
    "Vult": [
        _make_base_entry("Vult Base Líquida Matte Hidraluronic (Ácido Hialurônico)", "Fluido Matte Confortável Não-Comedogênico", ["B15 (Claro)", "B20 (Claro Neutro)", "R20 (Claro Rosado)", "B25 (Médio)", "B30 (Médio Quente)", "M15 (Morena)", "N10 (Negra)"], "base", "Drogasil"),
        _make_base_entry("Vult Base Líquida Efeito Natural / Nano HD", "Fluido Segunda Pele", ["B15", "B20", "B25", "B30", "M15"], "base", "Drogasil"),
        _make_base_entry("Vult Corretivo Líquido Nano HD", "Fluido Matte Alta Cobertura", ["B15", "B25", "R20", "M15", "N10"], "corretivo", "Drogasil"),
        _make_base_entry("Vult Blush Compacto Meu Blush (Matte e Acetinado)", "Pó Compacto Micronizado", [], "blush", "Drogasil"),
        _make_base_entry("Vult Facial Hidratante em Gel Ácido Hialurônico / Vitamina C", "Gel Hidratante Facial Oil-Free", [], "hidratante", "Drogasil"),
        _make_base_entry("Vult Sabonete Gel Limpeza Facial", "Gel Espumante Diário", [], "sabonete_facial", "Drogasil"),
    ],
    "Ruby Rose": [
        _make_base_entry("Ruby Rose Base Líquida Feels Mood", "Fluido Semi-Matte Hidratante", ["Castanha 10", "Castanha 20", "Amêndoa 30", "Caramelo 40", "Cacau 50", "Chocolate 60"], "base", "Época Cosméticos"),
        _make_base_entry("Ruby Rose Base Líquida Soft Matte", "Creme Matte Alta Cobertura", ["Bege 1", "Bege 2", "Bege 3", "Bege 4", "Bege 5", "Chocolate 1", "Chocolate 2", "Chocolate 6"], "base", "Época Cosméticos"),
        _make_base_entry("Ruby Rose Corretivo Líquido Feels Mood / Flawless", "Creme Fluido Alta Cobertura", ["Castanha 10", "Castanha 20", "Amêndoa 30", "Caramelo 40"], "corretivo", "Época Cosméticos"),
        _make_base_entry("Ruby Rose Blush Compacto / Líquido Cheek to Cheek", "Blush Pigmentado", [], "blush", "Época Cosméticos"),
        _make_base_entry("Ruby Rose Sabonete Mousse / Gel de Limpeza Nuvem", "Espuma Facial Demaquilante", [], "sabonete_facial", "Época Cosméticos"),
    ],
    "Océane": [
        _make_base_entry("Océane Edition Foundation Stick Base em Bastão", "Bastão Cremoso Acabamento Natural", ["Ivory", "Vanilla", "Sand", "Beige", "Bronze", "Cocoa"], "base", "Beleza na Web"),
        _make_base_entry("Océane Edition Skin Foundation Base Líquida", "Fluido Semi-Matte", ["100L", "200L", "300M", "400M", "500D"], "base", "Beleza na Web"),
        _make_base_entry("Océane Edition Concealer Corretivo Líquido", "Fluido Hidratante", ["Light", "Medium", "Tan", "Deep"], "corretivo", "Beleza na Web"),
        _make_base_entry("Océane Edition Blush Stick / Compacto Aveludado", "Compacto Ultrafino Acetinado/Matte", [], "blush", "Beleza na Web"),
        _make_base_entry("Océane Edition Contour Stick Contorno Frio-Neutro", "Creme-Pó Aveludado Sem Fundo Alaranjado", [], "contorno", "Beleza na Web"),
        _make_base_entry("Océane Edition Paleta de Sombras 9 Shades / 4 Shades", "Pó Aveludado Alta Fixação", [], "sombra", "Beleza na Web"),
        _make_base_entry("Océane Cleansing Oil Óleo de Limpeza Facial / Balm", "Óleo Demaquilante Coreano", [], "cleansing_oil", "Beleza na Web"),
        _make_base_entry("Océane Gel de Limpeza Facial Vitamina C / Cica", "Gel Espumante Coreano", [], "sabonete_facial", "Beleza na Web"),
        _make_base_entry("Océane Creme Hidratante Facial Cica / Ácido Hialurônico", "Gel-Creme Calmante", [], "hidratante", "Beleza na Web"),
        _make_base_entry("Océane Sérum Facial Niacinamida / Vitamina C", "Sérum Coreano Iluminador", [], "serum", "Beleza na Web"),
    ],
    "Payot": [
        _make_base_entry("Payot Base Matte Alta Cobertura / Retinol", "Fluido Matte Longa Duração", ["Cor 10", "Cor 20", "Cor 30", "Cor 40", "Cor 50", "Cor 60", "Cor 70"], "base", "Droga Raia"),
        _make_base_entry("Payot Corretivo Líquido Matte / Retinol", "Creme Fluido Corretivo", ["10", "20", "30", "40", "50"], "corretivo", "Droga Raia"),
        _make_base_entry("Payot Pó Solto Translúcido / Rosado Retinol", "Pó Micronizado Selador", [], "maquiagem", "Droga Raia"),
        _make_base_entry("Payot Vitamina C Sérum / Hidratante e Sabonete", "Linha Skincare Antioxidante", [], "serum", "Droga Raia"),
    ],
    "Catharine Hill": [
        _make_base_entry("Catharine Hill Base Líquida Angel Wings by Pri Lessa", "Fluido Blindado Resistente à Água Alta Cobertura", ["A1", "A2", "A3", "A4", "A5", "A6"], "base", "Beleza na Web"),
    ],
    "Alice Salazar": [
        _make_base_entry("Alice Salazar Base Líquida Mate", "Fluido Matte Profissional", ["1F", "2Q", "3N", "4Q", "5Q", "6N"], "base", "Época Cosméticos"),
    ],
    "LFpro (Lu Ferraes)": [
        _make_base_entry("LFpro Base Líquida Soft Matte Lu Ferraes", "Fluido Semi-Matte Confortável", ["Cor 01", "Cor 02", "Cor 04", "Cor 05", "Cor 07", "Cor 09"], "base", "Beleza na Web"),
    ],
    "Bauny Cosméticos": [
        _make_base_entry("Bauny Base Matte Líquida Vegana", "Fluido Matte Toque Seco", ["010", "020", "030", "050", "070", "090", "110"], "base", "Drogasil"),
    ],
    "BM Beauty (Bruna Malheiros)": [
        _make_base_entry("BM Beauty Base Sérum Tint / Corretivo", "Sérum Fluido Natural", ["Light 1", "Light 2", "Medium 1", "Medium 2", "Tan 1", "Deep 1"], "base", "Beleza na Web"),
    ],
    "Vic Beauté": [
        _make_base_entry("Vic Beauté Bastão Tudo (Base e Corretivo Stick)", "Stick Cremoso Acetinado Segunda Pele", ["Cor 1", "Cor 2", "Cor 3", "Cor 4", "Cor 5", "Cor 6", "Cor 7", "Cor 8"], "base", "Sephora"),
    ],
    "Simple Organic": [
        _make_base_entry("Simple Organic Base Líquida / BB Cream Natural", "Fluido Orgânico Vegano", ["01", "02", "03", "04", "05", "06"], "base", "Sephora"),
        _make_base_entry("Simple Organic Solução Niacinamida / Vitamina C / Ácido Glicólico", "Sérum Concentrado Clean Beauty Orgânico", [], "serum", "Sephora"),
        _make_base_entry("Simple Organic Gel de Limpeza Facial GABA / Green Water", "Gel Espumante Orgânico Suave", [], "sabonete_facial", "Sephora"),
        _make_base_entry("Simple Organic Hidratante Facial Ácido Hialurônico / CBA", "Emulsão Hidratante Vegana", [], "hidratante", "Sephora"),
        _make_base_entry("Simple Organic Protetor Solar Facial Mineral Daily Drops FPS 30", "Fluido Mineral Clean Beauty", [], "protetor_solar", "Sephora"),
    ],
    "Care Natural Beauty": [
        _make_base_entry("Care Natural Beauty Base Sérum Clean Beauty", "Sérum Hidratante Natural", ["Light 01", "Light 02", "Medium 01", "Medium 02", "Tan 01", "Deep 01"], "base", "Sephora"),
        _make_base_entry("Care Natural Beauty Duo Blush & Iluminador Stick", "Bastão Cremoso Clean Beauty", [], "blush", "Sephora"),
        _make_base_entry("Care Natural Beauty Sérum Skindrops Hialu / Glow", "Sérum Hidratante Antioxidante", [], "serum", "Sephora"),
        _make_base_entry("Care Natural Beauty Bruma HidratanteCare Mist", "Bruma Tonificante Calmante", [], "tonico", "Sephora"),
    ],
    "Contém 1g": [
        _make_base_entry("Contém 1g Base Líquida Velvet Segunda Pele", "Fluido Aveludado", ["Claro 02", "Claro 04", "Médio 06", "Médio 08", "Moreno 10", "Escuro 12"], "base", "Beleza na Web"),
        _make_base_entry("Contém 1g Primer Facial Efeito Blur / Glow", "Primer Aveludado Disfarce de Poros", [], "maquiagem", "Beleza na Web"),
        _make_base_entry("Contém 1g Blush Compacto / Cremoso", "Textura Micronizada", [], "blush", "Beleza na Web"),
    ],
    "Avon": [
        _make_base_entry("Avon Power Stay Base Líquida 24 Horas Matte", "Fluido Matte Intransferível 24h", ["110F", "120N", "140Q", "210Q", "220N", "240N", "310Q", "340Q", "410N", "430F"], "base", "Amazon"),
        _make_base_entry("Avon Base Sérum FPS 30 Tratamake", "Sérum-Base Radiante com Ativos de Skincare", ["110F", "140Q", "220N", "310Q", "410N"], "base", "Amazon"),
        _make_base_entry("Avon Power Stay Corretivo Líquido 18 Horas", "Corretivo Fluido Confortável", ["120N", "210Q", "240N", "310Q", "410N"], "corretivo", "Amazon"),
        _make_base_entry("Avon Renew Protetor Solar Matte Antioleosidade FPS 50 / FPS 70", "Loção Toque Seco Antissinais Com e Sem Cor", ["Incolor", "Claro", "Médio", "Moreno"], "protetor_solar", "Amazon", True),
        _make_base_entry("Avon Renew Vitamina C / Power Sérum Renovador", "Sérum Concentrado Antioxidante e Pró-Colágeno", [], "serum", "Amazon"),
        _make_base_entry("Avon Renew Gel de Limpeza Facial / Água Micelar", "Gel Limpador com Extrato de Carvão / Ácido Hialurônico", [], "sabonete_facial", "Amazon"),
        _make_base_entry("Avon Renew Hydra Pro Vita-D Gel-Creme Hidratante", "Gel Aquoso Hidratante 72h", [], "hidratante", "Amazon"),
    ],
    "Mary Kay": [
        _make_base_entry("Mary Kay Base Líquida TimeWise 3D Matte", "Fluido Matte Controle de Oleosidade 12h", ["Ivory W130", "Ivory N140", "Ivory N160", "Beige W100", "Beige C110", "Beige N150", "Beige W180", "Beige N190", "Beige N210", "Bronze W100", "Bronze W120", "Bronze W140"], "base", "Amazon"),
        _make_base_entry("Mary Kay Corretivo Perfecting Concealer / Yellow", "Creme Corretivo Alta Cobertura Camuflagem", ["Light Ivory", "Deep Ivory", "Light Beige", "Deep Beige", "Light Bronze", "Yellow"], "corretivo", "Amazon"),
        _make_base_entry("Mary Kay Gel de Limpeza 4 em 1 TimeWise 3D", "Gel Limpador Facial Diário", [], "sabonete_facial", "Amazon"),
        _make_base_entry("Mary Kay Hidratante Antioxidante TimeWise 3D (Pele Mista/Oleosa)", "Loção Hidratante Oil-Free", [], "hidratante", "Amazon"),
    ],

    # ==========================================================================
    # MARCAS NORTE-AMERICANAS (EUA) & INTERNACIONAIS (LINHA COMPLETA)
    # ==========================================================================
    "M·A·C Cosmetics": [
        _make_base_entry("M·A·C Studio Fix Fluid SPF 15 24HR Matte Foundation", "Fluido Matte Confortável Controlador de Oleosidade", ["NC10", "NC15", "NC20", "NC25", "NC30", "NC35", "NC37", "NC40", "NC42", "NC45", "NC50", "NW15", "NW20", "NW25", "NW35", "NW45"], "base", "Sephora"),
        _make_base_entry("M·A·C Studio Radiance Serum-Powered Foundation", "Base Sérum Hidratante Luminosa", ["NC15", "NC20", "NC25", "NC30", "NC40", "NC45", "NW20", "NW25"], "base", "Sephora"),
        _make_base_entry("M·A·C Studio Fix Powder Plus Foundation (Base em Pó)", "Pó Compacto Base Matte Aveludado", ["NC15", "NC20", "NC25", "NC30", "NC35", "NC40", "NC42", "NC45", "NW20", "NW25"], "base", "Sephora"),
        _make_base_entry("M·A·C Studio Fix 24-Hour Smooth Wear Concealer Corretivo", "Corretivo Fluido Matte Elástico", ["NC20", "NC25", "NC30", "NC42", "NW20", "NW25"], "corretivo", "Sephora"),
        _make_base_entry("M·A·C Pro Longwear Concealer Corretivo", "Corretivo Fluido Intransferível", ["NC15", "NC20", "NC25", "NC30", "NC35", "NC42", "NW20", "NW25"], "corretivo", "Sephora"),
        _make_base_entry("M·A·C Mineralize Blush / Powder Blush / Glow Play", "Blush Compacto e Bouncy", [], "blush", "Sephora"),
        _make_base_entry("M·A·C Mineralize Skinfinish Iluminador (Soft & Gentle / Global Glow)", "Pó Iluminador Assado Micronizado", [], "iluminador", "Sephora"),
        _make_base_entry("M·A·C Hyper Real Serumizer / Canvas Balm Hidratante", "Sérum-Hidratante Reparador de Barreira com Niacinamida e Ceramidas", [], "hidratante", "Sephora"),
        _make_base_entry("M·A·C Hyper Real Fresh Canvas Cleansing Oil", "Óleo Demaquilante Leve que Não Arde os Olhos", [], "cleansing_oil", "Sephora"),
        _make_base_entry("M·A·C Prep + Prime Fix+ Bruma Hidratante Fixadora", "Bruma Calmante com Camomila e Chá Verde", [], "tonico", "Sephora"),
    ],
    "Fenty Beauty": [
        _make_base_entry("Fenty Beauty Pro Filt'r Soft Matte Longwear Foundation", "Fluido Soft Matte Resistente ao Suor (50 Tons)", ["100 (Claro Neutro)", "120 (Claro Neutro)", "150 (Claro Neutro)", "185 (Claro-Médio Neutro)", "190 (Claro-Médio Quente)", "210 (Médio Neutro)", "240 (Médio Quente)", "290 (Médio Oliva)", "300 (Moreno Quente)", "330 (Moreno Dourado)", "385 (Moreno Escuro)", "420 (Negro Quente)", "450 (Negro Profundo)"], "base", "Sephora"),
        _make_base_entry("Fenty Beauty Eaze Drop Blurring Skin Tint", "Skin Tint Fluido Difusor de Poros", ["1", "2", "3", "5", "7", "9", "11", "13", "16", "19", "21", "24"], "base", "Sephora"),
        _make_base_entry("Fenty Beauty Pro Filt'r Instant Retouch Concealer", "Corretivo Cremoso Soft Matte", ["120", "150", "190", "240", "290", "330", "420"], "corretivo", "Sephora"),
        _make_base_entry("Fenty Beauty We're Even Hydrating Longwear Concealer", "Corretivo Hidratante Anti-Bolsas", ["150C", "190W", "240N", "290W", "330W", "420W"], "corretivo", "Sephora"),
        _make_base_entry("Fenty Beauty Match Stix Contour Skinstick Contorno", "Bastão Matte de Contorno (Amber / Mocha / Suedish)", [], "contorno", "Sephora"),
        _make_base_entry("Fenty Beauty Cheeks Out Freestyle Cream Blush", "Blush Cremoso Natural", [], "blush", "Sephora"),
        _make_base_entry("Fenty Beauty Diamond Bomb / Killawatt Freestyle Highlighter", "Iluminador Gel-Pó Espelhado", [], "iluminador", "Sephora"),
        _make_base_entry("Fenty Skin Total Cleans'r Remove-It-All Cleanser", "Gel-Creme Demaquilante e Limpador 2 em 1", [], "sabonete_facial", "Sephora"),
        _make_base_entry("Fenty Skin Fat Water Niacinamide Pore-Refining Toner Serum", "Tônico-Sérum Encorpado com Niacinamida", [], "tonico", "Sephora"),
        _make_base_entry("Fenty Skin Hydra Vizor Invisible Moisturizer SPF 30", "Hidratante Fotoprotetor Invisível com Niacinamida", [], "protetor_solar", "Sephora"),
    ],
    "Rare Beauty": [
        _make_base_entry("Rare Beauty Liquid Touch Weightless Foundation", "Sérum-Fluido Segunda Pele Respirável", ["100W", "110N", "130N", "140C", "170W", "190W", "210N", "230N (Médio Neutro-Oliva)", "260N", "280N", "320W", "360W", "410N", "460W", "520W"], "base", "Sephora"),
        _make_base_entry("Rare Beauty Positive Light Tinted Moisturizer SPF 20", "Hidratante Com Cor Luminoso", ["10N", "14W", "20W", "24N", "26W", "32N", "38W", "44N"], "base", "Sephora"),
        _make_base_entry("Rare Beauty Liquid Touch Brightening Concealer", "Corretivo Hidratante Iluminador", ["110N", "170W", "210N", "230N", "280N", "360W", "460W"], "corretivo", "Sephora"),
        _make_base_entry("Rare Beauty Soft Pinch Liquid Blush (Matte & Dewy)", "Blush Líquido Ultra Pigmentado (Hope, Happy, Bliss, Encourage, Worth)", [], "blush", "Sephora"),
        _make_base_entry("Rare Beauty Soft Pinch Luminous Powder Blush", "Blush Iluminado Aveludado", [], "blush", "Sephora"),
        _make_base_entry("Rare Beauty Warm Wishes Effortless Bronzer Stick", "Bastão de Contorno/Bronzer Cremoso Segunda Pele", [], "contorno", "Sephora"),
        _make_base_entry("Rare Beauty Positive Light Liquid / Silky Touch Highlighter", "Iluminador Líquido e Compacto Efeito Vidro", [], "iluminador", "Sephora"),
        _make_base_entry("Rare Beauty Always an Optimist 4-in-1 Prime & Set Mist", "Bruma Hidratante e Fixadora", [], "tonico", "Sephora"),
    ],
    "NARS": [
        _make_base_entry("NARS Light Reflecting Foundation Base Líquida", "Fluido Acetinado de Alta Performance Skincare", ["Oslo (Frio)", "Mont Blanc (Claro Neutro)", "Gobi (Claro Amarelado)", "Deauville (Claro-Médio)", "Fiji (Claro-Médio Quente)", "Punjab (Médio Dourado)", "Vallauris (Médio Rosado)", "Santa Fe (Médio Neutro)", "Stromboli (Médio Oliva)", "Barcelona (Médio-Escuro)", "Syracuse (Morena)", "Tahoe (Morena Escura)", "Macao (Negra)", "New Caledonia (Negra Profunda)"], "base", "Sephora"),
        _make_base_entry("NARS Natural Radiant Longwear Foundation", "Fluido Radiante de Alta Cobertura 16h", ["Mont Blanc", "Gobi", "Deauville", "Fiji", "Punjab", "Stromboli", "Barcelona", "Syracuse", "Tahoe", "Macao"], "base", "Sephora"),
        _make_base_entry("NARS Soft Matte Complete Foundation", "Base Fluida Matte Controle Total de Oleosidade", ["Mont Blanc", "Gobi", "Deauville", "Fiji", "Punjab", "Stromboli", "Syracuse", "Tahoe", "Macao"], "base", "Sephora"),
        _make_base_entry("NARS Pure Radiant Tinted Moisturizer SPF 30", "Hidratante Com Cor Luminoso FPS 30", ["Terre Neuve", "Finland", "Alaska", "St. Moritz", "Groenland", "Mykonos", "Marrakech"], "base", "Sephora", True),
        _make_base_entry("NARS Radiant Creamy Concealer Corretivo Líquido", "Creme Líquido Acetinado de Alta Cobertura", ["Chantilly (Muito Claro)", "Vanilla (Claro Rosado)", "Custard (Médio Neutro-Quente)", "Ginger (Médio Dourado)", "Biscuit (Médio-Escuro)", "Caramel (Morena)", "Amande (Morena Escura)", "Cacao (Negra)"], "corretivo", "Sephora"),
        _make_base_entry("NARS Soft Matte Complete Concealer (Corretivo de Pote)", "Bálsamo Matte Camuflagem de Acne e Olheiras", ["Chantilly", "Vanilla", "Custard", "Ginger", "Biscuit", "Caramel", "Cacao"], "corretivo", "Sephora"),
        _make_base_entry("NARS Blush Compacto / Afterglow Liquid Blush (Orgasm, Dolce Vita)", "Blush Compacto e Líquido", [], "blush", "Sephora"),
        _make_base_entry("NARS Laguna Bronzing Powder / CreamContorno", "Bronzer e Contorno Icônico", [], "contorno", "Sephora"),
    ],
    "Dragun Beauty": [
        _make_base_entry("Dragun Beauty DragunWing / Transformation Skin Tint & Concealer", "Creme Fluido de Alta Cobertura Transformadora (Nikita Dragun)", ["Fair 10 (Claro Frio)", "Light 20 (Claro Quente)", "Light-Medium 30 (Claro-Médio Neutro)", "Medium 40 (Médio Dourado)", "Tan 50 (Moreno Quente)", "Deep 60 (Pele Negra)"], "base", "Amazon"),
        _make_base_entry("Dragun Beauty DragunHeart Transformation Face Powder", "Pó Compacto Iluminador e Corretor de Subtom", ["Translucent Pink", "Yellow Correcting", "Universal Duo"], "corretivo", "Amazon"),
    ],
    "Haus Labs by Lady Gaga": [
        _make_base_entry("Haus Labs Triclone Skin Tech Medium Coverage Foundation", "Sérum-Base Fermentado com Arnica Calmante", ["040 Fair Neutral", "100 Light Neutral", "110 Light Warm", "145 Light Cool", "200 Light Medium Neutral", "210 Light Medium Warm", "260 Medium Neutral", "300 Medium Warm", "370 Medium Deep", "440 Deep Neutral", "510 Deep Warm"], "base", "Sephora"),
        _make_base_entry("Haus Labs Triclone Skin Tech Hydrating Concealer", "Corretivo Sérum Descongestionante", ["04 Fair", "11 Light Neutral", "21 Light Medium", "25 Medium", "32 Medium Tan", "41 Deep"], "corretivo", "Sephora"),
        _make_base_entry("Haus Labs Color Fuse Talc-Free Powder Blush", "Blush Compacto Sem Talco com Arnica Fermentada", [], "blush", "Sephora"),
        _make_base_entry("Haus Labs Bio-Radiant Gel-Powder Highlighter", "Iluminador Gel-Pó Espelhado", [], "iluminador", "Sephora"),
    ],
    "Huda Beauty": [
        _make_base_entry("Huda Beauty Easy Blur Natural Airbrush Foundation", "Sérum-Base Efeito Filtro Blur com Niacinamida", ["100B Milkshake", "140G Cashew", "200B Shortbread", "210B Chai", "240N Toasted Coconut", "300N Latte", "320G Tres Leches", "410G Brown Sugar", "500G Mocha"], "base", "Sephora"),
        _make_base_entry("Huda Beauty #FauxFilter Luminous Matte Foundation", "Creme Fluido Alta Cobertura Matte Luminoso", ["110N Angel Food", "140G Cashew", "210B Chai", "240N Toasted Coconut", "300N Latte", "320G Tres Leches", "410G Brown Sugar", "500G Mocha"], "base", "Sephora"),
        _make_base_entry("Huda Beauty #FauxFilter Luminous Matte Concealer", "Corretivo Fluido Matte Flexível", ["Sugar Biscuit", "Nougat", "Coconut Flakes", "Cotton Candy", "Granola", "Praline"], "corretivo", "Sephora"),
        _make_base_entry("Huda Beauty Blush Filter Liquid Blush", "Blush Líquido Efeito Blur", [], "blush", "Sephora"),
        _make_base_entry("Huda Beauty Easy Bake Loose Baking & Setting Powder", "Pó Solto Ultrafino Disfarce de Poros", [], "maquiagem", "Sephora"),
    ],
    "Too Faced": [
        _make_base_entry("Too Faced Born This Way Undetectable Medium-to-Full Foundation", "Fluido Hidratante com Água de Coco Segunda Pele", ["Swan (Muito Claro)", "Alabaster (Claro Frio)", "Ivory (Claro Quente)", "Vanilla (Claro Neutro)", "Light Beige (Claro-Médio)", "Natural Beige (Médio Neutro)", "Warm Beige (Médio Quente)", "Golden Beige (Médio Dourado)", "Sand (Médio Bronze)", "Caramel (Morena)", "Mocha (Morena Escura)", "Chai (Negra)"], "base", "Sephora"),
        _make_base_entry("Too Faced Born This Way Soft Matte Foundation", "Fluido Matte Oil-Free 24h", ["Swan", "Vanilla", "Light Beige", "Natural Beige", "Warm Beige", "Sand", "Mocha"], "base", "Sephora"),
        _make_base_entry("Too Faced Born This Way Super Coverage Multi-Use Concealer", "Corretivo Hidratante Alta Cobertura", ["Swan", "Vanilla", "Light Beige", "Natural Beige", "Warm Beige", "Sand", "Caramel", "Mocha"], "corretivo", "Sephora"),
        _make_base_entry("Too Faced Cloud Crush Blurring Blush", "Blush Compacto Efeito Nuvem", [], "blush", "Sephora"),
        _make_base_entry("Too Faced Hangover Replenishing Face Primer / Pillow Balm", "Primer Hidratante com Água de Coco e Probióticos", [], "hidratante", "Sephora"),
    ],
    "Estée Lauder": [
        _make_base_entry("Estée Lauder Double Wear Stay-in-Place Makeup SPF 10", "Fluido Matte Lendário Longa Duração 24h Oil-Free", ["1N1 Ivory Nude", "1W1 Bone", "1W2 Sand", "2N1 Desert Beige", "2W1 Dawn", "2W2 Rattan (Médio Quente)", "3N1 Ivory Beige", "3W1 Tawny (Médio Dourado)", "4N1 Shell Beige", "4W1 Honey Bronze", "5W1 Bronze", "6W1 Sandalwood"], "base", "Sephora"),
        _make_base_entry("Estée Lauder Futurist Hydra Rescue Moisturizing Makeup SPF 45", "Base Hidratante Luminosa com Probióticos", ["1N1", "1W2", "2N1", "2W1", "3W1", "4N1"], "base", "Sephora", True),
        _make_base_entry("Estée Lauder Double Wear Stay-in-Place Concealer", "Corretivo Matte 24h", ["1N", "2C", "2W", "3W", "4N"], "corretivo", "Sephora"),
        _make_base_entry("Estée Lauder Advanced Night Repair Synchronized Multi-Recovery Serum", "Sérum Reparador Noturno com Peptídeos e Ácido Hialurônico", [], "serum", "Sephora"),
    ],
    "Tarte Cosmetics": [
        _make_base_entry("Tarte Shape Tape Full Coverage Concealer", "Corretivo Matte de Altíssima Cobertura", ["12N Fair Neutral", "16N Fair-Light Neutral", "20B Light", "22N Light Neutral", "27S Light-Medium Sand", "29N Light-Medium", "35N Medium", "36S Medium-Tan Sand", "42S Tan Sand", "53N Deep"], "corretivo", "Sephora"),
        _make_base_entry("Tarte Shape Tape Ultra Creamy Concealer", "Corretivo Hidratante com Ácido Hialurônico", ["16N", "22N", "27S", "29N", "35N", "42S"], "corretivo", "Sephora"),
        _make_base_entry("Tarte Face Tape / Amazonian Clay 16HR Full Coverage Foundation", "Mousse Matte com Argila Amazônica", ["16N Fair-Light", "22N Light Neutral", "29N Light-Medium", "35N Medium", "42S Tan Sand", "53N Deep"], "base", "Sephora"),
        _make_base_entry("Tarte Amazonian Clay 12-Hour Blush", "Blush Compacto de Longa Duração", [], "blush", "Sephora"),
    ],
    "Charlotte Tilbury": [
        _make_base_entry("Charlotte Tilbury Airbrush Flawless Foundation", "Fluido Matte Natural Efeito Airbrush", ["1 Neutral", "2 Neutral", "3 Warm", "4 Neutral", "5.5 Neutral (Médio)", "6 Warm", "7.5 Neutral (Médio-Bronze)", "9 Warm (Morena)", "12 Neutral (Morena Escura)", "15 Neutral (Negra)"], "base", "Amazon"),
        _make_base_entry("Charlotte Tilbury Beautiful Skin Medium Coverage Foundation", "Creme Hidratante Radiante com Ácido Hialurônico", ["2 Neutral", "4 Neutral", "6 Warm", "8 Neutral", "10 Warm", "14 Neutral"], "base", "Amazon"),
        _make_base_entry("Charlotte Tilbury Hollywood Flawless Filter", "Booster Iluminador e Aperfeiçoador de Pele", ["1 Fair", "2 Fair", "3 Fair/Medium", "4 Medium", "4.5 Medium", "5.5 Tan", "6.5 Deep"], "base", "Amazon"),
        _make_base_entry("Charlotte Tilbury Beauty Light Wand Iluminador / Blush (Pinkgasm / Spotlight)", "Iluminador e Blush Líquido Cushion", [], "iluminador", "Amazon"),
        _make_base_entry("Charlotte Tilbury Charlotte's Magic Cream / Magic Water Cream", "Creme e Gel-Creme Hidratante Preparador de Pele", [], "hidratante", "Amazon"),
    ],
    "Maybelline NY": [
        _make_base_entry("Maybelline Fit Me Base Líquida Matte + Poreless", "Fluido Matte Disfarce Óptico de Poros", ["100 Warm Ivory", "110 Porcelain", "112 Natural Ivory", "115 Ivory", "120 Classic Ivory", "125 Nude Beige", "128 Warm Nude", "220 Natural Beige", "230 Natural Buff", "310 Sun Beige", "322 Warm Honey", "330 Toffee", "355 Coconut"], "base", "Droga Raia"),
        _make_base_entry("Maybelline SuperStay Active Wear 30H / Lumi-Matte Foundation", "Fluido Matte de Longa Duração 30h", ["110", "120", "128", "220", "310", "312", "355"], "base", "Drogasil"),
        _make_base_entry("Maybelline SuperStay 24H Skin Tint (Com Vitamina C)", "Sérum Com Cor Luminoso de Longa Duração", ["Cor 02", "Cor 05", "Cor 10", "Cor 20", "Cor 30", "Cor 48"], "base", "Droga Raia"),
        _make_base_entry("Maybelline Corretivo Líquido Instant Age Rewind Eraser", "Fluido Leve Elástico Não-Craquelante", ["Fair (110)", "Light (120)", "Medium (130)", "Honey (140)", "Caramel (144)"], "corretivo", "Droga Raia"),
        _make_base_entry("Maybelline Fit Me Corretivo Líquido", "Fluido Natural Sem Óleo", ["10 Claro", "20 Médio", "30 Escuro"], "corretivo", "Drogasil"),
        _make_base_entry("Maybelline Fit Me Blush Compacto / Pó Compacto Matte", "Pó Micronizado Controlador de Brilho", [], "blush", "Droga Raia"),
    ],
    "L'Oréal Paris": [
        _make_base_entry("L'Oréal Paris True Match Super-Blendable Foundation", "Sérum-Base com Ácido Hialurônico (Subtons W, N, C)", ["N1", "N2", "W2", "W3 (Claro-Médio Quente)", "N4", "W4.5 (Médio Quente)", "W5", "N6", "W7 (Morena)", "W8", "N9"], "base", "Amazon"),
        _make_base_entry("L'Oréal Paris Infallible 24HR Fresh Wear Foundation", "Fluido Respirável de Longa Duração", ["400 Pearl", "420 True Beige", "445 Vanilla", "465 Sand", "485 Golden Amber", "515 Copper"], "base", "Amazon"),
        _make_base_entry("L'Oréal Paris Solar Expertise Antioleosidade Com Cor FPS 60", "Fluido Gel-Creme Matte Antibrilho Com Cor", ["Clara", "Média", "Morena", "Escura"], "protetor_solar", "Drogaria São Paulo", True),
        _make_base_entry("L'Oréal Paris Solar Expertise Antioleosidade Sem Cor FPS 60", "Gel-Creme Toque Seco com Ácido Salicílico", [], "protetor_solar", "Droga Raia"),
        _make_base_entry("L'Oréal Paris Solar Expertise UV Defender Fluido Invisível FPS 60", "Fluido Ultraleve com Ácido Hialurônico", ["Incolor", "Claro", "Médio", "Moreno"], "protetor_solar", "Droga Raia", True),
        _make_base_entry("L'Oréal Paris Revitalift Hialurônico Gel de Limpeza Facial", "Gel Limpador Hidratante Preenchedor", [], "sabonete_facial", "Drogasil"),
        _make_base_entry("L'Oréal Paris Revitalift Hialurônico Água Micelar 5 em 1", "Solução Micelar Preenchedora Sem Óleo", [], "limpeza_micelar", "Droga Raia"),
        _make_base_entry("L'Oréal Paris Revitalift Hialurônico Sérum Preenchedor 1,5%", "Sérum Aquoso Hidratante Anti-Linhas", [], "serum", "Droga Raia"),
        _make_base_entry("L'Oréal Paris Revitalift Hialurônico Gel-Creme Hidratante Antissinais", "Gel-Creme Refrescante", [], "hidratante", "Droga Raia"),
        _make_base_entry("L'Oréal Paris Glycolic Bright Sérum / Gel de Limpeza Antimanchas", "Linha Renovadora com Ácido Glicólico e Niacinamida", [], "tratamento", "Droga Raia"),
        _make_base_entry("L'Oréal Paris Revitalift Retinol Puro Sérum Noturno", "Sérum Anti-Idade Renovador", [], "tratamento", "Droga Raia"),
    ],
    "Revlon": [
        _make_base_entry("Revlon ColorStay Longwear Makeup for Combination/Oily Skin", "Fluido Matte Controle de Oleosidade 24h", ["110 Ivory", "150 Buff (Claro Neutro-Oliva)", "180 Sand Beige", "220 Natural Beige", "240 Medium Beige", "300 Golden Beige", "330 Natural Tan", "370 Toast", "400 Caramel", "450 Mocha"], "base", "Droga Raia"),
    ],
    "e.l.f. Cosmetics": [
        _make_base_entry("e.l.f. Halo Glow Liquid Filter / Flawless Satin Foundation", "Fluido Glow / Acetinado Uniformizador", ["1 Fair", "2 Fair/Light", "3 Light/Medium", "4 Medium", "5 Medium/Tan", "6 Tan/Deep", "7 Deep", "8 Rich"], "base", "Amazon"),
        _make_base_entry("e.l.f. Hydrating Camo Concealer Corretivo", "Corretivo Líquido Alta Cobertura Hidratante", ["Fair Warm", "Light Sand", "Light Beige", "Medium Sand", "Medium Peach", "Tan Sand", "Deep Caramel"], "corretivo", "Amazon"),
    ],
    "NYX Professional Makeup": [
        _make_base_entry("NYX Can't Stop Won't Stop Full Coverage Foundation", "Fluido Matte À Prova D'Água 24h", ["Pale", "Fair", "Alabaster", "Light Ivory", "Vanilla", "Natural", "True Beige", "Medium Olive", "Golden", "Caramel", "Mocha", "Walnut"], "base", "Amazon"),
        _make_base_entry("NYX Bare With Me Blur Tint / Concealer Serum", "Sérum Corretivo e Skin Tint", ["Fair", "Light", "Vanilla", "Beige", "Golden", "Tan", "Deep"], "base", "Amazon"),
    ],
    "Milani": [
        _make_base_entry("Milani Conceal + Perfect 2-in-1 Foundation + Concealer", "Creme Fluido 2 em 1 de Alta Cobertura", ["00A Porcelain", "01 Creamy Vanilla", "02 Natural", "03 Light Beige", "04 Medium Beige", "05 Warm Beige", "06 Sand Beige", "09 Tan", "11 Amber", "14 Chestnut"], "base", "Amazon"),
    ],
    "CoverGirl": [
        _make_base_entry("CoverGirl TruBlend Matte Made / Clean Fresh Skin Milk", "Fluido Confortável", ["L10", "L20", "L40", "M10", "M30", "M60", "T20", "D10", "D50"], "base", "Amazon"),
    ],
    "Wet n Wild": [
        _make_base_entry("Wet n Wild Photo Focus Foundation (Matte / Dewy)", "Fluido Testado contra Flash Fotográfico", ["Porcelain", "Soft Ivory", "Nude Ivory", "Soft Beige", "Golden Beige", "Desert Beige", "Caramel", "Cocoa"], "base", "Amazon"),
    ],
    "ColourPop": [
        _make_base_entry("ColourPop Pretty Fresh Hyaluronic Hydrating Foundation", "Fluido Hidratante com Ácido Hialurônico", ["Fair 20N", "Light 45W", "Light 60N", "Medium 85N", "Medium 100W", "Medium Dark 120W", "Dark 165N", "Deep Dark 190W"], "base", "Amazon"),
    ],
    "Anastasia Beverly Hills": [
        _make_base_entry("Anastasia Beverly Hills Luminous Foundation", "Fluido Luminoso Resistente à Água", ["120W", "140N", "210N", "240W", "300C", "332C", "355N", "430W", "510W"], "base", "Sephora"),
    ],
    "IT Cosmetics": [
        _make_base_entry("IT Cosmetics Your Skin But Better CC+ Cream SPF 50+", "CC Cream Tratamento de Cobertura Total + FPS 50", ["Fair", "Fair Light", "Light", "Light Medium", "Medium", "Medium Tan", "Neutral Tan", "Rich", "Deep"], "base", "Sephora"),
    ],
    "Urban Decay": [
        _make_base_entry("Urban Decay Face Bond Self-Setting Waterproof Foundation / Stay Naked", "Fluido Auto-Selante À Prova D'Água", ["20NN", "30NN", "30WY", "40NN", "40WY", "50NN", "60NN", "70NN", "80NN"], "base", "Sephora"),
    ],
    "Clinique": [
        _make_base_entry("Clinique Even Better Clinical Serum Foundation SPF 25", "Base Sérum Sem Fragrância Dermatológica", ["WN 04 Bone", "CN 10 Alabaster", "CN 28 Ivory", "WN 38 Stone", "WN 46 Golden Neutral", "WN 56 Cashew", "WN 76 Toasted Wheat", "WN 98 Cream Caramel", "WN 114 Golden"], "base", "Sephora"),
        _make_base_entry("Clinique Moisture Surge 100H Auto-Replenishing Hydrator", "Gel-Creme Aquoso Oil-Free com Biofermento de Aloe e Ácido Hialurônico", [], "hidratante", "Sephora"),
        _make_base_entry("Clinique Dramatically Different Moisturizing Gel (Pele Mista a Oleosa)", "Gel Hidratante Fortalecedor de Barreira Sem Perfume", [], "hidratante", "Sephora"),
        _make_base_entry("Clinique Take The Day Off Cleansing Balm Demaquilante", "Bálsamo Demaquilante que vira Óleo Sedoso Sem Fragrância", [], "cleansing_oil", "Sephora"),
        _make_base_entry("Clinique All About Clean Liquid Facial Soap (Oily / Mild)", "Sabonete Líquido Dermatológico", [], "sabonete_facial", "Sephora"),
        _make_base_entry("Clinique Anti-Blemish Solutions Clinical Clearing Gel", "Gel Secativo Antiacne com Ácido Salicílico", [], "tratamento", "Sephora"),
        _make_base_entry("Clinique Cheek Pop Blush Compacto", "Blush Assado Aveludado", [], "blush", "Sephora"),
    ],
    "Laura Mercier": [
        _make_base_entry("Laura Mercier Real Flawless Weightless Perfecting Foundation", "Fluido Segunda Pele Resistente à Água", ["1N2 Vanille", "2N1 Cashew", "2W1 Macadamia", "3N1 Buff", "3W1 Dusk", "4N1 Nutmeg", "5N1 Pecan", "6N1 Truffle"], "base", "Sephora"),
    ],
    "Bobbi Brown": [
        _make_base_entry("Bobbi Brown Skin Long-Wear Weightless Foundation SPF 15", "Fluido Matte Dimensional", ["Porcelain", "Ivory", "Warm Ivory", "Sand", "Beige", "Natural", "Natural Tan", "Warm Almond", "Chestnut", "Espresso"], "base", "Sephora"),
    ],
    "Make Up For Ever": [
        _make_base_entry("Make Up For Ever HD Skin Undetectable Longwear Foundation", "Fluido Imperceptível 4K Sem Transferência", ["1N06 (Y215)", "1Y08 (Y225)", "1N14 (Y245)", "2N22 (Y315)", "2Y32 (Y365)", "3N42 (Y415)", "3Y52 (Y445)", "4N68 (Y535)"], "base", "Sephora"),
    ],
    "KVD Beauty": [
        _make_base_entry("KVD Beauty Good Apple Skin-Perfecting Foundation Balm / Serum", "Bálsamo/Sérum de Cobertura Total", ["Light 008", "Light 012", "Medium 027", "Medium 036", "Tan 060", "Deep 082"], "base", "Sephora"),
    ],
    "Kosas": [
        _make_base_entry("Kosas Revealer Skin-Improving Foundation SPF 25", "Base Tratamento com Niacinamida e Esqualano", ["100 Very Light Neutral", "130 Light Warm", "160 Light Neutral Olive", "200 Light Medium Neutral", "220 Medium Neutral", "270 Medium Tan", "320 Tan Neutral", "400 Deep Neutral"], "base", "Sephora"),
    ],
    "Ilia Beauty": [
        _make_base_entry("Ilia Super Serum Skin Tint SPF 40", "Sérum Com Cor Mineral + Esqualano + Niacinamida", ["ST3 Balos", "ST6 Diaz", "ST7 Formosa", "ST10 Paloma", "ST11.5erno", "ST13 Kamari", "ST15 Porto Ferro", "ST18 Pavones"], "base", "Sephora"),
    ],
    "Tower 28": [
        _make_base_entry("Tower 28 SunnyDays SPF 30 Tinted Sunscreen Foundation", "Base Fotoprotetora Segura para Pele Sensível/Eczema (Selo National Eczema Association)", ["10 Larchmont", "15 Melrose", "20 Mulholland", "30 PCH", "40 Runyon", "50 Silver Lake", "65 Bay"], "base", "Sephora", True),
    ],
    "Saie": [
        _make_base_entry("Saie Glowy Super Skin Weightless Hydrobounce Serum Foundation", "Base Sérum Hidratante 85% Skincare", ["2", "5", "8", "12", "15", "19", "24", "30", "35"], "base", "Sephora"),
    ],
    "Milk Makeup": [
        _make_base_entry("Milk Makeup Hydro Grip Gel Skin Tint / Sunshine Skin Tint", "Gel-Tint Hidratante de Longa Duração", ["2", "4", "6", "8", "10", "12", "14"], "base", "Sephora"),
    ],
    "Glossier": [
        _make_base_entry("Glossier Stretch Fluid Foundation / Perfecting Skin Tint", "Gel-Creme Elástico Natural (89% Base Skincare)", ["Very Light 2", "Light 2", "Light 4", "Medium 1", "Medium 3", "Medium Tan 2", "Tan 3", "Deep 2"], "base", "Amazon"),
    ],
    "Patrick Ta": [
        _make_base_entry("Patrick Ta Major Skin Hydrating Luxe Liquid Foundation", "Fluido Luminoso de Luxo", ["3 Light Neutral", "7 Light Medium", "12 Medium Neutral", "16 Medium Tan", "22 Tan", "26 Deep"], "base", "Sephora"),
    ],
    "ONE/SIZE by Patrick Starrr": [
        _make_base_entry("ONE/SIZE Turn Up The Base Butter Silk Concealer / Powder Foundation", "Pó-Base / Corretivo Matte Aveludado", ["Fair 2", "Light 2", "Medium 2", "Medium Dark 2", "Dark 2", "Deep 2"], "base", "Sephora"),
    ],
    "Morphe": [
        _make_base_entry("Morphe Filter Effect Soft-Focus Foundation", "Fluido Soft-Focus com Ácido Hialurônico", ["Filter Light 4", "Filter Light 7", "Filter Medium 11", "Filter Medium 14", "Filter Tan 19", "Filter Rich 26"], "base", "Amazon"),
    ],
    "Juvia's Place": [
        _make_base_entry("Juvia's Place I Am Magic Velvety Matte Foundation", "Creme Matte Alta Cobertura", ["710 Marisol", "630 Cebu", "540 Casablanca", "440 Nairobi", "330 Lagos", "210 Kano"], "base", "Amazon"),
    ],
    "Kylie Cosmetics": [
        _make_base_entry("Kylie Cosmetics Power Plush Longwear Foundation", "Fluido Acetinado Confortável 12h", ["1N", "2.5W", "4N", "5W", "6.5N", "8W", "9.5N"], "base", "Sephora"),
    ],
    "r.e.m. beauty": [
        _make_base_entry("r.e.m. beauty Sweetener Foundation (by Ariana Grande)", "Fluido Hidratante com Niacinamida", ["Fair 4N", "Light 5W", "Light Medium 6N", "Medium 4W", "Tan 5N", "Deep 6W"], "base", "Sephora"),
    ],
    "Smashbox": [
        _make_base_entry("Smashbox Always On Skin-Balancing Foundation", "Fluido Equilibrador de Oleosidade e Hidratação", ["L10N", "L20W", "M10N", "M20W", "T10N", "D20N"], "base", "Sephora"),
    ],
    "bareMinerals": [
        _make_base_entry("bareMinerals Complexion Rescue Tinted Moisturizer SPF 30", "Gel-Creme Hidratante Mineral Com Cor", ["Opal 01", "Vanilla 02", "Buttercream 03", "Natural 05", "Ginger 06", "Tan 07", "Spice 08", "Chestnut 09"], "base", "Sephora", True),
    ],
    "Dior": [
        _make_base_entry("Dior Forever Skin Glow / Matte Foundation SPF 20", "Fluido Alta Costura 24h com Extratos Florais", ["0N", "1N", "1.5N", "2N (Claro-Médio Neutro)", "2W (Claro-Médio Quente)", "2WO (Oliva)", "3N (Médio Neutro)", "3W (Médio Quente)", "4N (Morena)", "5N", "6N (Negra)"], "base", "Sephora"),
        _make_base_entry("Dior Backstage Face & Body Foundation", "Fluido À Prova D'Água Construível", ["1N", "2N", "2W", "3N", "3W", "4N", "5N", "6N"], "base", "Sephora"),
        _make_base_entry("Dior Forever Skin Correct Corretivo 24h", "Corretivo Cremoso Hidratante", ["0N", "1N", "2N", "2W", "3N", "4N"], "corretivo", "Sephora"),
        _make_base_entry("Dior Rosy Glow Blush / Backstage Glow Face Palette", "Blush e Paleta Iluminadora", [], "blush", "Sephora"),
        _make_base_entry("Dior Capture Totale Le Sérum / Hyalushot", "Sérum Anti-Idade Regenerador", [], "serum", "Sephora"),
    ],
    "Lancôme": [
        _make_base_entry("Lancôme Teint Idole Ultra Wear 24H Foundation", "Fluido Matte Confortável Respirável", ["110C", "125W", "210C", "220C", "240W", "310N", "335W", "410W", "450N", "510C"], "base", "Sephora"),
        _make_base_entry("Lancôme Teint Idole Ultra Wear All Over Concealer", "Corretivo Fluido Matte Confortável", ["110", "215", "250", "335", "415"], "corretivo", "Sephora"),
        _make_base_entry("Lancôme Advanced Génifique Sérum Ativador de Juventude", "Sérum Microbioma com Frações Prebióticas e Probióticas", [], "serum", "Sephora"),
        _make_base_entry("Lancôme Hydra Zen Gel-Cream Hidratante Calmante Anti-Stress", "Gel-Creme Hidratante Calmante", [], "hidratante", "Sephora"),
    ],
    "Giorgio Armani": [
        _make_base_entry("Giorgio Armani Luminous Silk Perfect Glow Flawless Foundation", "Fluido Acetinado Icônico Sem Óleo", ["2 (Claro)", "3.5 (Claro-Médio)", "4 (Claro-Médio Quente)", "5 (Médio Neutro)", "5.5 (Médio Rosado)", "6 (Médio Oliva/Quente)", "7.5 (Morena)", "9 (Morena Escura)", "11.5 (Negra)", "14 (Negra Profunda)"], "base", "Sephora"),
        _make_base_entry("Giorgio Armani Luminous Silk Multi-Purpose Concealer", "Corretivo Acetinado Iluminador", ["2", "4", "5", "5.5", "6", "8", "11"], "corretivo", "Sephora"),
        _make_base_entry("Giorgio Armani Luminous Silk Glow Blush / Fluid Sheer", "Blush Sedoso e Iluminador Fluido", [], "blush", "Sephora"),
    ],
    "Shiseido": [
        _make_base_entry("Shiseido Synchro Skin Self-Refreshing Foundation SPF 30", "Fluido Auto-Renovável Resistente ao Calor e Umidade", ["140 Porcelain", "220 Linen", "250 Sand", "310 Silk", "340 Oak", "410 Sunstone", "440 Amber", "510 Suede"], "base", "Sephora"),
        _make_base_entry("Shiseido RevitalEssence Skin Glow Foundation SPF 30", "Base Sérum Fermentada com Niacinamida e Kefir", ["140", "220", "250", "310", "340", "410"], "base", "Sephora"),
        _make_base_entry("Shiseido The Perfect Protector SPF 50+ WetForce & HeatForce", "Fluido Fotoprotetor Ativado por Água e Calor", [], "protetor_solar", "Sephora"),
        _make_base_entry("Shiseido Ultimune Power Infusing Concentrate Sérum", "Sérum Fortalecedor Antioxidante", [], "serum", "Sephora"),
        _make_base_entry("Shiseido Clarifying Cleansing Foam Espuma de Limpeza", "Espuma Cremosa Japonesa Purificante", [], "sabonete_facial", "Sephora"),
    ],
    "SHEGLAM": [
        _make_base_entry(
            "SHEGLAM Complexion Pro Long Lasting Breathable Matte Foundation",
            "Fluido Matte Respirável de Longa Duração À Prova de Suor",
            [
                "Chantilly (Muito Claro Neutro)",
                "Porcelain (Claro Rosado)",
                "Shell (Claro Neutro-Quente)",
                "Fair (Claro)",
                "Nude (Claro-Médio Neutro)",
                "Buff (Claro-Médio Quente)",
                "Warm Vanilla (Médio-Claro Quente)",
                "Sand (Médio Neutro-Quente)",
                "Butterscotch (Médio Dourado)",
                "Honey (Médio Quente)",
                "Golden (Médio-Escuro Dourado)",
                "Tan (Morena Quente)",
                "Caramel (Morena Dourada)",
                "Amber (Morena Escura)",
                "Mocha (Pele Negra Quente)",
                "Espresso (Pele Negra Profunda)",
                "Truffle (Pele Negra Retinta)",
            ],
            "base",
            "Amazon",
        ),
        _make_base_entry(
            "SHEGLAM Skinfinite Hydrating Foundation",
            "Base Fluida Hidratante Glow Natural Segunda Pele",
            ["Porcelain", "Shell", "Nude", "Buff", "Warm Vanilla", "Sand", "Honey", "Tan", "Caramel", "Mocha", "Espresso"],
            "base",
            "Amazon",
        ),
        _make_base_entry(
            "SHEGLAM Skin Influencer Full Coverage Foundation Balm",
            "Bálsamo-Base Compacto de Alta Cobertura Aveludada",
            ["Chantilly", "Shell", "Nude", "Buff", "Warm Vanilla", "Sand", "Honey", "Wheat", "Caramel", "Mocha", "Espresso"],
            "base",
            "Amazon",
        ),
        _make_base_entry(
            "SHEGLAM Complexion Boost Concealer (Corretivo Líquido Matte Hidratante)",
            "Corretivo Líquido Alta Cobertura Matte Flexível Não-Craquelante",
            ["Chantilly", "Shell", "Vanilla", "Fair", "Nude", "Sand", "Natural", "Butterscotch", "Honey", "Caramel", "Chestnut", "Mocha", "Espresso"],
            "corretivo",
            "Amazon",
        ),
        _make_base_entry(
            "SHEGLAM 12HR Full Coverage Concealer (Corretivo Líquido 12 Horas)",
            "Corretivo Líquido 12h Alta Cobertura À Prova D'Água e Vincos",
            ["Chantilly", "Shell", "Vanilla", "Fair", "Warm Vanilla", "Sand", "Natural", "Honey", "Tan", "Caramel", "Mocha", "Truffle"],
            "corretivo",
            "Amazon",
        ),
        _make_base_entry(
            "SHEGLAM Magic Touch Cushion Concealer (Corretivo Aplicador Esponja Cushion)",
            "Corretivo Fluido Iluminador com Esponjinha Cushion Integrada",
            ["Porcelain", "Shell", "Vanilla", "Nude", "Sand", "Butterscotch", "Honey", "Caramel", "Mocha"],
            "corretivo",
            "Amazon",
        ),
        _make_base_entry(
            "SHEGLAM Out Of Sight Concealer Pot (Corretivo Cremoso de Pote Alta Camuflagem)",
            "Bálsamo Corretivo Cremoso de Pote para Camuflagem de Olheiras e Acne",
            ["Fair", "Light", "Nude", "Sand", "Medium", "Honey", "Tan", "Deep"],
            "corretivo",
            "Amazon",
        ),
        _make_base_entry(
            "SHEGLAM Under Eye Brightening Liquid Concealer (Corretivo Iluminador Área dos Olhos)",
            "Fluido Ultraleve Iluminador e Neutralizador de Olheiras Sem Marcar Linhas",
            ["Fair Pink (Rosado Iluminador)", "Light Peach (Pêssego Claro)", "Warm Apricot (Damasco Médio)", "Golden Caramel (Caramelo Quente)"],
            "corretivo",
            "Amazon",
        ),
        _make_base_entry(
            "SHEGLAM Stick It To 'Em Concealer Stick (Corretivo em Bastão Cremoso)",
            "Stick Corretivo Cremoso Aveludado de Cobertura Média a Alta",
            ["Shell", "Vanilla", "Nude", "Sand", "Honey", "Caramel", "Mocha"],
            "corretivo",
            "Amazon",
        ),
        _make_base_entry(
            "SHEGLAM Correct & Perfect Color Corrector / Palette (Corretivo Colorido Camuflagem)",
            "Corretivo Neutralizador de Subtom (Pêssego, Verde, Lavanda, Amarelo)",
            ["Peach (Neutraliza Olheiras Arroxeadas/Azuis)", "Green (Neutraliza Vermelhidão/Acne/Rosácea)", "Lavender (Neutraliza Manchas Amareladas)", "Yellow (Ilumina Olheiras Escuras)"],
            "corretivo",
            "Amazon",
        ),
        _make_base_entry(
            "SHEGLAM Color Bloom Liquid Blush Matte / Glow",
            "Blush Líquido Gel-Creme Esfumável com Aplicador Cushion",
            ["Love Cake", "Rose Ritual", "Devoted", "Risky Business", "Hush Hush", "Petal Talk"],
            "blush",
            "Amazon",
        ),
        _make_base_entry(
            "SHEGLAM Sun Sculpt Liquid Contour Contorno Líquido",
            "Contorno Líquido Gel-Creme Fácil de Esfumar",
            ["Soft Tan (Fundo Frio/Neutro)", "Golden Sun", "Warm Honey", "Camel", "Terracotta", "Earth"],
            "contorno",
            "Amazon",
        ),
        _make_base_entry(
            "SHEGLAM Glow Bloom Liquid Highlighter Iluminador",
            "Iluminador Líquido Gel-Creme Sem Partículas de Glitter",
            ["Vanilla Frost", "Bellini Brunch", "Tea Cake"],
            "iluminador",
            "Amazon",
        ),
        _make_base_entry(
            "SHEGLAM Insta-Ready Face & Under Eye Setting Powder Duo",
            "Duo de Pó Compacto (Área dos Olhos) + Pó Solto Matificante à Prova D'Água",
            [],
            "maquiagem",
            "Amazon",
        ),
        _make_base_entry(
            "SHEGLAM Good Grip Hydrating Primer",
            "Primer Gel Aderente Hidratante (Fixa a Maquiagem sem Esfarelar)",
            [],
            "hidratante",
            "Amazon",
        ),
        _make_base_entry(
            "SHEGLAM Birthday Skin Primer (Disfarce Óptico de Poros)",
            "Primer Bálsamo Aveludado Efeito Blur Instantâneo",
            [],
            "maquiagem",
            "Amazon",
        ),
        _make_base_entry(
            "SHEGLAM Lock'd In Setting Spray Bruma Fixadora",
            "Spray Fixador de Maquiagem de Longa Duração",
            [],
            "tonico",
            "Amazon",
        ),
    ],
}


def resolve_brand_name(raw_brand_query: str) -> Optional[str]:
    """Resolves a user-typed brand string (BR, US, or International) to the official brand."""
    norm = _normalize(raw_brand_query)
    if not norm:
        return None

    # 1. Exact or substring match against canonical brand names
    for canonical in BRAND_PRODUCT_CATALOG:
        c_norm = _normalize(canonical)
        if norm == c_norm or norm in c_norm or c_norm in norm:
            return canonical

    # 2. Match against aliases
    for canonical, aliases in BRAND_ALIASES.items():
        for al in aliases:
            al_norm = _normalize(al)
            if norm == al_norm or al_norm in norm or norm in al_norm:
                return canonical

    return None


def search_products_by_brand(
    brand_query: str,
    product_hint: Optional[str] = None,
    only_shade_products: bool = False,
) -> BrandSearchResponse:
    """
    Confirms the brand typed by the user and returns all corresponding products
    from that brand. If the brand is truly not found, returns `found=False` without
    inventing fake products.
    """
    cleaned_brand = brand_query.strip()
    resolved_brand = resolve_brand_name(cleaned_brand)
    hint_norm = _normalize(product_hint or "")

    if resolved_brand and resolved_brand in BRAND_PRODUCT_CATALOG:
        raw_items = BRAND_PRODUCT_CATALOG[resolved_brand]
        options: List[BrandProductOption] = []
        for it in raw_items:
            if only_shade_products:
                # Etapa 1.A (Findation): ONLY bases, concealers, and tinted sunscreens
                is_shade_anchor = (
                    it.get("category_detected") in ("base", "corretivo")
                    or bool(it.get("is_tinted_sunscreen"))
                )
                if not is_shade_anchor:
                    continue
            store_name = it.get("store_name", "Droga Raia")
            options.append(
                BrandProductOption(
                    official_name=it["official_name"],
                    brand=resolved_brand,
                    predominant_texture=it["predominant_texture"],
                    category_detected=it.get("category_detected", "skincare"),
                    is_tinted_sunscreen=bool(it.get("is_tinted_sunscreen", False)),
                    shades=list(it.get("shades", [])),
                    store_name=store_name,
                    store_url=build_reliable_store_url(it["official_name"], store_name),
                )
            )

        if hint_norm:
            matching = [
                o for o in options if hint_norm in _normalize(f"{o.official_name} {o.predominant_texture} {o.category_detected}")
            ]
            non_matching = [o for o in options if o not in matching]
            options = matching + non_matching

        return BrandSearchResponse(
            found=True,
            confirmed_brand=resolved_brand,
            brand_aliases_matched=[cleaned_brand],
            products=options,
            message=f"Marca confirmada: {resolved_brand} ({len(options)} produto(s) oficial(is) encontrado(s)).",
        )

    # Dynamic lookup for any other brand typed by the user via Tavily + LLM
    norm_brand = _normalize(cleaned_brand)
    search_q = f"marca {cleaned_brand} {product_hint or ''} cosméticos".strip()
    hits = tavily_search_official_stores(search_q, max_results=5)
    verified_hits = [
        h
        for h in hits
        if norm_brand in _normalize(f"{h.get('title', '')} {h.get('content', '')} {h.get('url', '')}")
    ]
    llm = get_openai_llm(smart=False) or get_gemini_llm()

    if llm and verified_hits:
        try:
            structured_llm = llm.with_structured_output(LLMDiscoveredBrandProducts)
            snippets = "\n".join(
                f"- {h.get('title')} | {h.get('url')} | {h.get('content', '')[:200]}"
                for h in verified_hits
            )
            prompt = (
                f"O usuário digitou a marca de cosméticos '{cleaned_brand}' (filtro opcional: '{product_hint or ''}').\n"
                f"Resultados reais encontrados nas farmácias e lojas oficiais:\n{snippets}\n\n"
                "Extraia SOMENTE os produtos reais dessa marca que aparecem nos resultados acima. "
                "NÃO invente produtos genéricos se a marca não estiver nos resultados."
            )
            discovered: LLMDiscoveredBrandProducts = structured_llm.invoke(prompt)
            if discovered and discovered.products:
                title_brand = discovered.confirmed_brand or cleaned_brand.title()
                for p in discovered.products:
                    p.brand = title_brand
                    p.store_url = build_reliable_store_url(p.official_name, p.store_name or "Droga Raia")
                return BrandSearchResponse(
                    found=True,
                    confirmed_brand=title_brand,
                    brand_aliases_matched=[cleaned_brand],
                    products=discovered.products,
                    message=f"Marca confirmada: {title_brand} ({len(discovered.products)} produtos encontrados nas lojas oficiais).",
                )
        except Exception as exc:
            logger.warning("Dynamic brand lookup error for '%s': %s", cleaned_brand, exc)

    # Brand was NOT found — never invent fake products!
    not_found_msg = (
        f"Marca não encontrada no Findation: Não encontramos bases ou corretivos da marca '{cleaned_brand}'."
        if only_shade_products
        else f"Marca não encontrada: Não encontramos produtos oficiais da marca '{cleaned_brand}' nas lojas e farmácias verificadas."
    )
    return BrandSearchResponse(
        found=False,
        confirmed_brand=None,
        brand_aliases_matched=[],
        products=[],
        message=not_found_msg,
    )


def get_catalog_for_frontend() -> Dict[str, Any]:
    """Returns the full brand -> products catalog with reliable store URLs."""
    brands_list: List[str] = list(BRAND_PRODUCT_CATALOG.keys())
    products_by_brand: Dict[str, List[Dict[str, Any]]] = {}

    for brand, items in BRAND_PRODUCT_CATALOG.items():
        enriched_items = []
        for it in items:
            copy_it = dict(it)
            copy_it["brand"] = brand
            copy_it["store_url"] = build_reliable_store_url(it["official_name"], it.get("store_name", "Droga Raia"))
            enriched_items.append(copy_it)
        products_by_brand[brand] = enriched_items

    return {
        "brands": brands_list,
        "products_by_brand": products_by_brand,
    }
