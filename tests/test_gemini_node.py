import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "gemini_outfit_caption_test_module",
    ROOT / "nodes" / "gemini.py",
    submodule_search_locations=[str(ROOT / "nodes")],
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class GeminiNodeTests(unittest.TestCase):
    def test_mapping_contains_exactly_one_node(self):
        self.assertEqual(["GeminiOutfitCaption"], list(MODULE.NODE_CLASS_MAPPINGS))
        inputs = MODULE.GeminiOutfitCaption.INPUT_TYPES()
        model_choices = inputs["required"]["model"][0]
        self.assertIsInstance(model_choices, list)
        self.assertGreaterEqual(len(model_choices), 1)
        self.assertNotIn("api_key", inputs["required"])
        self.assertEqual(
            "GEMINI_API_KEY_COMFYUI", inputs["required"]["api_key_env"][1]["default"]
        )
        self.assertEqual(4096, inputs["required"]["max_tokens"][1]["default"])

    def test_parser_and_adapter(self):
        raw = """**Garment pieces:**
One-piece outfit.

**Coverage and silhouette:**
Deep plunging V-neckline.

Accessories:
- Sunglasses.
- Garment belt.
- Black handbag with a loose shoulder strap.

Preservation:
- Preserve the coat seams.
- Preserve the handbag and its handle.

Uncertainty:
Maybe two pieces.
"""
        response = {"candidates": [{"content": {"parts": [{"text": raw}]}}]}
        self.assertEqual(raw.strip(), MODULE.extract_output_text(response))
        adapted = MODULE.adapt_caption(raw, no_person_accessories=True)
        self.assertNotIn("Uncertainty", adapted)
        self.assertNotIn("Sunglasses", adapted)
        self.assertNotIn("handbag", adapted.lower())
        self.assertNotIn("Maybe two pieces", adapted)
        self.assertIn("Garment belt", adapted)
        self.assertIn("coat seams", adapted)
        self.assertIn("outer garments must remain outermost", adapted)
        self.assertIn("Do not remove described outer-garment sleeves", adapted)
        self.assertIn("front neckline", adapted)
        self.assertEqual("Red", MODULE.split_sections("Color: Red")["Color"])
        self.assertIn("handbag", MODULE.adapt_caption(raw, no_person_accessories=False).lower())

    def test_gemini_request_sets_output_token_limit(self):
        response = {"candidates": [{"content": {"parts": [{"text": "Color: Black"}]}}]}
        with mock.patch.object(MODULE, "_request_json", return_value=response) as request_json:
            MODULE.generate_caption(b"png", "gemini-test", "secret", 10.0, 1234)
        request = request_json.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(1234, payload["generationConfig"]["maxOutputTokens"])

    def test_accessory_filter_preserves_mixed_color_material_and_footwear_details(self):
        raw = """Color:
The shirt is dark grey. The skirt is black. The footwear is black. The clutch bag is black. The earrings are silver-toned.

Material family:
Latex/PVC/rubber (for shirt and skirt), Patent leather (for footwear and clutch bag), Metal (for earrings).

Preservation:
Preserve the shirt seams and black skirt. Crucially, preserve the extreme platform and stiletto heel design. Maintain the clutch bag and earrings.
"""
        adapted = MODULE.adapt_caption(raw, no_person_accessories=True)
        self.assertIn("The skirt is black", adapted)
        self.assertIn("The footwear is black", adapted)
        self.assertIn("Patent leather (for footwear)", adapted)
        self.assertIn("extreme platform and stiletto heel design", adapted)
        self.assertNotIn("clutch bag", adapted.lower())
        self.assertNotIn("earrings", adapted.lower())

    def test_filter_sort_and_version_drift(self):
        catalog = {
            "models": [
                {"id": "yellow", "rating": "yellow", "score": 70, "tested_model_version": "1"},
                {"id": "green-low", "rating": "green", "score": 80, "tested_model_version": "1"},
                {"id": "green-high", "rating": "green", "score": 95, "tested_model_version": "1"},
                {"id": "changed", "rating": "green", "score": 99, "tested_model_version": "1"},
            ]
        }
        live = [
            {"id": "unknown", "version": "1"},
            {"id": "yellow", "version": "1"},
            {"id": "green-low", "version": "1"},
            {"id": "green-high", "version": "1"},
            {"id": "changed", "version": "2"},
        ]
        ranked = MODULE.rank_models(catalog, live)
        self.assertEqual(["green-high", "green-low", "yellow", "changed"], [item["id"] for item in ranked])
        self.assertEqual("untested", ranked[-1]["rating"])

    def test_provider_failure_uses_stale_cache_without_changing_rating(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "models.json"
            with mock.patch.object(MODULE, "CACHE_PATH", cache), mock.patch.object(
                MODULE, "fetch_model_list", side_effect=TimeoutError("temporary")
            ):
                cache.write_text(
                    json.dumps({"models": [{"id": "gemini-2.5-flash", "version": "001"}]})
                )
                result = MODULE.available_models("secret")
        self.assertTrue(result["stale"])
        self.assertEqual("green", result["models"][0]["rating"])
        self.assertNotIn("[stale]", result["models"][0]["label"])

    def test_scored_entries_pin_the_tested_version(self):
        # Invariante: ein abgeschlossener Benchmark (score gesetzt) muss die exakt
        # getestete Version festhalten, sonst zeigt die Bewertung auf eine andere
        # Modellversion als die gemessene. Konkrete Werte (93, "green") sind reine
        # models.json-Daten und werden bewusst nicht mitgeprueft.
        for entry in MODULE.load_catalog()["models"]:
            if entry.get("score") is not None:
                self.assertEqual(entry["model_version"], entry["tested_model_version"])

    def test_batch_is_rejected_before_encoding(self):
        image = mock.Mock()
        image.shape = (2, 16, 16, 3)
        with self.assertRaisesRegex(ValueError, "batches"):
            MODULE.encode_image(image, 1.0)

    def test_custom_model_and_secret_sanitizing(self):
        self.assertEqual("gemini-custom", MODULE.resolve_model(MODULE.CUSTOM_MODEL, "models/gemini-custom"))
        self.assertNotIn("top-secret", MODULE._sanitize_error("failed top-secret", "top-secret"))


if __name__ == "__main__":
    unittest.main()
