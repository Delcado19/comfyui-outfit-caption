import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "nvidia_outfit_caption_test_module",
    ROOT / "nodes" / "nvidia_nim.py",
    submodule_search_locations=[str(ROOT / "nodes")],
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class NvidiaNodeTests(unittest.TestCase):
    def test_mapping_contains_exactly_one_node(self):
        self.assertEqual(["NvidiaNimOutfitCaption"], list(MODULE.NODE_CLASS_MAPPINGS))
        inputs = MODULE.NvidiaNimOutfitCaption.INPUT_TYPES()
        model_choices = inputs["required"]["model"][0]
        self.assertIsInstance(model_choices, list)
        self.assertGreaterEqual(len(model_choices), 1)
        self.assertNotIn("api_key", inputs["required"])
        self.assertEqual(
            "NVIDIA_API_KEY_COMFYUI", inputs["required"]["api_key_env"][1]["default"]
        )

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
        response = {"choices": [{"message": {"content": raw}, "finish_reason": "stop"}]}
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

    def test_reasoning_without_final_text_is_failure(self):
        response = {
            "choices": [
                {"message": {"content": None, "reasoning_content": "thinking"}, "finish_reason": "length"}
            ]
        }
        with self.assertRaisesRegex(RuntimeError, "reasoning but no final caption"):
            MODULE.extract_output_text(response)

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
            "excluded_models": {"blocked": "Not callable in this account."},
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
            {"id": "vendor/new-vision", "version": "1"},
            {"id": "blocked", "version": "1"},
        ]
        ranked = MODULE.rank_models(catalog, live, ["new-vision", "blocked"])
        self.assertEqual(
            ["green-high", "green-low", "yellow", "changed", "vendor/new-vision"],
            [item["id"] for item in ranked],
        )
        self.assertEqual("untested", ranked[-1]["rating"])

    def test_live_catalog_parser_deduplicates_embedded_endpoint_names(self):
        page = (
            r'\"resourceType\":\"ENDPOINT\",\"name\":\"kimi-k2.6\" '
            r'\"resourceType\":\"ENDPOINT\",\"name\":\"google-paligemma\" '
            r'\"resourceType\":\"ENDPOINT\",\"name\":\"kimi-k2.6\"'
        )
        self.assertEqual(
            ["kimi-k2.6", "google-paligemma"], MODULE._parse_image_model_names(page)
        )
        self.assertTrue(
            MODULE._is_official_image_model("google/paligemma", {"google-paligemma"})
        )

    def test_provider_failure_uses_stale_cache_without_changing_rating(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "models.json"
            with mock.patch.object(MODULE, "CACHE_PATH", cache), mock.patch.object(
                MODULE, "fetch_model_list", side_effect=TimeoutError("temporary")
            ):
                cache.write_text(
                    json.dumps(
                        {
                            "models": [
                                {
                                    "id": "meta/llama-3.2-11b-vision-instruct",
                                    "version": "735790403",
                                }
                            ],
                            "image_model_names": ["llama-3.2-11b-vision-instruct"],
                            "verified": True,
                        }
                    )
                )
                result = MODULE.available_models("secret")
        self.assertTrue(result["stale"])
        self.assertEqual("red", result["models"][0]["rating"])
        self.assertNotIn("[stale]", result["models"][0]["label"])

    def test_refresh_probes_candidates_and_caches_only_working_models(self):
        live = [
            {"id": "vendor/good-vision", "version": "1"},
            {"id": "vendor/bad-vision", "version": "1"},
        ]

        def probe(model_id, api_key, timeout):
            if model_id == "vendor/bad-vision":
                raise RuntimeError("HTTP 404: not found")

        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "models.json"
            with mock.patch.object(MODULE, "CACHE_PATH", cache), \
                mock.patch.object(MODULE, "fetch_model_list", return_value=live), \
                mock.patch.object(MODULE, "fetch_image_model_names", return_value=["good-vision", "bad-vision"]), \
                mock.patch.object(MODULE, "load_catalog", return_value={"models": []}), \
                mock.patch.object(MODULE, "_probe_model", side_effect=probe):
                result = MODULE.available_models("secret")
                cached = json.loads(cache.read_text(encoding="utf-8"))
        self.assertEqual(["vendor/good-vision"], [item["id"] for item in result["models"]])
        self.assertIn("vendor/bad-vision", result["probe_errors"])
        self.assertTrue(cached["verified"])
        self.assertEqual(["vendor/good-vision"], [item["id"] for item in cached["models"]])

    def test_dropdown_model_must_be_from_verified_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "models.json"
            cache.write_text(
                json.dumps({"models": [{"id": "vendor/good-vision", "version": "1"}], "verified": True}),
                encoding="utf-8",
            )
            with mock.patch.object(MODULE, "CACHE_PATH", cache):
                self.assertEqual("vendor/good-vision", MODULE.resolve_model("⚪ — | vendor/good-vision", ""))
                with self.assertRaisesRegex(ValueError, "has not passed the live NVIDIA probe"):
                    MODULE.resolve_model("⚪ — | vendor/bad-vision", "")
                self.assertEqual(
                    "vendor/bad-vision",
                    MODULE.resolve_model("⚪ — | vendor/bad-vision", "vendor/bad-vision"),
                )

    def test_completed_catalog_scores_match_per_image_totals_and_rating_bands(self):
        catalog = MODULE.load_catalog()
        for entry in catalog["models"]:
            if entry["rating"] == "untested":
                continue
            self.assertEqual(entry["score"], sum(item["total"] for item in entry["benchmark_scores"]))
            expected = "green" if entry["score"] >= 80 else "yellow" if entry["score"] >= 60 else "red"
            self.assertEqual(expected, entry["rating"])

    def test_batch_is_rejected_before_encoding(self):
        image = mock.Mock()
        image.shape = (2, 16, 16, 3)
        with self.assertRaisesRegex(ValueError, "batches"):
            MODULE.encode_image(image, 1.0)

    def test_saved_excluded_dropdown_model_is_rejected_before_request(self):
        blocked = "old label | microsoft/phi-3-vision-128k-instruct [stale]"
        with self.assertRaisesRegex(ValueError, "microsoft/phi-3-vision-128k-instruct.*excluded"):
            MODULE.resolve_model(blocked, "")
        self.assertEqual(
            "microsoft/phi-3-vision-128k-instruct",
            MODULE.resolve_model(blocked, "microsoft/phi-3-vision-128k-instruct"),
        )

    def test_custom_model_and_secret_sanitizing(self):
        self.assertEqual("vendor/custom", MODULE.resolve_model(MODULE.CUSTOM_MODEL, "vendor/custom"))
        self.assertNotIn("top-secret", MODULE._sanitize_error("failed top-secret", "top-secret"))


if __name__ == "__main__":
    unittest.main()
