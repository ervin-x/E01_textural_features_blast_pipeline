from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
VENDOR_ROOT = PROJECT_ROOT / "vendor"
VENV_SITE_PACKAGES = sorted((PROJECT_ROOT / ".venv" / "lib").glob("python*/site-packages"))

for candidate in [*VENV_SITE_PACKAGES, SRC_ROOT, VENDOR_ROOT]:
    candidate_str = str(candidate)
    if candidate.exists() and candidate_str not in sys.path:
        sys.path.append(candidate_str)

import pyarrow.parquet as pq

from dataset.build_index import build_index_artifacts
from dataset.class_mapping import parse_predefined_classes
from dataset.parse_detection_labels import normalize_class_id, parse_label_file
from dataset.parse_masks import parse_mask_filename


class DatasetParsingTests(unittest.TestCase):
    def test_parse_predefined_classes_supports_both_formats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            indexed = Path(tmp_dir) / "indexed.txt"
            indexed.write_text("0 Lymphocyte\n7 blast\n15 gumpricht shadows\n", encoding="utf-8")
            line_only = Path(tmp_dir) / "line_only.txt"
            line_only.write_text("Lymphocyte\nblast\ngumpricht shadows\n", encoding="utf-8")

            self.assertEqual(parse_predefined_classes(indexed), {0: "Lymphocyte", 7: "blast", 15: "gumpricht shadows"})
            self.assertEqual(parse_predefined_classes(line_only), {0: "Lymphocyte", 1: "blast", 2: "gumpricht shadows"})

    def test_normalize_class_id_detects_float_like_tokens(self) -> None:
        self.assertEqual(normalize_class_id("7"), (7, False))
        self.assertEqual(normalize_class_id("7.0"), (7, True))

    def test_parse_label_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            label_path = Path(tmp_dir) / "sample.txt"
            label_path.write_text("7.0 0.5 0.5 0.1 0.1\n12 0.2 0.2 0.05 0.05\n", encoding="utf-8")
            rows = parse_label_file(label_path)

            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["class_id"], 7)
            self.assertTrue(rows[0]["class_id_was_float"])
            self.assertEqual(rows[1]["class_id"], 12)
            self.assertEqual(rows[1]["object_id_within_image"], 1)

    def test_parse_mask_filename(self) -> None:
        parsed = parse_mask_filename(Path("img123_class_7_obj_0004.png"))
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed["image_id"], "img123")
        self.assertEqual(parsed["class_id"], 7)
        self.assertEqual(parsed["object_id_within_image"], 4)


class BuildIndexIntegrationTests(unittest.TestCase):
    def test_build_index_artifacts_on_small_synthetic_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            data_root = tmp_root / "data"
            masks_root = tmp_root / "data_masks"
            output_root = tmp_root / "outputs"

            patient_dir = data_root / "P000001.001"
            patient_dir.mkdir(parents=True)
            (patient_dir / "images").mkdir()
            (patient_dir / "images_labels").mkdir()
            (patient_dir / "predefined_classes.txt").write_text(
                "0 Lymphocyte\n7 blast\n12 Bad cells\n15 gumpricht shadows\n", encoding="utf-8"
            )
            (patient_dir / "images" / "img001.jpg").write_bytes(b"")
            (patient_dir / "images_labels" / "img001.txt").write_text(
                "7 0.5 0.5 0.1 0.1\n12.0 0.2 0.2 0.05 0.05\n", encoding="utf-8"
            )

            part_dir = data_root / "part-1"
            part_dir.mkdir(parents=True)
            (part_dir / "images").mkdir()
            (part_dir / "images_labels").mkdir()
            (part_dir / "predefined_classes.txt").write_text("Lymphocyte\nblast\n", encoding="utf-8")
            (part_dir / "images" / "aux001.jpg").write_bytes(b"")
            (part_dir / "images_labels" / "aux001.txt").write_text("7 0.5 0.5 0.1 0.1\n", encoding="utf-8")

            mask_patient_dir = masks_root / "P000001.001"
            (mask_patient_dir / "images_results" / "masks_sam").mkdir(parents=True)
            (mask_patient_dir / "images").mkdir()
            (mask_patient_dir / "images_labels").mkdir()
            (mask_patient_dir / "predefined_classes.txt").write_text(
                "0 Lymphocyte\n7 blast\n12 Bad cells\n15 gumpricht shadows\n", encoding="utf-8"
            )
            (mask_patient_dir / "images_results" / "masks_sam" / "img001_class_7_obj_0000.png").write_bytes(b"")

            mask_part_dir = masks_root / "part-1"
            (mask_part_dir / "images_results" / "masks_sam").mkdir(parents=True)
            (mask_part_dir / "images").mkdir()
            (mask_part_dir / "images_labels").mkdir()
            (mask_part_dir / "predefined_classes.txt").write_text("Lymphocyte\nblast\n", encoding="utf-8")

            (data_root / "predefined_classes.txt").write_text("Lymphocyte\nblast\nBad cells\n", encoding="utf-8")

            result = build_index_artifacts(
                data_root=data_root,
                masks_root=masks_root,
                output_root=output_root,
                main_protocol_excludes={"part-1"},
            )

            self.assertTrue((output_root / "dataset_index" / "patient_inventory.csv").exists())
            self.assertTrue((output_root / "dataset_index" / "image_index.parquet").exists())
            self.assertTrue((output_root / "dataset_index" / "object_index.parquet").exists())
            self.assertTrue((output_root / "dataset_index" / "class_mapping.json").exists())
            self.assertTrue((output_root / "reports" / "data_audit_runtime.md").exists())

            object_table = pq.read_table(output_root / "dataset_index" / "object_index.parquet")
            image_table = pq.read_table(output_root / "dataset_index" / "image_index.parquet")
            self.assertEqual(object_table.num_rows, 3)
            self.assertEqual(image_table.num_rows, 2)

            class_mapping_payload = json.loads(
                (output_root / "dataset_index" / "class_mapping.json").read_text(encoding="utf-8")
            )
            self.assertIn("canonical_mapping", class_mapping_payload)
            self.assertEqual(result["summary"]["main_patient_dirs"], 1)
            self.assertEqual(result["summary"]["main_objects"], 2)


if __name__ == "__main__":
    unittest.main()
