from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
VENDOR_ROOT = PROJECT_ROOT / "vendor"
VENV_SITE_PACKAGES = sorted((PROJECT_ROOT / ".venv" / "lib").glob("python*/site-packages"))

for candidate in [*VENV_SITE_PACKAGES, SRC_ROOT, VENDOR_ROOT]:
    candidate_str = str(candidate)
    if candidate.exists() and candidate_str not in sys.path:
        sys.path.append(candidate_str)

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ModuleNotFoundError as exc:
    raise SystemExit(
        "pyarrow is required for A0 because the plan requires parquet outputs. "
        "Install it into the experiment environment before running this script."
    ) from exc

from dataset.class_mapping import mapping_signature, normalize_mapping_for_json, parse_predefined_classes
from dataset.parse_detection_labels import parse_label_file
from dataset.parse_masks import build_mask_index


MAIN_PROTOCOL_EXCLUDES = {"part-1"}
QUALITY_OR_ARTIFACT_CLASS_IDS = {12, 14, 15}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

EXPECTED_SUMMARY = {
    "raw_top_level_dirs_data": 123,
    "raw_top_level_dirs_masks": 123,
    "raw_images": 12521,
    "raw_label_files": 12489,
    "raw_objects": 86873,
    "raw_mask_files": 86835,
    "main_patient_dirs": 122,
    "main_images": 12488,
    "main_label_files": 12456,
    "main_objects": 86396,
    "main_empty_label_files": 26,
    "main_images_without_labels": 32,
    "main_mask_files": 86360,
    "main_images_with_any_mask": 12428,
    "main_images_without_masks": 60,
    "main_checked_images_equal_label_mask_counts": 12395,
    "main_images_with_exact_label_mask_order_match": 12395,
    "main_float_class_rows": 4991,
    "main_blast_patients": 100,
    "main_nonblast_patients": 22,
    "main_images_with_blast": 8618,
    "main_images_only_blast": 3508,
    "main_images_blast_mixed": 5110,
    "main_images_without_blast": 3812,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build A0 dataset index artifacts for experiment E01.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT.parent / "Data" / "data",
        help="Path to the data directory containing patient folders.",
    )
    parser.add_argument(
        "--masks-root",
        type=Path,
        default=PROJECT_ROOT.parent / "Data" / "data_masks",
        help="Path to the mask directory containing patient folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs",
        help="Experiment outputs root.",
    )
    parser.add_argument(
        "--exclude-from-main-protocol",
        action="append",
        default=[],
        help="Top-level directories to exclude from the main patient-level protocol.",
    )
    return parser.parse_args()


def list_image_files(image_dir: Path) -> dict[str, Path]:
    if not image_dir.exists():
        return {}
    return {
        image_path.stem: image_path
        for image_path in sorted(image_dir.iterdir())
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES
    }


def list_label_files(labels_dir: Path) -> dict[str, Path]:
    if not labels_dir.exists():
        return {}
    return {label_path.stem: label_path for label_path in sorted(labels_dir.glob("*.txt"))}


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    ensure_parent(path)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(payload: dict[str, object], path: Path) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_parquet(rows: list[dict[str, object]], path: Path) -> None:
    ensure_parent(path)
    if rows:
        table = pa.Table.from_pylist(rows)
    else:
        table = pa.table({})
    pq.write_table(table, path)


def determine_image_status(
    has_image: bool,
    has_label: bool,
    empty_label_file: bool,
    label_count: int,
    mask_count: int,
    class_order_match: bool,
) -> str:
    if not has_image and has_label:
        return "missing_image"
    if has_image and not has_label:
        return "missing_label"
    if empty_label_file:
        return "empty_label_file"
    if label_count == 0 and mask_count == 0:
        return "no_labels_no_masks"
    if mask_count == 0:
        return "no_masks"
    if label_count != mask_count:
        return "mask_count_mismatch"
    if not class_order_match:
        return "mask_class_order_mismatch"
    return "ok"


def build_class_mapping_payload(
    patient_class_mappings: dict[str, dict[int, str]],
    root_reference_mapping: dict[int, str],
) -> tuple[dict[str, object], dict[tuple[tuple[int, str], ...], str]]:
    signatures = Counter(mapping_signature(mapping) for mapping in patient_class_mappings.values())
    signature_to_variant_id: dict[tuple[tuple[int, str], ...], str] = {}

    for index, (signature, _) in enumerate(
        sorted(signatures.items(), key=lambda item: (-item[1], item[0])), start=1
    ):
        signature_to_variant_id[signature] = f"variant_{index:02d}"

    canonical_signature = max(signatures.items(), key=lambda item: item[1])[0] if signatures else ()
    variant_payload = []
    for signature, count in sorted(signatures.items(), key=lambda item: (-item[1], item[0])):
        variant_id = signature_to_variant_id[signature]
        variant_mapping = {class_id: class_name for class_id, class_name in signature}
        variant_payload.append(
            {
                "variant_id": variant_id,
                "patient_count": count,
                "patients": sorted(
                    patient_id
                    for patient_id, patient_mapping in patient_class_mappings.items()
                    if mapping_signature(patient_mapping) == signature
                ),
                "mapping": normalize_mapping_for_json(variant_mapping),
            }
        )

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "canonical_mapping": normalize_mapping_for_json({class_id: class_name for class_id, class_name in canonical_signature}),
        "root_reference_mapping": normalize_mapping_for_json(root_reference_mapping),
        "mapping_variants": variant_payload,
        "main_protocol_excludes": sorted(MAIN_PROTOCOL_EXCLUDES),
        "notes": [
            "canonical_mapping is selected as the most frequent patient-specific mapping",
            "part-1 is excluded from the main patient-level protocol by default",
        ],
    }
    return payload, signature_to_variant_id


def summarize_protocol(
    patient_inventory_rows: list[dict[str, object]],
    image_rows: list[dict[str, object]],
    object_rows: list[dict[str, object]],
    raw_top_level_dirs_data: int,
    raw_top_level_dirs_masks: int,
) -> dict[str, int]:
    raw_patients = patient_inventory_rows
    main_patients = [row for row in patient_inventory_rows if bool(row["include_in_main_protocol"])]
    main_images = [row for row in image_rows if bool(row["include_in_main_protocol"]) and bool(row["has_image"])]
    main_objects = [row for row in object_rows if bool(row["include_in_main_protocol"])]

    def sum_field(rows: Iterable[dict[str, object]], field: str) -> int:
        return sum(int(row.get(field, 0)) for row in rows)

    summary = {
        "raw_top_level_dirs_data": raw_top_level_dirs_data,
        "raw_top_level_dirs_masks": raw_top_level_dirs_masks,
        "raw_images": sum_field(raw_patients, "images_count"),
        "raw_label_files": sum_field(raw_patients, "label_files_count"),
        "raw_objects": sum_field(raw_patients, "objects_count"),
        "raw_mask_files": sum_field(raw_patients, "mask_files_count"),
        "main_patient_dirs": len(main_patients),
        "main_images": len(main_images),
        "main_label_files": sum_field(main_patients, "label_files_count"),
        "main_objects": len(main_objects),
        "main_empty_label_files": sum_field(main_patients, "empty_label_files_count"),
        "main_images_without_labels": sum_field(main_patients, "images_without_labels_count"),
        "main_mask_files": sum_field(main_patients, "mask_files_count"),
        "main_images_with_any_mask": sum_field(main_patients, "images_with_any_mask_count"),
        "main_images_without_masks": sum_field(main_patients, "images_without_masks_count"),
        "main_checked_images_equal_label_mask_counts": sum_field(
            main_patients, "checked_images_equal_label_mask_counts"
        ),
        "main_images_with_exact_label_mask_order_match": sum_field(
            main_patients, "images_with_exact_label_mask_order_match"
        ),
        "main_float_class_rows": sum_field(main_patients, "float_class_rows_count"),
        "main_blast_patients": sum(1 for row in main_patients if bool(row["patient_has_blast"])),
        "main_nonblast_patients": sum(1 for row in main_patients if not bool(row["patient_has_blast"])),
        "main_images_with_blast": sum_field(main_patients, "images_with_blast_count"),
        "main_images_only_blast": sum_field(main_patients, "images_only_blast_count"),
        "main_images_blast_mixed": sum_field(main_patients, "images_blast_mixed_count"),
        "main_images_without_blast": sum_field(main_patients, "images_without_blast_count"),
    }
    return summary


def build_runtime_report(
    summary: dict[str, int],
    output_paths: dict[str, str],
    path: Path,
) -> None:
    ensure_parent(path)
    lines = [
        "# Runtime Data Audit",
        "",
        f"Generated at: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "## Expected Comparison",
        "",
        "| Metric | Expected | Actual | Status |",
        "|---|---:|---:|---|",
    ]

    for key in sorted(EXPECTED_SUMMARY):
        actual = summary.get(key, 0)
        expected = EXPECTED_SUMMARY[key]
        status = "MATCH" if actual == expected else "DIFF"
        lines.append(f"| {key} | {expected} | {actual} | {status} |")

    lines.extend(
        [
            "",
            "## Output Artifacts",
            "",
        ]
    )
    for name, artifact_path in sorted(output_paths.items()):
        lines.append(f"- `{name}`: `{artifact_path}`")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_index_artifacts(
    data_root: Path,
    masks_root: Path,
    output_root: Path,
    main_protocol_excludes: set[str] | None = None,
) -> dict[str, object]:
    excludes = set(main_protocol_excludes or set()) | MAIN_PROTOCOL_EXCLUDES

    data_dirs = {
        path.name: path
        for path in sorted(data_root.iterdir())
        if path.is_dir() and not path.name.startswith(".")
    }
    mask_dirs = {
        path.name: path
        for path in sorted(masks_root.iterdir())
        if path.is_dir() and not path.name.startswith(".")
    }
    top_level_ids = sorted(set(data_dirs) | set(mask_dirs))

    patient_class_mappings: dict[str, dict[int, str]] = {}
    for top_level_id in top_level_ids:
        candidate_paths = [
            data_dirs.get(top_level_id, Path()) / "predefined_classes.txt" if top_level_id in data_dirs else None,
            mask_dirs.get(top_level_id, Path()) / "predefined_classes.txt" if top_level_id in mask_dirs else None,
        ]
        mapping_path = next((path for path in candidate_paths if path is not None and path.exists()), None)
        patient_class_mappings[top_level_id] = parse_predefined_classes(mapping_path) if mapping_path else {}

    root_reference_mapping = parse_predefined_classes(data_root / "predefined_classes.txt")
    class_mapping_payload, signature_to_variant_id = build_class_mapping_payload(
        patient_class_mappings=patient_class_mappings,
        root_reference_mapping=root_reference_mapping,
    )

    patient_inventory_rows: list[dict[str, object]] = []
    image_rows: list[dict[str, object]] = []
    object_rows: list[dict[str, object]] = []

    for patient_id in top_level_ids:
        data_dir = data_dirs.get(patient_id)
        mask_dir = mask_dirs.get(patient_id)
        include_in_main_protocol = patient_id not in excludes
        class_mapping = patient_class_mappings.get(patient_id, {})
        class_mapping_variant_id = signature_to_variant_id.get(mapping_signature(class_mapping), "variant_unknown")

        image_files = list_image_files(data_dir / "images") if data_dir else list_image_files(mask_dir / "images")
        label_files = (
            list_label_files(data_dir / "images_labels")
            if data_dir and (data_dir / "images_labels").exists()
            else list_label_files(mask_dir / "images_labels") if mask_dir else {}
        )
        mask_entries_by_image = (
            build_mask_index(mask_dir / "images_results" / "masks_sam") if mask_dir else {}
        )

        patient_counts = Counter()
        patient_counts["images_count"] = len(image_files)
        patient_counts["label_files_count"] = len(label_files)
        patient_counts["mask_files_count"] = sum(len(entries) for entries in mask_entries_by_image.values())

        image_ids = sorted(set(image_files) | set(label_files) | set(mask_entries_by_image))

        for image_id in image_ids:
            image_path = image_files.get(image_id)
            label_path = label_files.get(image_id)
            mask_entries = mask_entries_by_image.get(image_id, [])
            label_entries = parse_label_file(label_path) if label_path else []

            has_image = image_path is not None
            has_label = label_path is not None
            empty_label_file = has_label and not label_entries
            label_count = len(label_entries)
            mask_count = len(mask_entries)
            contains_blast = any(int(entry["class_id"]) == 7 for entry in label_entries)
            contains_non_blast = any(int(entry["class_id"]) != 7 for entry in label_entries)
            count_match = has_label and label_count > 0 and label_count == mask_count
            class_order_match = count_match and [
                int(entry["class_id"]) for entry in label_entries
            ] == [int(entry["class_id"]) for entry in mask_entries]
            has_any_mask = mask_count > 0

            if has_image and not has_label:
                patient_counts["images_without_labels_count"] += 1
            if has_label and not has_image:
                patient_counts["labels_without_images_count"] += 1
            if empty_label_file:
                patient_counts["empty_label_files_count"] += 1
            if has_any_mask:
                patient_counts["images_with_any_mask_count"] += 1
            if has_image and not has_any_mask:
                patient_counts["images_without_masks_count"] += 1
            if count_match:
                patient_counts["checked_images_equal_label_mask_counts"] += 1
            if class_order_match:
                patient_counts["images_with_exact_label_mask_order_match"] += 1
            if has_label and label_count != mask_count:
                patient_counts["images_with_count_mismatch"] += 1

            if has_label and not empty_label_file:
                if contains_blast:
                    patient_counts["images_with_blast_count"] += 1
                    if contains_non_blast:
                        patient_counts["images_blast_mixed_count"] += 1
                    else:
                        patient_counts["images_only_blast_count"] += 1
                else:
                    patient_counts["images_without_blast_count"] += 1

            image_rows.append(
                {
                    "patient_id": patient_id,
                    "include_in_main_protocol": include_in_main_protocol,
                    "image_id": image_id,
                    "image_path": str(image_path.resolve()) if image_path else None,
                    "label_path": str(label_path.resolve()) if label_path else None,
                    "has_image": has_image,
                    "has_label": has_label,
                    "empty_label_file": empty_label_file,
                    "has_any_mask": has_any_mask,
                    "label_object_count": label_count,
                    "mask_object_count": mask_count,
                    "label_mask_count_match": label_count == mask_count if has_label else False,
                    "label_mask_class_order_match": class_order_match,
                    "contains_blast": contains_blast,
                    "contains_non_blast": contains_non_blast,
                    "blast_object_count": sum(1 for entry in label_entries if int(entry["class_id"]) == 7),
                    "non_blast_object_count": sum(1 for entry in label_entries if int(entry["class_id"]) != 7),
                    "image_status": determine_image_status(
                        has_image=has_image,
                        has_label=has_label,
                        empty_label_file=empty_label_file,
                        label_count=label_count,
                        mask_count=mask_count,
                        class_order_match=class_order_match,
                    ),
                }
            )

            for detection_entry in label_entries:
                object_id = int(detection_entry["object_id_within_image"])
                expected_mask = mask_entries[object_id] if object_id < len(mask_entries) else None
                has_mask = expected_mask is not None and int(expected_mask["class_id"]) == int(detection_entry["class_id"])

                class_id = int(detection_entry["class_id"])
                class_name = class_mapping.get(class_id, f"UNKNOWN_CLASS_{class_id}")
                patient_counts["objects_count"] += 1
                if bool(detection_entry["class_id_was_float"]):
                    patient_counts["float_class_rows_count"] += 1
                if class_id == 7:
                    patient_counts["blast_objects_count"] += 1
                else:
                    patient_counts["non_blast_objects_count"] += 1
                if not has_mask:
                    patient_counts["objects_without_masks_count"] += 1

                object_rows.append(
                    {
                        "patient_id": patient_id,
                        "include_in_main_protocol": include_in_main_protocol,
                        "image_id": image_id,
                        "image_path": str(image_path.resolve()) if image_path else None,
                        "label_path": str(label_path.resolve()) if label_path else None,
                        "mask_path": expected_mask["mask_path"] if has_mask else None,
                        "object_id_within_image": object_id,
                        "raw_class_id": detection_entry["raw_class_id"],
                        "class_id": class_id,
                        "class_name": class_name,
                        "class_id_was_float": bool(detection_entry["class_id_was_float"]),
                        "x_center_norm": float(detection_entry["x_center_norm"]),
                        "y_center_norm": float(detection_entry["y_center_norm"]),
                        "width_norm": float(detection_entry["width_norm"]),
                        "height_norm": float(detection_entry["height_norm"]),
                        "has_mask": has_mask,
                        "mask_status": "matched" if has_mask else "missing",
                        "mask_class_id": int(expected_mask["class_id"]) if expected_mask is not None else None,
                        "label_count_for_image": label_count,
                        "mask_count_for_image": mask_count,
                        "image_has_any_mask": has_any_mask,
                        "is_blast": class_id == 7,
                        "is_quality_or_artifact": class_id in QUALITY_OR_ARTIFACT_CLASS_IDS,
                    }
                )

        patient_inventory_rows.append(
            {
                "patient_id": patient_id,
                "include_in_main_protocol": include_in_main_protocol,
                "has_data_dir": data_dir is not None,
                "has_mask_dir": mask_dir is not None,
                "images_count": patient_counts["images_count"],
                "label_files_count": patient_counts["label_files_count"],
                "empty_label_files_count": patient_counts["empty_label_files_count"],
                "images_without_labels_count": patient_counts["images_without_labels_count"],
                "labels_without_images_count": patient_counts["labels_without_images_count"],
                "objects_count": patient_counts["objects_count"],
                "float_class_rows_count": patient_counts["float_class_rows_count"],
                "mask_files_count": patient_counts["mask_files_count"],
                "images_with_any_mask_count": patient_counts["images_with_any_mask_count"],
                "images_without_masks_count": patient_counts["images_without_masks_count"],
                "checked_images_equal_label_mask_counts": patient_counts["checked_images_equal_label_mask_counts"],
                "images_with_exact_label_mask_order_match": patient_counts[
                    "images_with_exact_label_mask_order_match"
                ],
                "images_with_count_mismatch": patient_counts["images_with_count_mismatch"],
                "objects_without_masks_count": patient_counts["objects_without_masks_count"],
                "blast_objects_count": patient_counts["blast_objects_count"],
                "non_blast_objects_count": patient_counts["non_blast_objects_count"],
                "patient_has_blast": patient_counts["blast_objects_count"] > 0,
                "images_with_blast_count": patient_counts["images_with_blast_count"],
                "images_only_blast_count": patient_counts["images_only_blast_count"],
                "images_blast_mixed_count": patient_counts["images_blast_mixed_count"],
                "images_without_blast_count": patient_counts["images_without_blast_count"],
                "class_mapping_size": len(class_mapping),
                "class_mapping_variant_id": class_mapping_variant_id,
            }
        )

    dataset_index_dir = output_root / "dataset_index"
    reports_dir = output_root / "reports"

    patient_inventory_path = dataset_index_dir / "patient_inventory.csv"
    image_index_path = dataset_index_dir / "image_index.parquet"
    object_index_path = dataset_index_dir / "object_index.parquet"
    class_mapping_path = dataset_index_dir / "class_mapping.json"
    mask_coverage_report_path = dataset_index_dir / "mask_coverage_report.csv"
    runtime_report_path = reports_dir / "data_audit_runtime.md"

    write_csv(patient_inventory_rows, patient_inventory_path)
    write_parquet(image_rows, image_index_path)
    write_parquet(object_rows, object_index_path)
    write_json(class_mapping_payload, class_mapping_path)
    write_csv(image_rows, mask_coverage_report_path)

    summary = summarize_protocol(
        patient_inventory_rows=patient_inventory_rows,
        image_rows=image_rows,
        object_rows=object_rows,
        raw_top_level_dirs_data=len(data_dirs),
        raw_top_level_dirs_masks=len(mask_dirs),
    )
    build_runtime_report(
        summary=summary,
        output_paths={
            "patient_inventory_csv": str(patient_inventory_path),
            "image_index_parquet": str(image_index_path),
            "object_index_parquet": str(object_index_path),
            "class_mapping_json": str(class_mapping_path),
            "mask_coverage_report_csv": str(mask_coverage_report_path),
        },
        path=runtime_report_path,
    )

    return {
        "summary": summary,
        "paths": {
            "patient_inventory_csv": str(patient_inventory_path),
            "image_index_parquet": str(image_index_path),
            "object_index_parquet": str(object_index_path),
            "class_mapping_json": str(class_mapping_path),
            "mask_coverage_report_csv": str(mask_coverage_report_path),
            "data_audit_runtime_md": str(runtime_report_path),
        },
    }


def main() -> None:
    args = parse_args()
    excludes = set(args.exclude_from_main_protocol or []) | MAIN_PROTOCOL_EXCLUDES
    result = build_index_artifacts(
        data_root=args.data_root,
        masks_root=args.masks_root,
        output_root=args.output_root,
        main_protocol_excludes=excludes,
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(json.dumps(result["paths"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
