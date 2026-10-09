from __future__ import annotations

from typing import Any

from saga.core.protocol import SagaSample, UNKNOWN


def validate_samples(samples: list[SagaSample], required_fields: list[str] | None = None) -> dict[str, Any]:
    required_fields = required_fields or ["class"]
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    for sample in samples:
        for field in required_fields:
            value = sample.label.class_name if field == "class" else sample.metadata.get(field)
            if value in {None, "", UNKNOWN}:
                errors.append(
                    {
                        "sample_id": sample.sample_id,
                        "field": field,
                        "message": f"Missing required field: {field}",
                        "path": sample.image.path,
                    }
                )
        if sample.metadata.get("parse_status") in {"partial", "failed"}:
            warnings.append(
                {
                    "sample_id": sample.sample_id,
                    "message": f"Parse status is {sample.metadata.get('parse_status')}",
                    "path": sample.image.path,
                }
            )
    return {
        "valid": not errors,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors[:100],
        "warnings": warnings[:100],
    }
