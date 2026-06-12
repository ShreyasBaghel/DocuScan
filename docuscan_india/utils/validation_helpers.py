from typing import List, Dict, Any
from utils.document_packet import ValidationResult, FieldResult

class ValidationHelpers:
    @staticmethod
    def get_validation_summary(validation_results: List[ValidationResult]) -> Dict[str, Any]:
        """Summarizes validation status into features."""
        failed_count = 0
        warn_count = 0
        checksum_failed = 0

        for r in validation_results:
            if r.status == "FAIL":
                failed_count += 1
                if "checksum" in r.field_name or "verhoeff" in r.field_name:
                    checksum_failed = 1
            elif r.status == "WARN":
                warn_count += 1

        return {
            "validation_fails": failed_count,
            "validation_warns": warn_count,
            "checksum_failed": checksum_failed
        }

    @staticmethod
    def get_missing_fields_count(fields: Dict[str, FieldResult]) -> int:
        """Counts how many extracted fields were not found or empty."""
        missing = 0
        for val in fields.values():
            if not val or val.value == "NOT_FOUND" or not val.value.strip():
                missing += 1
        return missing
