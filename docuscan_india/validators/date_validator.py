from datetime import datetime, date
from typing import List, Dict
from utils.document_packet import ValidationResult, FieldResult, DocumentType

class DateValidator:
    @staticmethod
    def validate(doc_type: DocumentType, fields: Dict[str, FieldResult]) -> List[ValidationResult]:
        results: List[ValidationResult] = []
        today = date.today()

        def parse_date(val: str) -> datetime:
            try:
                return datetime.strptime(val, "%Y-%m-%d").date()
            except ValueError:
                return None

        # 1. DOB validity check
        dob_field = fields.get("dob")
        dob_dt = None
        if dob_field and dob_field.value != "NOT_FOUND":
            dob_dt = parse_date(dob_field.value)
            if dob_dt:
                if dob_dt > today:
                    results.append(ValidationResult("FAIL", "dob_range", "DOB must be in the past", f"DOB: {dob_field.value} is in the future"))
                else:
                    age = today.year - dob_dt.year - ((today.month, today.day) < (dob_dt.month, dob_dt.day))
                    if age > 120:
                        results.append(ValidationResult("WARN", "dob_range", "Age should be realistic (< 120)", f"Age: {age} is abnormally high"))
                    else:
                        results.append(ValidationResult("PASS", "dob_range", "DOB is in the past", f"DOB is valid, age: {age}"))
            else:
                results.append(ValidationResult("FAIL", "dob_range", "DOB parsed as YYYY-MM-DD", f"Could not parse DOB: {dob_field.value}"))

        # 2. Expiry date validity check (for Passport and DL)
        if doc_type in [DocumentType.PASSPORT, DocumentType.DRIVING_LICENCE]:
            exp_field_name = "expiry" if doc_type == DocumentType.PASSPORT else "validity"
            exp_field = fields.get(exp_field_name)
            if exp_field and exp_field.value != "NOT_FOUND":
                exp_dt = parse_date(exp_field.value)
                if exp_dt:
                    if exp_dt < today:
                        results.append(ValidationResult("WARN", "expiry_range", "Document must not be expired", f"Expired on {exp_field.value}"))
                    else:
                        results.append(ValidationResult("PASS", "expiry_range", "Document is valid", f"Active till {exp_field.value}"))
                else:
                    results.append(ValidationResult("FAIL", "expiry_range", "Expiry parsed as YYYY-MM-DD", f"Could not parse Expiry: {exp_field.value}"))

        # 3. Age Restriction Check (Driving Licence minimum age of 18)
        if doc_type == DocumentType.DRIVING_LICENCE and dob_dt:
            age = today.year - dob_dt.year - ((today.month, today.day) < (dob_dt.month, dob_dt.day))
            if age < 18:
                results.append(ValidationResult("FAIL", "dl_min_age", "DL holder age >= 18", f"Age is {age} (Minor)"))
            else:
                results.append(ValidationResult("PASS", "dl_min_age", "DL holder age >= 18", f"Age is {age} (Adult)"))

        return results
