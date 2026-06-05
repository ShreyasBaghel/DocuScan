import re
from typing import List, Dict
from utils.document_packet import ValidationResult, FieldResult, DocumentType

class FormatValidator:
    @staticmethod
    def validate(doc_type: DocumentType, fields: Dict[str, FieldResult]) -> List[ValidationResult]:
        results: List[ValidationResult] = []

        # Validate Date format helper
        def check_date_format(field_name: str):
            res_field = fields.get(field_name)
            if not res_field or res_field.value == "NOT_FOUND":
                results.append(ValidationResult("FAIL", field_name, "YYYY-MM-DD", "Missing field"))
                return
            val = res_field.value
            if re.match(r"^\d{4}-\d{2}-\d{2}$", val):
                results.append(ValidationResult("PASS", field_name, "YYYY-MM-DD", val))
            else:
                results.append(ValidationResult("FAIL", field_name, "YYYY-MM-DD", val))

        # Check Name helper
        def check_name_format(field_name: str):
            res_field = fields.get(field_name)
            if not res_field or res_field.value == "NOT_FOUND":
                results.append(ValidationResult("FAIL", field_name, "Uppercase Alphabetic", "Missing field"))
                return
            val = res_field.value
            # Name should contain alphabetic characters, spaces, and dots
            if re.match(r"^[A-Z\s\.]+$", val):
                results.append(ValidationResult("PASS", field_name, "Uppercase Alphabetic", val))
            else:
                results.append(ValidationResult("WARN", field_name, "Uppercase Alphabetic", f"Contains lowercase or symbols: {val}"))

        if doc_type == DocumentType.AADHAAR:
            # 1. Aadhaar number format (12 digits)
            a_num = fields.get("aadhaar_number")
            if a_num and a_num.value != "NOT_FOUND":
                val = a_num.value
                if re.match(r"^\d{12}$", val):
                    results.append(ValidationResult("PASS", "aadhaar_number_format", "12 digits", val))
                else:
                    results.append(ValidationResult("FAIL", "aadhaar_number_format", "12 digits", val))
            else:
                results.append(ValidationResult("FAIL", "aadhaar_number_format", "12 digits", "Missing field"))

            # 2. DOB format
            check_date_format("dob")

            # 3. Gender format (MALE/FEMALE/OTHER)
            gender = fields.get("gender")
            if gender and gender.value != "NOT_FOUND":
                val = gender.value
                if val in ["MALE", "FEMALE", "OTHER"]:
                    results.append(ValidationResult("PASS", "gender_format", "MALE/FEMALE/OTHER", val))
                else:
                    results.append(ValidationResult("FAIL", "gender_format", "MALE/FEMALE/OTHER", val))
            else:
                results.append(ValidationResult("FAIL", "gender_format", "MALE/FEMALE/OTHER", "Missing field"))

            # 4. Name format
            check_name_format("name")

        elif doc_type == DocumentType.PAN:
            # 1. PAN format
            pan_num = fields.get("pan_number")
            if pan_num and pan_num.value != "NOT_FOUND":
                val = pan_num.value
                if re.match(r"^[A-Z]{5}[0-9]{4}[A-Z]$", val):
                    results.append(ValidationResult("PASS", "pan_format", "5 Letters, 4 Digits, 1 Letter", val))
                else:
                    results.append(ValidationResult("FAIL", "pan_format", "5 Letters, 4 Digits, 1 Letter", val))
            else:
                results.append(ValidationResult("FAIL", "pan_format", "5 Letters, 4 Digits, 1 Letter", "Missing field"))

            # 2. DOB format
            check_date_format("dob")

            # 3. Name format
            check_name_format("name")

            # 4. Father's Name format
            check_name_format("father_name")

        elif doc_type == DocumentType.PASSPORT:
            # 1. Passport format
            p_num = fields.get("passport_number")
            if p_num and p_num.value != "NOT_FOUND":
                val = p_num.value
                if re.match(r"^[A-Z][0-9]{7}$", val):
                    results.append(ValidationResult("PASS", "passport_format", "1 Letter, 7 Digits", val))
                else:
                    results.append(ValidationResult("FAIL", "passport_format", "1 Letter, 7 Digits", val))
            else:
                results.append(ValidationResult("FAIL", "passport_format", "1 Letter, 7 Digits", "Missing field"))

            # 2. DOB format
            check_date_format("dob")

            # 3. Expiry format
            check_date_format("expiry")

            # 4. Name format
            check_name_format("name")

        elif doc_type == DocumentType.DRIVING_LICENCE:
            # 1. DL format (MH0320140123456 - state, rto, year, index = 15 chars)
            dl_num = fields.get("dl_number")
            if dl_num and dl_num.value != "NOT_FOUND":
                val = dl_num.value
                # DL format can have minor variances but standard is 15-16 characters
                if re.match(r"^[A-Z]{2}[0-9]{2}[0-9]{4}[0-9]{7}$", val):
                    results.append(ValidationResult("PASS", "dl_format", "State+RTO+Year+7 Digits", val))
                elif len(val) >= 10:
                    results.append(ValidationResult("WARN", "dl_format", "State+RTO+Year+7 Digits", f"Alternative format: {val}"))
                else:
                    results.append(ValidationResult("FAIL", "dl_format", "State+RTO+Year+7 Digits", val))
            else:
                results.append(ValidationResult("FAIL", "dl_format", "State+RTO+Year+7 Digits", "Missing field"))

            # 2. DOB format
            check_date_format("dob")

            # 3. Validity format
            check_date_format("validity")

            # 4. Name format
            check_name_format("name")

        return results
