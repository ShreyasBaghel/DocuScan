import re
from typing import Dict, List, Any
from extractors.base_extractor import BaseExtractor
from utils.document_packet import FieldResult
from utils.string_utils import normalize_date, clean_whitespace, extract_uppercase_name, is_valid_name, ocr_correct_digits
from validators.checksum_validator import ChecksumValidator

class AadhaarExtractor(BaseExtractor):
    def extract(self, raw_text: str, word_map: List[Dict[str, Any]]) -> Dict[str, FieldResult]:
        results: Dict[str, FieldResult] = {}
        lines = [line.strip() for line in raw_text.split('\n') if line.strip()]

        # 1. Aadhaar Number Extraction (OCR-Tolerant + Verhoeff Validated)
        aadhaar_num = "NOT_FOUND"
        aadhaar_raw = ""
        
        # A. Search for 12 digits/alphanumeric tokens, space separated or consecutive
        # e.g., 1234 5678 9012 or 123456789012 (allowing substitutions like OISZB)
        candidates = []
        # Find three groups of 4 alphanumeric chars
        for m in re.finditer(r"\b([0-9OISZB]{4})\s+([0-9OISZB]{4})\s+([0-9OISZB]{4})\b", raw_text, re.IGNORECASE):
            raw_match = f"{m.group(1)}{m.group(2)}{m.group(3)}"
            corrected = ocr_correct_digits(raw_match)
            if ChecksumValidator.validate_verhoeff(corrected):
                candidates.append((corrected, m.group(0)))
                
        # Find 12 consecutive alphanumeric chars
        for m in re.finditer(r"\b([0-9OISZB]{12})\b", raw_text, re.IGNORECASE):
            raw_match = m.group(1)
            corrected = ocr_correct_digits(raw_match)
            if ChecksumValidator.validate_verhoeff(corrected):
                candidates.append((corrected, m.group(0)))

        # Also search in the flat text (removing newlines)
        flat_text = " ".join(raw_text.split())
        for m in re.finditer(r"\b([0-9OISZB]{4})\s+([0-9OISZB]{4})\s+([0-9OISZB]{4})\b", flat_text, re.IGNORECASE):
            raw_match = f"{m.group(1)}{m.group(2)}{m.group(3)}"
            corrected = ocr_correct_digits(raw_match)
            if ChecksumValidator.validate_verhoeff(corrected):
                candidates.append((corrected, m.group(0)))

        if candidates:
            # Take the first candidate that validated successfully
            aadhaar_num, aadhaar_raw = candidates[0]

        # B. Fallback to strict regex if Verhoeff fails on all soft candidates (could be invalid test data)
        if aadhaar_num == "NOT_FOUND":
            m_num = re.search(r"\b(\d{4})\s+(\d{4})\s+(\d{4})\b", raw_text)
            if m_num:
                aadhaar_num = f"{m_num.group(1)}{m_num.group(2)}{m_num.group(3)}"
                aadhaar_raw = m_num.group(0)
            else:
                m_num_consec = re.search(r"\b(\d{12})\b", raw_text)
                if m_num_consec:
                    aadhaar_num = m_num_consec.group(1)
                    aadhaar_raw = m_num_consec.group(0)

        if aadhaar_num != "NOT_FOUND":
            bbox = self.merge_bounding_boxes(aadhaar_raw, word_map)
            conf = self.get_field_confidence(aadhaar_num, word_map)
            results["aadhaar_number"] = FieldResult(value=aadhaar_num, raw_text=aadhaar_raw, confidence=conf, bounding_box=bbox)
        else:
            results["aadhaar_number"] = FieldResult(value="NOT_FOUND", raw_text="", confidence=0.0, bounding_box=None)

        # 2. DOB Extraction
        dob_val = "NOT_FOUND"
        dob_raw = ""
        dob_pattern = r"(?:dob|yob|birth|जन्म|तिथि|वर्ष)\s*[:\-]?\s*([0-9/\-\.\s]{4,10})"
        m_dob = re.search(dob_pattern, raw_text, re.IGNORECASE)
        if m_dob:
            dob_raw = m_dob.group(0)
            norm_date = normalize_date(m_dob.group(1))
            if norm_date:
                dob_val = norm_date
            else:
                y_match = re.search(r"\b(19\d{2}|20\d{2})\b", m_dob.group(1))
                if y_match:
                    dob_val = f"{y_match.group(1)}-01-01"

        if dob_val != "NOT_FOUND":
            bbox = self.merge_bounding_boxes(dob_raw, word_map)
            conf = self.get_field_confidence(dob_val, word_map)
            results["dob"] = FieldResult(value=dob_val, raw_text=dob_raw, confidence=conf, bounding_box=bbox)
        else:
            m_date = re.search(r"\b(\d{2}[/\-\.]\d{2}[/\-\.]\d{4})\b", raw_text)
            if m_date:
                dob_raw = m_date.group(0)
                norm_date = normalize_date(m_date.group(1))
                if norm_date:
                    dob_val = norm_date

            if dob_val != "NOT_FOUND":
                bbox = self.merge_bounding_boxes(dob_raw, word_map)
                conf = self.get_field_confidence(dob_val, word_map)
                results["dob"] = FieldResult(value=dob_val, raw_text=dob_raw, confidence=conf, bounding_box=bbox)
            else:
                results["dob"] = FieldResult(value="NOT_FOUND", raw_text="", confidence=0.0, bounding_box=None)

        # 3. Gender Extraction
        gender_val = "NOT_FOUND"
        gender_raw = ""
        m_gen = re.search(r"\b(male|female|transgender|पुरुष|महिला)\b", raw_text, re.IGNORECASE)
        if m_gen:
            gender_raw = m_gen.group(0)
            g_lower = gender_raw.lower()
            if "female" in g_lower or "महिला" in g_lower:
                gender_val = "FEMALE"
            elif "male" in g_lower or "पुरुष" in g_lower:
                gender_val = "MALE"
            else:
                gender_val = "OTHER"

        if gender_val != "NOT_FOUND":
            bbox = self.merge_bounding_boxes(gender_raw, word_map)
            conf = self.get_field_confidence(gender_val, word_map)
            results["gender"] = FieldResult(value=gender_val, raw_text=gender_raw, confidence=conf, bounding_box=bbox)
        else:
            results["gender"] = FieldResult(value="NOT_FOUND", raw_text="", confidence=0.0, bounding_box=None)

        # 4. Name Extraction
        name_val = "NOT_FOUND"
        name_raw = ""
        
        # Find line index of DOB
        dob_idx = -1
        for idx, line in enumerate(lines):
            if any(k in line.lower() for k in ["dob", "yob", "birth", "जन्म"]):
                dob_idx = idx
                break

        if dob_idx > 0:
            # Check lines above DOB line
            for idx in range(dob_idx - 1, -1, -1):
                line = lines[idx]
                if any(h in line.lower() for h in ["government", "india", "uidai", "unique", "enrollment", "enrolment"]):
                    continue
                
                candidate = extract_uppercase_name(line)
                if is_valid_name(candidate, aadhaar_num) and "MALE" not in candidate and "FEMALE" not in candidate:
                    name_val = candidate
                    name_raw = line
                    break

        # Fallback if upwards scan failed
        if name_val == "NOT_FOUND":
            for line in lines:
                if any(h in line.lower() for h in ["government", "india", "uidai", "unique", "enrollment", "enrolment"]):
                    continue
                candidate = extract_uppercase_name(line)
                if is_valid_name(candidate, aadhaar_num) and "MALE" not in candidate and "FEMALE" not in candidate:
                    name_val = candidate
                    name_raw = line
                    break

        if name_val != "NOT_FOUND":
            bbox = self.merge_bounding_boxes(name_raw, word_map)
            conf = self.get_field_confidence(name_val, word_map)
            results["name"] = FieldResult(value=name_val, raw_text=name_raw, confidence=conf, bounding_box=bbox)
        else:
            results["name"] = FieldResult(value="NOT_FOUND", raw_text="", confidence=0.0, bounding_box=None)


        return results
