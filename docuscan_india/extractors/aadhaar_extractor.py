import re
from typing import Dict, List, Any
from extractors.base_extractor import BaseExtractor
from utils.document_packet import FieldResult
from utils.string_utils import normalize_date, clean_whitespace

class AadhaarExtractor(BaseExtractor):
    def extract(self, raw_text: str, word_map: List[Dict[str, Any]]) -> Dict[str, FieldResult]:
        results: Dict[str, FieldResult] = {}
        lines = [line.strip() for line in raw_text.split('\n') if line.strip()]

        # 1. Aadhaar Number Extraction
        aadhaar_num = "NOT_FOUND"
        aadhaar_raw = ""
        # Look for 12 digits (with spaces or consecutive)
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
            results["aadhaar_number"] = FieldResult(value=aadhaar_num, raw_text=aadhaar_raw, confidence=0.95, bounding_box=bbox)
        else:
            results["aadhaar_number"] = FieldResult(value="NOT_FOUND", raw_text="", confidence=0.0, bounding_box=None)

        # 2. DOB Extraction
        dob_val = "NOT_FOUND"
        dob_raw = ""
        # Patterns for DOB
        dob_pattern = r"(?:dob|yob|birth|जन्म|तिथि|वर्ष)\s*[:\-]?\s*([0-9/\-\.\s]{4,10})"
        m_dob = re.search(dob_pattern, raw_text, re.IGNORECASE)
        if m_dob:
            dob_raw = m_dob.group(0)
            norm_date = normalize_date(m_dob.group(1))
            if norm_date:
                dob_val = norm_date
            else:
                # If only Year is present (common in old Aadhaar cards, e.g. YOB: 1985)
                y_match = re.search(r"\b(19\d{2}|20\d{2})\b", m_dob.group(1))
                if y_match:
                    dob_val = f"{y_match.group(1)}-01-01"  # Default to Jan 1st of YOB

        if dob_val != "NOT_FOUND":
            bbox = self.merge_bounding_boxes(dob_raw, word_map)
            results["dob"] = FieldResult(value=dob_val, raw_text=dob_raw, confidence=0.90, bounding_box=bbox)
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
            results["gender"] = FieldResult(value=gender_val, raw_text=gender_raw, confidence=0.95, bounding_box=bbox)
        else:
            results["gender"] = FieldResult(value="NOT_FOUND", raw_text="", confidence=0.0, bounding_box=None)

        # 4. Name Extraction
        # Strategy: Look above the DOB line for a line containing standard uppercase letters
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
                # Filter out lines containing common headers
                if any(h in line.lower() for h in ["government", "india", "uidai", "unique", "enrollment"]):
                    continue
                # Clean line (remove symbols, digits, hindi characters if we want English Name)
                clean_line = re.sub(r'[^a-zA-Z\s\.]', '', line).strip()
                # English name should typically be uppercase words of length >= 3
                if len(clean_line) >= 3 and clean_line.isupper():
                    name_val = clean_line
                    name_raw = line
                    break

        # Fallback if upwards scan failed: look for uppercase words of length >= 5
        if name_val == "NOT_FOUND":
            for line in lines:
                if any(h in line.lower() for h in ["government", "india", "uidai", "unique"]):
                    continue
                clean_line = re.sub(r'[^a-zA-Z\s\.]', '', line).strip()
                if len(clean_line) >= 5 and clean_line.isupper() and "MALE" not in clean_line and "FEMALE" not in clean_line:
                    name_val = clean_line
                    name_raw = line
                    break

        if name_val != "NOT_FOUND":
            bbox = self.merge_bounding_boxes(name_raw, word_map)
            results["name"] = FieldResult(value=name_val, raw_text=name_raw, confidence=0.85, bounding_box=bbox)
        else:
            results["name"] = FieldResult(value="NOT_FOUND", raw_text="", confidence=0.0, bounding_box=None)

        return results
