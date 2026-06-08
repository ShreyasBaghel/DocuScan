import re
from typing import Dict, List, Any
from extractors.base_extractor import BaseExtractor
from utils.document_packet import FieldResult
from utils.string_utils import normalize_date, clean_whitespace

class PANExtractor(BaseExtractor):
    def extract(self, raw_text: str, word_map: List[Dict[str, Any]]) -> Dict[str, FieldResult]:
        results: Dict[str, FieldResult] = {}
        lines = [line.strip() for line in raw_text.split('\n') if line.strip()]

        # 1. PAN Number Extraction
        pan_num = "NOT_FOUND"
        pan_raw = ""
        m_pan = re.search(r"(?:[^A-Z0-9]|^)([A-Z]{5}[0-9]{4}[A-Z])(?:[^A-Z0-9]|$)", raw_text)
        if m_pan:
            pan_num = m_pan.group(1)
            pan_raw = m_pan.group(0)

        if pan_num != "NOT_FOUND":
            bbox = self.merge_bounding_boxes(pan_raw, word_map)
            results["pan_number"] = FieldResult(value=pan_num, raw_text=pan_raw, confidence=0.98, bounding_box=bbox)
        else:
            results["pan_number"] = FieldResult(value="NOT_FOUND", raw_text="", confidence=0.0, bounding_box=None)

        # 2. Extract Name, Father's Name, DOB by label reference
        name_val = "NOT_FOUND"
        name_raw = ""
        father_val = "NOT_FOUND"
        father_raw = ""
        dob_val = "NOT_FOUND"
        dob_raw = ""

        # Scan lines to find label indexes
        name_label_idx = -1
        father_label_idx = -1
        dob_label_idx = -1

        for idx, line in enumerate(lines):
            l_lower = line.lower()
            # Name label (should not match Father's Name)
            if ("name" in l_lower or "नाम" in l_lower) and "father" not in l_lower and "पिता" not in l_lower:
                if name_label_idx == -1:
                    name_label_idx = idx
            # Father's Name label
            if "father" in l_lower or "पिता" in l_lower:
                father_label_idx = idx
            # DOB label
            if "birth" in l_lower or "जन्म" in l_lower or "date of" in l_lower:
                dob_label_idx = idx

        # Extract values (usually on the next line or two)
        if name_label_idx != -1 and name_label_idx + 1 < len(lines):
            val_line = lines[name_label_idx + 1]
            # Ensure it is a valid name format (mostly uppercase text)
            clean_val = re.sub(r'[^a-zA-Z\s\.]', '', val_line).strip()
            if len(clean_val) >= 3 and clean_val.isupper():
                name_val = clean_val
                name_raw = val_line

        if father_label_idx != -1 and father_label_idx + 1 < len(lines):
            val_line = lines[father_label_idx + 1]
            clean_val = re.sub(r'[^a-zA-Z\s\.]', '', val_line).strip()
            if len(clean_val) >= 3 and clean_val.isupper():
                father_val = clean_val
                father_raw = val_line

        if dob_label_idx != -1 and dob_label_idx + 1 < len(lines):
            val_line = lines[dob_label_idx + 1]
            norm_date = normalize_date(val_line)
            if norm_date:
                dob_val = norm_date
                dob_raw = val_line

        # Fallback date scanning in case DOB label was not found
        if dob_val == "NOT_FOUND":
            # Search for any date in YYYY-MM-DD or DD/MM/YYYY formats
            m_date = re.search(r"\b(\d{2}[/\-\.]\d{2}[/\-\.]\d{4})\b", raw_text)
            if m_date:
                norm_date = normalize_date(m_date.group(1))
                if norm_date:
                    dob_val = norm_date
                    dob_raw = m_date.group(0)

        # Fallback Name and Father's Name scanning in case labels were missed
        # Standard PAN card structure before 2018:
        # Line 1: Income Tax Department
        # Line 2: Name
        # Line 3: Father's Name
        # Line 4: DOB
        if name_val == "NOT_FOUND" or father_val == "NOT_FOUND":
            uppercase_lines = []
            for line in lines:
                if any(h in line.lower() for h in ["income", "tax", "department", "permanent", "account", "card", "govt", "india"]):
                    continue
                # Skip lines containing numbers (such as PAN number or date) to prevent false name extraction
                if sum(c.isdigit() for c in line) >= 3:
                    continue
                clean_line = re.sub(r'[^a-zA-Z\s\.]', '', line).strip()
                if len(clean_line) >= 4 and clean_line.isupper():
                    uppercase_lines.append((line, clean_line))

            if len(uppercase_lines) >= 2:
                if name_val == "NOT_FOUND":
                    name_raw, name_val = uppercase_lines[0]
                if father_val == "NOT_FOUND" and len(uppercase_lines) > 1:
                    father_raw, father_val = uppercase_lines[1]

        # Write results
        if name_val != "NOT_FOUND":
            bbox = self.merge_bounding_boxes(name_raw, word_map)
            results["name"] = FieldResult(value=name_val, raw_text=name_raw, confidence=0.88, bounding_box=bbox)
        else:
            results["name"] = FieldResult(value="NOT_FOUND", raw_text="", confidence=0.0, bounding_box=None)

        if father_val != "NOT_FOUND":
            bbox = self.merge_bounding_boxes(father_raw, word_map)
            results["father_name"] = FieldResult(value=father_val, raw_text=father_raw, confidence=0.85, bounding_box=bbox)
        else:
            results["father_name"] = FieldResult(value="NOT_FOUND", raw_text="", confidence=0.0, bounding_box=None)

        if dob_val != "NOT_FOUND":
            bbox = self.merge_bounding_boxes(dob_raw, word_map)
            results["dob"] = FieldResult(value=dob_val, raw_text=dob_raw, confidence=0.92, bounding_box=bbox)
        else:
            results["dob"] = FieldResult(value="NOT_FOUND", raw_text="", confidence=0.0, bounding_box=None)

        return results
