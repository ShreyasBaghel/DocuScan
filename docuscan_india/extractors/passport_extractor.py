import re
from typing import Dict, List, Any
from extractors.base_extractor import BaseExtractor
from utils.document_packet import FieldResult
from utils.string_utils import normalize_date, clean_whitespace

class PassportExtractor(BaseExtractor):
    def extract(self, raw_text: str, word_map: List[Dict[str, Any]]) -> Dict[str, FieldResult]:
        results: Dict[str, FieldResult] = {}
        
        # 1. Identify MRZ Lines
        mrz_line1 = ""
        mrz_line2 = ""
        
        # Clean lines
        lines = [line.strip().replace(" ", "") for line in raw_text.split('\n') if line.strip()]
        
        # Search for MRZ line 1 (starts with P< or P<IND)
        for idx, line in enumerate(lines):
            # Clean symbols except '<' and alphanumeric
            cleaned = re.sub(r'[^A-Z0-9<]', '', line.upper())
            if len(cleaned) >= 35 and cleaned.startswith("P<"):
                mrz_line1 = cleaned
                # The next line is typically MRZ Line 2
                if idx + 1 < len(lines):
                    next_cleaned = re.sub(r'[^A-Z0-9<]', '', lines[idx + 1].upper())
                    if len(next_cleaned) >= 35:
                        mrz_line2 = next_cleaned
                break

        # Standardize MRZ length to 44 if slightly shorter or longer due to OCR noise
        if mrz_line1:
            if len(mrz_line1) < 44:
                mrz_line1 = mrz_line1.ljust(44, "<")
            elif len(mrz_line1) > 44:
                mrz_line1 = mrz_line1[:44]
        if mrz_line2:
            if len(mrz_line2) < 44:
                mrz_line2 = mrz_line2.ljust(44, "<")
            elif len(mrz_line2) > 44:
                mrz_line2 = mrz_line2[:44]

        # Extract from MRZ if available
        if mrz_line1 and mrz_line2:
            results.update(self._extract_from_mrz(mrz_line1, mrz_line2, word_map))
        else:
            results.update(self._extract_from_visual(raw_text, lines, word_map))

        # Ensure all required fields exist in results
        required_fields = ["name", "passport_number", "nationality", "dob", "expiry", "mrz_line1", "mrz_line2"]
        for f in required_fields:
            if f not in results:
                results[f] = FieldResult(value="NOT_FOUND", raw_text="", confidence=0.0, bounding_box=None)

        return results

    def _extract_from_mrz(self, m1: str, m2: str, word_map: List[Dict[str, Any]]) -> Dict[str, FieldResult]:
        res = {}
        
        # Passport Number: chars 0-9 of Line 2
        pass_no_raw = m2[0:9].replace("<", "")
        # Nationality: chars 10-13 of Line 2
        nationality = m2[10:13]
        
        # DOB: chars 13-19 of Line 2 (YYMMDD)
        dob_yy = m2[13:15]
        dob_mm = m2[15:17]
        dob_dd = m2[17:19]
        # Pivot year: assume birth year is 19YY if YY > current_year, else 20YY
        curr_yy = 26 # (2026)
        prefix = "19" if int(dob_yy) > curr_yy else "20"
        dob = f"{prefix}{dob_yy}-{dob_mm}-{dob_dd}"

        # Expiry: chars 21-27 of Line 2 (YYMMDD)
        exp_yy = m2[21:23]
        exp_mm = m2[23:25]
        exp_dd = m2[25:27]
        expiry = f"20{exp_yy}-{exp_mm}-{exp_dd}"

        # Name: From line 1, after country code (chars 5-44)
        name_part = m1[5:]
        # Split by << to separate surname and given names
        parts = [p.replace("<", " ").strip() for p in name_part.split("<<") if p]
        if len(parts) >= 2:
            surname = parts[0]
            given_names = parts[1]
            full_name = f"{given_names} {surname}".strip()
        elif len(parts) == 1:
            full_name = parts[0]
        else:
            full_name = "UNKNOWN"

        # Bounding box maps
        pass_bbox = self.merge_bounding_boxes(pass_no_raw, word_map)
        name_bbox = self.merge_bounding_boxes(full_name, word_map)

        res["passport_number"] = FieldResult(value=pass_no_raw, raw_text=pass_no_raw, confidence=self.get_field_confidence(pass_no_raw, word_map), bounding_box=pass_bbox)
        res["nationality"] = FieldResult(value=nationality, raw_text=nationality, confidence=self.get_field_confidence(nationality, word_map), bounding_box=None)
        res["dob"] = FieldResult(value=dob, raw_text=m2[13:19], confidence=self.get_field_confidence(dob, word_map), bounding_box=None)
        res["expiry"] = FieldResult(value=expiry, raw_text=m2[21:27], confidence=self.get_field_confidence(expiry, word_map), bounding_box=None)
        res["name"] = FieldResult(value=full_name, raw_text=name_part, confidence=self.get_field_confidence(full_name, word_map), bounding_box=name_bbox)
        res["mrz_line1"] = FieldResult(value=m1, raw_text=m1, confidence=self.get_field_confidence(m1, word_map), bounding_box=None)
        res["mrz_line2"] = FieldResult(value=m2, raw_text=m2, confidence=self.get_field_confidence(m2, word_map), bounding_box=None)

        return res

    def _extract_from_visual(self, raw_text: str, lines: List[str], word_map: List[Dict[str, Any]]) -> Dict[str, FieldResult]:
        res = {}
        
        # Passport Number
        m_pass = re.search(r"\b([A-Z][0-9]{7})\b", raw_text)
        if m_pass:
            pass_val = m_pass.group(1)
            bbox = self.merge_bounding_boxes(pass_val, word_map)
            res["passport_number"] = FieldResult(value=pass_val, raw_text=m_pass.group(0), confidence=self.get_field_confidence(pass_val, word_map), bounding_box=bbox)
        
        # Nationality (Usually "INDIAN" or "REPUBLIC OF INDIA")
        m_nat = re.search(r"\b(indian|republic\s+of\s+india)\b", raw_text, re.IGNORECASE)
        if m_nat:
            res["nationality"] = FieldResult(value="IND", raw_text=m_nat.group(0), confidence=self.get_field_confidence("IND", word_map), bounding_box=None)
        else:
            res["nationality"] = FieldResult(value="IND", raw_text="INDIAN", confidence=self.get_field_confidence("IND", word_map), bounding_box=None)

        # Dates (DOB and Expiry)
        # Search for dates. Passport page 1 has: Date of Birth, Date of Expiry, Date of Issue.
        dates = []
        for line in raw_text.split('\n'):
            norm = normalize_date(line)
            if norm:
                dates.append((norm, line))

        # Usually, DOB is the oldest date, and Expiry is the furthest future date
        if dates:
            sorted_dates = sorted(dates, key=lambda x: x[0])
            dob_val, dob_raw = sorted_dates[0]
            res["dob"] = FieldResult(value=dob_val, raw_text=dob_raw, confidence=self.get_field_confidence(dob_val, word_map), bounding_box=self.merge_bounding_boxes(dob_raw, word_map))
            if len(sorted_dates) > 1:
                exp_val, exp_raw = sorted_dates[-1]
                res["expiry"] = FieldResult(value=exp_val, raw_text=exp_raw, confidence=self.get_field_confidence(exp_val, word_map), bounding_box=self.merge_bounding_boxes(exp_raw, word_map))

        # Name: Look for surname and given names labels
        # Standard Visual Zone labels: "Given Name(s)", "Surname"
        name_val = "NOT_FOUND"
        name_raw = ""
        for idx, line in enumerate(lines):
            if "givenname" in line.lower() and idx + 1 < len(lines):
                name_val = lines[idx + 1]
                name_raw = name_val
                # If Surname is above it or around, we can try to merge, but we keep it simple
                break

        if name_val != "NOT_FOUND":
            res["name"] = FieldResult(value=name_val, raw_text=name_raw, confidence=self.get_field_confidence(name_val, word_map), bounding_box=self.merge_bounding_boxes(name_raw, word_map))

        return res
