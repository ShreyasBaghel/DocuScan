import re
from typing import Dict, List, Any
from extractors.base_extractor import BaseExtractor
from utils.document_packet import FieldResult
from utils.string_utils import normalize_date, clean_whitespace

class DLExtractor(BaseExtractor):
    def extract(self, raw_text: str, word_map: List[Dict[str, Any]]) -> Dict[str, FieldResult]:
        results: Dict[str, FieldResult] = {}
        lines = [line.strip() for line in raw_text.split('\n') if line.strip()]

        # 1. DL Number Extraction
        dl_num = "NOT_FOUND"
        dl_raw = ""
        # Match State + RTO + Year + 7 digits (with optional spaces/dashes)
        m_dl = re.search(r"\b([A-Z]{2})\s*[-/]?\s*([0-9]{2})\s*[-/]?\s*([0-9]{4})\s*[-/]?\s*([0-9]{7})\b", raw_text, re.IGNORECASE)
        if m_dl:
            dl_num = f"{m_dl.group(1).upper()}{m_dl.group(2)}{m_dl.group(3)}{m_dl.group(4)}"
            dl_raw = m_dl.group(0)
        else:
            # Fallback check for any state code followed by 13 digits
            m_dl_consec = re.search(r"\b([A-Z]{2}[0-9]{13})\b", raw_text, re.IGNORECASE)
            if m_dl_consec:
                dl_num = m_dl_consec.group(1).upper()
                dl_raw = m_dl_consec.group(0)

        if dl_num != "NOT_FOUND":
            bbox = self.merge_bounding_boxes(dl_raw, word_map)
            results["dl_number"] = FieldResult(value=dl_num, raw_text=dl_raw, confidence=0.95, bounding_box=bbox)
        else:
            results["dl_number"] = FieldResult(value="NOT_FOUND", raw_text="", confidence=0.0, bounding_box=None)

        # 2. DOB Extraction
        dob_val = "NOT_FOUND"
        dob_raw = ""
        # Patterns for DOB in DL
        m_dob = re.search(r"(?:dob|d\.o\.b|birth|जन्म)\s*[:\-]?\s*([0-9/\-\.]{10})", raw_text, re.IGNORECASE)
        if m_dob:
            dob_raw = m_dob.group(0)
            norm = normalize_date(m_dob.group(1))
            if norm:
                dob_val = norm
        else:
            # Fallback to search any date which is NOT validity date
            # We will try to find dates in text
            dates = re.findall(r"\b(\d{2}[/\-\.]\d{2}[/\-\.]\d{4})\b", raw_text)
            # Usually DOB is the oldest date on the DL
            normalized_dates = []
            for d in dates:
                n = normalize_date(d)
                if n:
                    normalized_dates.append((n, d))
            if normalized_dates:
                normalized_dates.sort(key=lambda x: x[0])
                dob_val, dob_raw = normalized_dates[0]

        if dob_val != "NOT_FOUND":
            bbox = self.merge_bounding_boxes(dob_raw, word_map)
            results["dob"] = FieldResult(value=dob_val, raw_text=dob_raw, confidence=0.90, bounding_box=bbox)
        else:
            results["dob"] = FieldResult(value="NOT_FOUND", raw_text="", confidence=0.0, bounding_box=None)

        # 3. Validity Extraction (Expiry)
        validity_val = "NOT_FOUND"
        validity_raw = ""
        # Driving licences usually have validity ranges, e.g., "Valid Till: DD-MM-YYYY" or "NT: DD-MM-YYYY"
        m_val = re.search(r"(?:valid|till|expiry|nt|validity)\s*[:\-]?\s*([0-9/\-\.]{10})", raw_text, re.IGNORECASE)
        if m_val:
            validity_raw = m_val.group(0)
            norm = normalize_date(m_val.group(1))
            if norm:
                validity_val = norm
        else:
            # Fallback: get the newest date (validity/expiry date is in the future relative to issue/birth)
            dates = re.findall(r"\b(\d{2}[/\-\.]\d{2}[/\-\.]\d{4})\b", raw_text)
            normalized_dates = []
            for d in dates:
                n = normalize_date(d)
                if n:
                    normalized_dates.append((n, d))
            if len(normalized_dates) >= 2:
                normalized_dates.sort(key=lambda x: x[0])
                # Expiry date is the latest date
                validity_val, validity_raw = normalized_dates[-1]

        if validity_val != "NOT_FOUND":
            bbox = self.merge_bounding_boxes(validity_raw, word_map)
            results["validity"] = FieldResult(value=validity_val, raw_text=validity_raw, confidence=0.88, bounding_box=bbox)
        else:
            results["validity"] = FieldResult(value="NOT_FOUND", raw_text="", confidence=0.0, bounding_box=None)

        # 4. Name Extraction
        name_val = "NOT_FOUND"
        name_raw = ""
        
        # Search for Name label line
        name_lbl_idx = -1
        for idx, line in enumerate(lines):
            l_lower = line.lower()
            if "name" in l_lower and "father" not in l_lower and "licence" not in l_lower:
                name_lbl_idx = idx
                break
        
        if name_lbl_idx != -1:
            line = lines[name_lbl_idx]
            # Try to match name value inline (e.g. "NAME: KARTIK KAPOOR" or "NAME - KARTIK KAPOOR")
            match_name_inline = re.search(r"(?:name|नाम)\s*[:\-]\s*([a-zA-Z\s\.]+)", line, re.IGNORECASE)
            if match_name_inline and len(match_name_inline.group(1).strip()) >= 3:
                name_val = re.sub(r'[^a-zA-Z\s\.]', '', match_name_inline.group(1)).strip().upper()
                name_raw = line
            elif name_lbl_idx + 1 < len(lines):
                val_line = lines[name_lbl_idx + 1]
                clean_val = re.sub(r'[^a-zA-Z\s\.]', '', val_line).strip()
                if len(clean_val) >= 4 and clean_val.isupper():
                    name_val = clean_val
                    name_raw = val_line

        # Fallback Name check: look for uppercase lines that are not headers or numbers
        if name_val == "NOT_FOUND":
            for line in lines:
                if any(h in line.lower() for h in ["driving", "licence", "license", "authority", "union", "india", "transport", "dl no", "lic no", "licence no", "lic.no"]):
                    continue
                clean_line = re.sub(r'[^a-zA-Z\s\.]', '', line).strip()
                # Check for 2 or more words in uppercase
                if len(clean_line) >= 6 and clean_line.isupper() and len(clean_line.split()) >= 2:
                    name_val = clean_line
                    name_raw = line
                    break

        if name_val != "NOT_FOUND":
            bbox = self.merge_bounding_boxes(name_raw, word_map)
            results["name"] = FieldResult(value=name_val, raw_text=name_raw, confidence=0.85, bounding_box=bbox)
        else:
            results["name"] = FieldResult(value="NOT_FOUND", raw_text="", confidence=0.0, bounding_box=None)

        # 5. Vehicle Class Extraction
        classes = []
        for w in word_map:
            text = w['text'].upper()
            if text in ["MCWG", "LMV", "MCWOG", "HMV", "TRANS", "COV"]:
                classes.append(text)
        
        classes = list(set(classes))
        class_val = ", ".join(classes) if classes else "NOT_FOUND"

        if class_val != "NOT_FOUND":
            bbox = self.merge_bounding_boxes(classes[0], word_map) if classes else None
            results["vehicle_class"] = FieldResult(value=class_val, raw_text=class_val, confidence=0.90, bounding_box=bbox)
        else:
            results["vehicle_class"] = FieldResult(value="NOT_FOUND", raw_text="", confidence=0.0, bounding_box=None)

        return results
