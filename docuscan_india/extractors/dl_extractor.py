import re
from typing import Dict, List, Any
from extractors.base_extractor import BaseExtractor
from utils.document_packet import FieldResult
from utils.string_utils import normalize_date, clean_whitespace, extract_uppercase_name, is_valid_name, ocr_correct_digits

class DLExtractor(BaseExtractor):
    def extract(self, raw_text: str, word_map: List[Dict[str, Any]]) -> Dict[str, FieldResult]:
        results: Dict[str, FieldResult] = {}
        lines = [line.strip() for line in raw_text.split('\n') if line.strip()]

        # 1. DL Number Extraction (OCR-Tolerant)
        dl_num = "NOT_FOUND"
        dl_raw = ""
        
        # A. Try strict regex first
        # MH-03-2014-0123456 or MH0320140123456
        m_dl = re.search(r"\b([A-Z]{2})\s*[-/]?\s*([0-9]{2})\s*[-/]?\s*([0-9]{4})\s*[-/]?\s*([0-9]{7})\b", raw_text, re.IGNORECASE)
        if m_dl:
            dl_num = f"{m_dl.group(1).upper()}{m_dl.group(2)}{m_dl.group(3)}{m_dl.group(4)}"
            dl_raw = m_dl.group(0)
        else:
            # B. Try soft regex allowing OCR substitutions
            m_dl_soft = re.search(
                r"\b([A-Z]{2})\s*[-/]?\s*([0-9OISZB]{2})\s*[-/]?\s*([0-9OISZB]{4})\s*[-/]?\s*([0-9OISZB]{7})\b",
                raw_text,
                re.IGNORECASE
            )
            if m_dl_soft:
                raw_rto = ocr_correct_digits(m_dl_soft.group(2))
                raw_year = ocr_correct_digits(m_dl_soft.group(3))
                raw_idx = ocr_correct_digits(m_dl_soft.group(4))
                corrected = f"{m_dl_soft.group(1).upper()}{raw_rto}{raw_year}{raw_idx}"
                if len(corrected) == 15:
                    dl_num = corrected
                    dl_raw = m_dl_soft.group(0)
            else:
                # C. Check consecutive 15 chars
                m_dl_consec = re.search(r"\b([A-Z]{2}[0-9OISZB]{13})\b", raw_text, re.IGNORECASE)
                if m_dl_consec:
                    raw_val = m_dl_consec.group(1)
                    corrected = raw_val[:2].upper() + ocr_correct_digits(raw_val[2:])
                    if len(corrected) == 15:
                        dl_num = corrected
                        dl_raw = m_dl_consec.group(0)

        if dl_num != "NOT_FOUND":
            bbox = self.merge_bounding_boxes(dl_raw, word_map)
            results["dl_number"] = FieldResult(value=dl_num, raw_text=dl_raw, confidence=0.95, bounding_box=bbox)
        else:
            results["dl_number"] = FieldResult(value="NOT_FOUND", raw_text="", confidence=0.0, bounding_box=None)

        # 2. DOB Extraction
        dob_val = "NOT_FOUND"
        dob_raw = ""
        m_dob = re.search(r"(?:dob|d\.o\.b|birth|जन्म)\s*[:\-]?\s*([0-9/\-\.]{10})", raw_text, re.IGNORECASE)
        if m_dob:
            dob_raw = m_dob.group(0)
            norm = normalize_date(m_dob.group(1))
            if norm:
                dob_val = norm
        else:
            dates = re.findall(r"\b(\d{2}[/\-\.]\d{2}[/\-\.]\d{4})\b", raw_text)
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
        m_val = re.search(r"(?:valid|till|expiry|nt|validity)\s*[:\-]?\s*([0-9/\-\.]{10})", raw_text, re.IGNORECASE)
        if m_val:
            validity_raw = m_val.group(0)
            norm = normalize_date(m_val.group(1))
            if norm:
                validity_val = norm
        else:
            dates = re.findall(r"\b(\d{2}[/\-\.]\d{2}[/\-\.]\d{4})\b", raw_text)
            normalized_dates = []
            for d in dates:
                n = normalize_date(d)
                if n:
                    normalized_dates.append((n, d))
            if len(normalized_dates) >= 2:
                normalized_dates.sort(key=lambda x: x[0])
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
            match_name_inline = re.search(r"(?:name|नाम)\s*[:\-]\s*([a-zA-Z\s\.]+)", line, re.IGNORECASE)
            if match_name_inline:
                candidate = extract_uppercase_name(match_name_inline.group(1))
                if is_valid_name(candidate, dl_num):
                    name_val = candidate
                    name_raw = line
            
            if name_val == "NOT_FOUND" and name_lbl_idx + 1 < len(lines):
                val_line = lines[name_lbl_idx + 1]
                candidate = extract_uppercase_name(val_line)
                if is_valid_name(candidate, dl_num):
                    name_val = candidate
                    name_raw = val_line

        # Fallback Name check
        if name_val == "NOT_FOUND":
            for line in lines:
                if any(h in line.lower() for h in ["driving", "licence", "license", "authority", "union", "india", "transport", "dl no", "lic no", "licence no", "lic.no"]):
                    continue
                candidate = extract_uppercase_name(line)
                if is_valid_name(candidate, dl_num) and len(candidate.split()) >= 2:
                    name_val = candidate
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
