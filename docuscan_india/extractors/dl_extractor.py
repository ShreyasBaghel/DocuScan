import re
from typing import Dict, List, Any
from extractors.base_extractor import BaseExtractor
from utils.document_packet import FieldResult, DocumentType
from utils.string_utils import normalize_date, clean_whitespace, extract_uppercase_name, is_valid_name, ocr_correct_digits

class DLExtractor(BaseExtractor):
    def extract(self, raw_text: str, word_map: List[Dict[str, Any]]) -> Dict[str, FieldResult]:
        results: Dict[str, FieldResult] = {}
        lines = [line.strip() for line in raw_text.split('\n') if line.strip()]

        # 1. DL Number candidates
        cands_no = []
        for m in re.finditer(r"\b([A-Z]{2})\s*[-/]?\s*([0-9OISZB]{2})\s*[-/]?\s*([0-9OISZB]{4})\s*[-/]?\s*([0-9OISZB]{7})\b", raw_text, re.IGNORECASE):
            raw_rto = ocr_correct_digits(m.group(2))
            raw_year = ocr_correct_digits(m.group(3))
            raw_idx = ocr_correct_digits(m.group(4))
            corrected = f"{m.group(1).upper()}{raw_rto}{raw_year}{raw_idx}"
            cands_no.append({
                "text": corrected,
                "raw_text": m.group(0),
                "bbox": self.merge_bounding_boxes(m.group(0), word_map),
                "ocr_confidence": self.get_field_confidence(corrected, word_map),
                "page_source": "visual"
            })
            
        for m in re.finditer(r"\b([A-Z]{2}[0-9OISZB]{13})\b", raw_text, re.IGNORECASE):
            raw_val = m.group(1)
            corrected = raw_val[:2].upper() + ocr_correct_digits(raw_val[2:])
            cands_no.append({
                "text": corrected,
                "raw_text": m.group(0),
                "bbox": self.merge_bounding_boxes(m.group(0), word_map),
                "ocr_confidence": self.get_field_confidence(corrected, word_map),
                "page_source": "visual"
            })
            
        seen = set()
        cands_no_uniq = []
        for c in cands_no:
            if c["text"] not in seen:
                seen.add(c["text"])
                cands_no_uniq.append(c)
                
        results["dl_number"] = self.select_best_candidate("dl_number", cands_no_uniq, DocumentType.DRIVING_LICENCE, word_map, raw_text)
        dl_num = results["dl_number"].value if results["dl_number"].value != "NOT_FOUND" else ""
        
        # 2. DOB candidates
        cands_dob = []
        for m in re.finditer(r"(?:dob|d\.o\.b|birth|जन्म)\s*[:\-]?\s*([0-9/\-\.]{10})", raw_text, re.IGNORECASE):
            val_raw = m.group(1)
            norm = normalize_date(val_raw)
            if norm:
                cands_dob.append({
                    "text": norm,
                    "raw_text": m.group(0),
                    "bbox": self.merge_bounding_boxes(m.group(0), word_map),
                    "ocr_confidence": self.get_field_confidence(norm, word_map),
                    "page_source": "visual"
                })
        for m in re.finditer(r"\b(\d{2}[/\-\.]\d{2}[/\-\.]\d{4})\b", raw_text):
            val_raw = m.group(0)
            norm = normalize_date(val_raw)
            if norm:
                cands_dob.append({
                    "text": norm,
                    "raw_text": val_raw,
                    "bbox": self.merge_bounding_boxes(val_raw, word_map),
                    "ocr_confidence": self.get_field_confidence(norm, word_map),
                    "page_source": "visual"
                })
                
        seen = set()
        cands_dob_uniq = []
        for c in cands_dob:
            if c["text"] not in seen:
                seen.add(c["text"])
                cands_dob_uniq.append(c)
                
        if len(cands_dob_uniq) >= 2:
            cands_dob_uniq.sort(key=lambda x: x["text"])
            
        results["dob"] = self.select_best_candidate("dob", cands_dob_uniq, DocumentType.DRIVING_LICENCE, word_map, raw_text)
        
        # 3. Validity candidates
        cands_val = []
        for m in re.finditer(r"(?:valid|till|expiry|nt|validity)\s*[:\-]?\s*([0-9/\-\.]{10})", raw_text, re.IGNORECASE):
            val_raw = m.group(1)
            norm = normalize_date(val_raw)
            if norm:
                cands_val.append({
                    "text": norm,
                    "raw_text": m.group(0),
                    "bbox": self.merge_bounding_boxes(m.group(0), word_map),
                    "ocr_confidence": self.get_field_confidence(norm, word_map),
                    "page_source": "visual"
                })
        for m in re.finditer(r"\b(\d{2}[/\-\.]\d{2}[/\-\.]\d{4})\b", raw_text):
            val_raw = m.group(0)
            norm = normalize_date(val_raw)
            if norm:
                cands_val.append({
                    "text": norm,
                    "raw_text": val_raw,
                    "bbox": self.merge_bounding_boxes(val_raw, word_map),
                    "ocr_confidence": self.get_field_confidence(norm, word_map),
                    "page_source": "visual"
                })
                
        seen = set()
        cands_val_uniq = []
        for c in cands_val:
            if c["text"] not in seen:
                seen.add(c["text"])
                cands_val_uniq.append(c)
                
        if len(cands_val_uniq) >= 2:
            cands_val_uniq.sort(key=lambda x: x["text"], reverse=True)
            
        results["validity"] = self.select_best_candidate("validity", cands_val_uniq, DocumentType.DRIVING_LICENCE, word_map, raw_text)
        
        # 4. Name candidates
        name_lbl_idx = -1
        for idx, line in enumerate(lines):
            l_lower = line.lower()
            if "name" in l_lower and "father" not in l_lower and "licence" not in l_lower:
                name_lbl_idx = idx
                break
                
        cands_name = []
        for idx, line in enumerate(lines):
            if any(h in line.lower() for h in ["driving", "licence", "license", "authority", "union", "india", "transport", "dl no", "lic no", "licence no", "lic.no"]):
                continue
            cand_name = extract_uppercase_name(line)
            if cand_name == "NOT_FOUND" or not is_valid_name(cand_name, dl_num):
                continue
                
            bbox = self.merge_bounding_boxes(line, word_map)
            ocr_conf = self.get_field_confidence(cand_name, word_map)
            
            boost = 0.0
            if name_lbl_idx != -1:
                if idx == name_lbl_idx:
                    boost += 1.5
                elif idx == name_lbl_idx + 1:
                    boost += 2.0
                    
            cands_name.append({
                "text": cand_name,
                "raw_text": line,
                "bbox": bbox,
                "ocr_confidence": ocr_conf,
                "page_source": "visual",
                "boost": boost
            })
            
        seen = set()
        cands_name_uniq = []
        for c in cands_name:
            if c["text"] not in seen:
                seen.add(c["text"])
                cands_name_uniq.append(c)
                
        results["name"] = self.select_best_candidate("name", cands_name_uniq, DocumentType.DRIVING_LICENCE, word_map, raw_text)
        
        # 5. Vehicle class
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
