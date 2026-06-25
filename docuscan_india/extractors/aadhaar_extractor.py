import re
from typing import Dict, List, Any
from extractors.base_extractor import BaseExtractor
from utils.document_packet import FieldResult, DocumentType
from utils.string_utils import normalize_date, clean_whitespace, extract_uppercase_name, is_valid_name, ocr_correct_digits
from validators.checksum_validator import ChecksumValidator

class AadhaarExtractor(BaseExtractor):
    def extract(self, raw_text: str, word_map: List[Dict[str, Any]]) -> Dict[str, FieldResult]:
        results: Dict[str, FieldResult] = {}
        lines = [line.strip() for line in raw_text.split('\n') if line.strip()]

        # 1. Aadhaar Number candidates
        cands_no = []
        
        # Soft / strict regex matches
        for m in re.finditer(r"\b([0-9OISZB]{4})\s+([0-9OISZB]{4})\s+([0-9OISZB]{4})\b", raw_text, re.IGNORECASE):
            raw_match = m.group(0)
            corrected = ocr_correct_digits(raw_match.replace(" ", ""))
            cands_no.append({
                "text": corrected,
                "raw_text": raw_match,
                "bbox": self.merge_bounding_boxes(raw_match, word_map),
                "ocr_confidence": self.get_field_confidence(corrected, word_map),
                "page_source": "visual"
            })
            
        for m in re.finditer(r"\b([0-9OISZB]{12})\b", raw_text, re.IGNORECASE):
            raw_match = m.group(0)
            corrected = ocr_correct_digits(raw_match)
            cands_no.append({
                "text": corrected,
                "raw_text": raw_match,
                "bbox": self.merge_bounding_boxes(raw_match, word_map),
                "ocr_confidence": self.get_field_confidence(corrected, word_map),
                "page_source": "visual"
            })
            
        # Standard flat text search
        flat_text = " ".join(raw_text.split())
        for m in re.finditer(r"\b([0-9OISZB]{4})\s+([0-9OISZB]{4})\s+([0-9OISZB]{4})\b", flat_text, re.IGNORECASE):
            raw_match = m.group(0)
            corrected = ocr_correct_digits(raw_match.replace(" ", ""))
            cands_no.append({
                "text": corrected,
                "raw_text": raw_match,
                "bbox": self.merge_bounding_boxes(raw_match, word_map),
                "ocr_confidence": self.get_field_confidence(corrected, word_map),
                "page_source": "visual"
            })

        # Deduplicate by text
        seen = set()
        cands_no_uniq = []
        for c in cands_no:
            if c["text"] not in seen:
                seen.add(c["text"])
                cands_no_uniq.append(c)
                
        results["aadhaar_number"] = self.select_best_candidate("aadhaar_number", cands_no_uniq, DocumentType.AADHAAR, word_map, raw_text)
        
        # 2. DOB candidates
        cands_dob = []
        date_patterns = [
            r"(?:dob|yob|birth|जन्म|तिथि|वर्ष)\s*[:\-]?\s*([0-9/\-\.\s]{4,10})",
            r"\b(\d{2}[/\-\.]\d{2}[/\-\.]\d{4})\b"
        ]
        for pattern in date_patterns:
            for m in re.finditer(pattern, raw_text, re.IGNORECASE):
                val_raw = m.group(1) if len(m.groups()) >= 1 else m.group(0)
                norm = normalize_date(val_raw)
                if norm:
                    cands_dob.append({
                        "text": norm,
                        "raw_text": m.group(0),
                        "bbox": self.merge_bounding_boxes(m.group(0), word_map),
                        "ocr_confidence": self.get_field_confidence(norm, word_map),
                        "page_source": "visual"
                    })
        # Year of birth (YOB) fallback
        for m in re.finditer(r"\b(19\d{2}|20\d{2})\b", raw_text):
            y = m.group(1)
            norm = f"{y}-01-01"
            cands_dob.append({
                "text": norm,
                "raw_text": m.group(0),
                "bbox": self.merge_bounding_boxes(m.group(0), word_map),
                "ocr_confidence": self.get_field_confidence(norm, word_map),
                "page_source": "visual",
                "boost": -2.0
            })
            
        seen = set()
        cands_dob_uniq = []
        for c in cands_dob:
            if c["text"] not in seen:
                seen.add(c["text"])
                cands_dob_uniq.append(c)
                
        results["dob"] = self.select_best_candidate("dob", cands_dob_uniq, DocumentType.AADHAAR, word_map, raw_text)
        
        # 3. Gender candidates
        cands_gen = []
        for m in re.finditer(r"\b(male|female|transgender|पुरुष|महिला)\b", raw_text, re.IGNORECASE):
            raw_match = m.group(0)
            g_lower = raw_match.lower()
            if "female" in g_lower or "महिला" in g_lower:
                gender_val = "FEMALE"
            elif "male" in g_lower or "पुरुष" in g_lower:
                gender_val = "MALE"
            else:
                gender_val = "OTHER"
            cands_gen.append({
                "text": gender_val,
                "raw_text": raw_match,
                "bbox": self.merge_bounding_boxes(raw_match, word_map),
                "ocr_confidence": self.get_field_confidence(gender_val, word_map),
                "page_source": "visual"
            })
            
        seen = set()
        cands_gen_uniq = []
        for c in cands_gen:
            if c["text"] not in seen:
                seen.add(c["text"])
                cands_gen_uniq.append(c)
                
        results["gender"] = self.select_best_candidate("gender", cands_gen_uniq, DocumentType.AADHAAR, word_map, raw_text)
        
        # 4. Name candidates
        cands_name = []
        
        dob_line_idx = -1
        num_line_idx = -1
        
        for idx, line in enumerate(lines):
            if any(k in line.lower() for k in ["dob", "yob", "birth", "जन्म"]):
                dob_line_idx = idx
            if re.search(r"\b\d{4}\s+\d{4}\s+\d{4}\b", line) or re.search(r"\b\d{12}\b", line):
                num_line_idx = idx
                
        for idx, line in enumerate(lines):
            if any(h in line.lower() for h in ["government", "india", "uidai", "unique", "enrollment", "enrolment"]):
                continue
            cand_name = extract_uppercase_name(line)
            if cand_name == "NOT_FOUND" or not is_valid_name(cand_name):
                continue
                
            if len(cand_name.split()) < 2:
                continue
            if any(w in cand_name.upper().split() for w in ["MALE", "FEMALE", "TRANSGENDER", "DOB", "YOB", "BIRTH"]):
                continue
                
            bbox = self.merge_bounding_boxes(line, word_map)
            ocr_conf = self.get_field_confidence(cand_name, word_map)
            
            # Additional boosts for Aadhaar Name:
            # - Bounding box height boost ("largest nearby text")
            height_boost = 0.0
            if bbox:
                height_boost = min(bbox['h'] / 20.0, 1.5)
                
            # - Location relative to DOB/Aadhaar line (above is preferred)
            position_boost = 0.0
            if dob_line_idx != -1 and idx < dob_line_idx:
                position_boost += 1.5
            if num_line_idx != -1 and idx < num_line_idx:
                position_boost += 1.5
                
            cands_name.append({
                "text": cand_name,
                "raw_text": line,
                "bbox": bbox,
                "ocr_confidence": ocr_conf,
                "page_source": "visual",
                "boost": height_boost + position_boost
            })
            
        seen = set()
        cands_name_uniq = []
        for c in cands_name:
            if c["text"] not in seen:
                seen.add(c["text"])
                cands_name_uniq.append(c)
                
        results["name"] = self.select_best_candidate("name", cands_name_uniq, DocumentType.AADHAAR, word_map, raw_text)
        
        return results
