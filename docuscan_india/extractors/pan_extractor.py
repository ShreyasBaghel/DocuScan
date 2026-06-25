import re
from typing import Dict, List, Any
from extractors.base_extractor import BaseExtractor
from utils.document_packet import FieldResult, DocumentType
from utils.string_utils import normalize_date, clean_whitespace, extract_uppercase_name, is_valid_name, ocr_correct_digits

class PANExtractor(BaseExtractor):
    def extract(self, raw_text: str, word_map: List[Dict[str, Any]]) -> Dict[str, FieldResult]:
        results: Dict[str, FieldResult] = {}
        lines = [line.strip() for line in raw_text.split('\n') if line.strip()]

        # 1. PAN Number candidates
        cands_no = []
        for m in re.finditer(r"(?:[^A-Z0-9]|^)([A-Z]{5}[0-9OISZB]{4}[A-Z])(?:[^A-Z0-9]|$)", raw_text):
            val_raw = m.group(1)
            corrected = val_raw[:5].upper() + ocr_correct_digits(val_raw[5:9]) + val_raw[9].upper()
            cands_no.append({
                "text": corrected,
                "raw_text": val_raw,
                "bbox": self.merge_bounding_boxes(val_raw, word_map),
                "ocr_confidence": self.get_field_confidence(corrected, word_map),
                "page_source": "visual"
            })
            
        seen = set()
        cands_no_uniq = []
        for c in cands_no:
            if c["text"] not in seen:
                seen.add(c["text"])
                cands_no_uniq.append(c)
                
        results["pan_number"] = self.select_best_candidate("pan_number", cands_no_uniq, DocumentType.PAN, word_map, raw_text)
        pan_num = results["pan_number"].value if results["pan_number"].value != "NOT_FOUND" else ""
        
        # 2. DOB candidates
        cands_dob = []
        date_patterns = [
            r"\b(\d{2}[/\-\.]\d{2}[/\-\.]\d{4})\b",
            r"\b[0-9OISZB]{2}[/\-\.][0-9OISZB]{2}[/\-\.][0-9OISZB]{4}\b"
        ]
        
        birth_lbl_idx = -1
        for idx, line in enumerate(lines):
            if any(k in line.lower() for k in ["birth", "जन्म", "date"]):
                birth_lbl_idx = idx
                break
        if birth_lbl_idx != -1 and birth_lbl_idx + 1 < len(lines):
            cands_dob.append({
                "text": normalize_date(lines[birth_lbl_idx + 1]) or "NOT_FOUND",
                "raw_text": lines[birth_lbl_idx + 1],
                "bbox": self.merge_bounding_boxes(lines[birth_lbl_idx + 1], word_map),
                "ocr_confidence": 0.85,
                "page_source": "visual"
            })
            
        for pattern in date_patterns:
            for m in re.finditer(pattern, raw_text):
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
            if c["text"] not in seen and c["text"] != "NOT_FOUND":
                seen.add(c["text"])
                cands_dob_uniq.append(c)
                
        results["dob"] = self.select_best_candidate("dob", cands_dob_uniq, DocumentType.PAN, word_map, raw_text)
        
        # 3. Name and Father's Name candidates
        name_lbl_idx = -1
        father_lbl_idx = -1
        for idx, line in enumerate(lines):
            l_lower = line.lower()
            if ("name" in l_lower or "नाम" in l_lower) and "father" not in l_lower and "पिता" not in l_lower:
                if name_lbl_idx == -1:
                    name_lbl_idx = idx
            if "father" in l_lower or "पिता" in l_lower:
                father_lbl_idx = idx
                
        cands_name = []
        cands_father = []
        
        for idx, line in enumerate(lines):
            if any(h in line.lower() for h in ["income", "tax", "department", "permanent", "account", "card", "govt", "india", "signature"]):
                continue
            if sum(c.isdigit() for c in line) >= 3:
                continue
                
            cand_name = extract_uppercase_name(line)
            if cand_name == "NOT_FOUND" or not is_valid_name(cand_name, pan_num):
                continue
                
            bbox = self.merge_bounding_boxes(line, word_map)
            ocr_conf = self.get_field_confidence(cand_name, word_map)
            
            name_boost = 0.0
            if name_lbl_idx != -1:
                if idx == name_lbl_idx + 1:
                    name_boost += 2.0
                elif idx == name_lbl_idx + 2:
                    name_boost += 1.0
                    
            father_boost = 0.0
            if father_lbl_idx != -1:
                if idx == father_lbl_idx + 1:
                    father_boost += 2.0
                elif idx == father_lbl_idx + 2:
                    father_boost += 1.0
                    
            cands_name.append({
                "text": cand_name,
                "raw_text": line,
                "bbox": bbox,
                "ocr_confidence": ocr_conf,
                "page_source": "visual",
                "boost": name_boost,
                "pan_number": pan_num
            })
            
            cands_father.append({
                "text": cand_name,
                "raw_text": line,
                "bbox": bbox,
                "ocr_confidence": ocr_conf,
                "page_source": "visual",
                "boost": father_boost,
                "pan_number": pan_num
            })
            
        seen_name = set()
        cands_name_uniq = []
        for c in cands_name:
            if c["text"] not in seen_name:
                seen_name.add(c["text"])
                cands_name_uniq.append(c)
                
        seen_father = set()
        cands_father_uniq = []
        for c in cands_father:
            if c["text"] not in seen_father:
                seen_father.add(c["text"])
                cands_father_uniq.append(c)

        # Select Name first
        # To avoid name == father_name, we pass father's name as context if available.
        # But we do not have father's name selected yet. So we select name first, then validate father's name against the selected name.
        results["name"] = self.select_best_candidate("name", cands_name_uniq, DocumentType.PAN, word_map, raw_text)
        selected_name = results["name"].value
        
        # Now pass selected_name and pan_number to father_name candidates
        for c in cands_father_uniq:
            c["selected_name"] = selected_name

        results["father_name"] = self.select_best_candidate("father_name", cands_father_uniq, DocumentType.PAN, word_map, raw_text)
        selected_father = results["father_name"].value

        # Post-verification: if Name was selected but we now find it equals selected_father, we re-verify name
        if selected_name == selected_father and selected_name != "NOT_FOUND":
            # Re-select Name with selected_father as context
            cands_name_filtered = []
            for c in cands_name_uniq:
                c_copy = c.copy()
                c_copy["selected_father_name"] = selected_father
                cands_name_filtered.append(c_copy)
            results["name"] = self.select_best_candidate("name", cands_name_filtered, DocumentType.PAN, word_map, raw_text)
        
        return results
