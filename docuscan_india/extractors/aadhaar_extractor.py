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
        
        # Soft / strict regex matches from raw text
        for m in re.finditer(r"\b([0-9OISZB]{4})\s+([0-9OISZB]{4})\s+([0-9OISZB]{4})\b", raw_text, re.IGNORECASE):
            raw_match = m.group(0)
            corrected = ocr_correct_digits(raw_match.replace(" ", ""))
            bbox, constituent_boxes = self.merge_bounding_boxes_with_details(raw_match, word_map)
            cands_no.append({
                "text": corrected,
                "raw_text": raw_match,
                "bbox": bbox,
                "constituent_boxes": constituent_boxes,
                "ocr_confidence": self.get_field_confidence(corrected, word_map),
                "page_source": "visual"
            })
            
        for m in re.finditer(r"\b([0-9OISZB]{12})\b", raw_text, re.IGNORECASE):
            raw_match = m.group(0)
            corrected = ocr_correct_digits(raw_match)
            bbox, constituent_boxes = self.merge_bounding_boxes_with_details(raw_match, word_map)
            cands_no.append({
                "text": corrected,
                "raw_text": raw_match,
                "bbox": bbox,
                "constituent_boxes": constituent_boxes,
                "ocr_confidence": self.get_field_confidence(corrected, word_map),
                "page_source": "visual"
            })
            
        # Reconstruct Aadhaar number from adjacent OCR fragments in word map (tolerant of fragmentation)
        # Find all digit-like tokens on each line, sort horizontally, and check if any subsegment forms a 12-digit number
        line_groups = []
        for w in word_map:
            t = w['text']
            t_corrected = ocr_correct_digits(t)
            if t_corrected.isdigit() and len(t_corrected) > 0:
                cy = w['top'] + w['height'] / 2.0
                added = False
                for group in line_groups:
                    g_cy = sum(box['top'] + box['height'] / 2.0 for box, _ in group) / len(group)
                    if abs(cy - g_cy) < 15:
                        group.append((w, t_corrected))
                        added = True
                        break
                if not added:
                    line_groups.append([(w, t_corrected)])

        for group in line_groups:
            # Sort horizontally
            sorted_group = sorted(group, key=lambda x: x[0]['left'])
            n = len(sorted_group)
            for i in range(n):
                for j in range(i, n):
                    subsegment = sorted_group[i:j+1]
                    
                    # Horizontal spacing must be reasonable
                    valid_gaps = True
                    for idx in range(len(subsegment) - 1):
                        gap = subsegment[idx+1][0]['left'] - (subsegment[idx][0]['left'] + subsegment[idx][0]['width'])
                        if gap < 0 or gap > 120:
                            valid_gaps = False
                            break
                    if not valid_gaps:
                        continue
                        
                    combined = "".join(t_corr for _, t_corr in subsegment)
                    if len(combined) == 12:
                        w_list = [w for w, _ in subsegment]
                        min_x = min(w['left'] for w in w_list)
                        min_y = min(w['top'] for w in w_list)
                        max_r = max(w['left'] + w['width'] for w in w_list)
                        max_b = max(w['top'] + w['height'] for w in w_list)
                        bbox = {
                            'x': min_x,
                            'y': min_y,
                            'w': max_r - min_x,
                            'h': max_b - min_y
                        }
                        raw_comb = " ".join(w['text'] for w in w_list)
                        cands_no.append({
                            "text": combined,
                            "raw_text": raw_comb,
                            "bbox": bbox,
                            "constituent_boxes": w_list,
                            "ocr_confidence": sum(w.get('conf', 0.90) for w in w_list) / len(w_list),
                            "page_source": "visual"
                        })

        # Deduplicate candidates by text
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
            
            # Additional boosts for Aadhaar Name
            height_boost = 0.0
            if bbox:
                height_boost = min(bbox['h'] / 20.0, 1.5)
                
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
        
        # Field-to-BoundingBox Mapping (Single Source of Truth)
        for field_name, field_res in results.items():
            if field_res.value != "NOT_FOUND":
                bbox, constituent = self.map_field_to_bbox(field_name, field_res.value, word_map)
                field_res.bounding_box = bbox
                field_res.constituent_boxes = constituent
                
        return results
