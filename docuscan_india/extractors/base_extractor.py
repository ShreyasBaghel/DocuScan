import re
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from utils.document_packet import FieldResult, DocumentType
from utils.string_utils import normalize_date, clean_whitespace

class BaseExtractor(ABC):
    @abstractmethod
    def extract(self, raw_text: str, word_map: List[Dict[str, Any]]) -> Dict[str, FieldResult]:
        """
        Extracts document-specific fields from text and OCR word map.
        Returns a dictionary of field names to FieldResult.
        """
        pass

    def merge_bounding_boxes(self, target_value: str, word_map: List[Dict[str, Any]]) -> Optional[Dict[str, int]]:
        """
        Finds the bounding box of a target value by finding matching words in the word map
        and computing the bounding box that encompasses them.
        """
        if not target_value or not word_map:
            return None

        # Clean and split target value into tokens
        tokens = [t.lower() for t in target_value.split() if re.match(r"^[a-zA-Z0-9/<]+$", t)]
        if not tokens:
            return None

        matched_boxes = []

        # Find words matching tokens
        for token in tokens:
            clean_token = re.sub(r'[^a-z0-9/<]', '', token)
            if not clean_token:
                continue
            for w in word_map:
                w_text = w['text'].lower()
                clean_w_text = re.sub(r'[^a-z0-9/<]', '', w_text)
                if clean_token == clean_w_text:
                    matched_boxes.append(w)

        if not matched_boxes:
            return None

        # Group matched boxes by line (y-coordinate)
        line_groups = []
        for box in matched_boxes:
            box_center_y = box['top'] + box['height'] / 2
            added = False
            for group in line_groups:
                group_center_y = sum(b['top'] + b['height'] / 2 for b in group) / len(group)
                if abs(box_center_y - group_center_y) < 20:  # 20 pixels vertical tolerance
                    group.append(box)
                    added = True
                    break
            if not added:
                line_groups.append([box])

        # Pick the line group that has the most matched boxes
        best_group = max(line_groups, key=len)
        matched_boxes = best_group

        # Compute bounding box encompassing all matched words in the best group
        min_x = min(w['left'] for w in matched_boxes)
        min_y = min(w['top'] for w in matched_boxes)
        max_r = max(w['left'] + w['width'] for w in matched_boxes)
        max_b = max(w['top'] + w['height'] for w in matched_boxes)

        return {
            'x': min_x,
            'y': min_y,
            'w': max_r - min_x,
            'h': max_b - min_y
        }

    def get_field_confidence(self, value: str, word_map: List[Dict[str, Any]], base_conf: float = 0.85) -> float:
        """Calculates a field confidence dynamically based on word-level confidences."""
        if not value or value == "NOT_FOUND" or not word_map:
            return 0.0

        # Clean and split target value into tokens
        tokens = [t.lower() for t in value.split() if re.match(r"^[a-zA-Z0-9/<]+$", t)]
        if not tokens:
            # Fallback to mean of all word confidences
            all_confs = [w['conf'] for w in word_map if 'conf' in w]
            return float(np.mean(all_confs)) if all_confs else base_conf

        matched_confs = []
        for token in tokens:
            clean_token = re.sub(r'[^a-z0-9/<]', '', token)
            if not clean_token:
                continue
            for w in word_map:
                w_text = w['text'].lower()
                clean_w_text = re.sub(r'[^a-z0-9/<]', '', w_text)
                if clean_token == clean_w_text:
                    matched_confs.append(w['conf'])

        if matched_confs:
            return float(np.mean(matched_confs))

        # Dynamic fallback: mean of all word confidences
        all_confs = [w['conf'] for w in word_map if 'conf' in w]
        return float(np.mean(all_confs)) if all_confs else base_conf

    def extract_field_with_regex(self, pattern: str, text: str, word_map: List[Dict[str, Any]], group_name: str = None) -> FieldResult:
        """Helper to run regex on raw text, return FieldResult with bounding box."""
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            val = m.group(group_name) if group_name else m.group(0)
            val_clean = clean_whitespace(val)
            bbox = self.merge_bounding_boxes(val_clean, word_map)
            # Estimate confidence dynamically from matching words
            conf = self.get_field_confidence(val_clean, word_map)
            return FieldResult(value=val_clean, raw_text=m.group(0), confidence=conf, bounding_box=bbox)
        
        return FieldResult(value="NOT_FOUND", raw_text="", confidence=0.0, bounding_box=None)

    def is_valid_calendar_date(self, date_str: str) -> bool:
        if not date_str or date_str == "NOT_FOUND":
            return False
        parts = date_str.split("-")
        if len(parts) != 3:
            return False
        try:
            y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
            if y < 1900 or y > 2100:
                return False
            if m < 1 or m > 12:
                return False
            if d < 1 or d > 31:
                return False
            from datetime import datetime
            datetime(y, m, d)
            return True
        except Exception:
            return False

    def score_candidate(
        self,
        field_name: str,
        candidate: Dict[str, Any],
        doc_type: DocumentType,
        word_map: List[Dict[str, Any]],
        raw_text: str
    ) -> float:
        from utils.string_utils import is_valid_name
        
        text = candidate.get("text", "")
        raw_val = candidate.get("raw_text", "")
        bbox = candidate.get("bbox", None)
        ocr_conf = candidate.get("ocr_confidence", 0.85)
        page_source = candidate.get("page_source", "standard")
        
        # 1. OCR confidence
        score = ocr_conf
        
        # 2. Location score
        loc_score = 0.0
        img_w = 1000
        img_h = 1000
        if word_map:
            max_r = max((w.get('left', 0) + w.get('width', 0) for w in word_map), default=1000)
            max_b = max((w.get('top', 0) + w.get('height', 0) for w in word_map), default=1000)
            if max_r > 0: img_w = max_r
            if max_b > 0: img_h = max_b
            
        if bbox:
            cx = bbox['x'] + bbox['w'] / 2.0
            cy = bbox['y'] + bbox['h'] / 2.0
            x_norm = cx / img_w
            y_norm = cy / img_h
            
            if doc_type == DocumentType.AADHAAR:
                if field_name == "aadhaar_number":
                    if y_norm > 0.5:
                        loc_score = 1.0
                    else:
                        loc_score = 0.2
                elif field_name == "name":
                    if 0.2 < y_norm < 0.6:
                        loc_score = 1.0
                    else:
                        loc_score = 0.1
                elif field_name == "dob":
                    if 0.3 < y_norm < 0.7:
                        loc_score = 1.0
                    else:
                        loc_score = 0.3
            elif doc_type == DocumentType.PAN:
                if field_name == "pan_number":
                    if y_norm > 0.5:
                        loc_score = 1.0
                    else:
                        loc_score = 0.2
                elif field_name == "name":
                    if 0.2 < y_norm < 0.6:
                        loc_score = 1.0
                    else:
                        loc_score = 0.1
                elif field_name == "father_name":
                    if 0.3 < y_norm < 0.7:
                        loc_score = 1.0
                    else:
                        loc_score = 0.1
            elif doc_type == DocumentType.PASSPORT:
                if field_name == "passport_number":
                    if page_source == "page_2" or y_norm > 0.7:
                        loc_score = 1.0
                    elif y_norm < 0.3 and x_norm > 0.5:
                        loc_score = 1.0
                    else:
                        loc_score = 0.2
                elif field_name == "name":
                    if page_source == "page_2" or y_norm > 0.7:
                        loc_score = 1.0
                    elif 0.15 < y_norm < 0.5:
                        loc_score = 1.0
                    else:
                        loc_score = 0.1
            elif doc_type == DocumentType.DRIVING_LICENCE:
                if field_name == "dl_number":
                    if y_norm < 0.5:
                        loc_score = 1.0
                    else:
                        loc_score = 0.4
                elif field_name == "name":
                    if 0.2 < y_norm < 0.6:
                        loc_score = 1.0
                    else:
                        loc_score = 0.2
        else:
            if page_source in ["page_2", "page_1"] or "mrz" in page_source:
                loc_score = 0.8
            else:
                loc_score = 0.3
                
        score += loc_score
        
        # 3. Label proximity score
        label_score = 0.0
        lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
        
        candidate_line_idx = -1
        for idx, line in enumerate(lines):
            if raw_val in line or text in line:
                candidate_line_idx = idx
                break
                
        label_keywords = {
            "name": ["name", "नाम", "given name", "surname", "gvn", "nam", "nom"],
            "father_name": ["father", "पिता", "father's name", "fthr", "pit"],
            "dob": ["dob", "birth", "जन्म", "yob", "date of birth", "nasc", "d.o.b", "dt of birth"],
            "expiry": ["expiry", "till", "valid", "validity", "exp", "val"],
            "validity": ["expiry", "till", "valid", "validity", "exp", "val"],
            "aadhaar_number": ["aadhaar", "unique", "enrollment", "enrolment", "uidai", "aadhar", "adh", "number", "no"],
            "pan_number": ["permanent", "account", "tax", "income", "card", "pan", "p.a.n"],
            "passport_number": ["passport", "pass no", "passport no", "document no", "pasp", "no", "number"],
            "dl_number": ["licence", "license", "dl no", "lic no", "licence no", "lic.no", "driving"]
        }
        
        keywords = label_keywords.get(field_name, [])
        
        if candidate_line_idx != -1:
            same_line = lines[candidate_line_idx].lower()
            for kw in keywords:
                if kw in same_line:
                    kw_pos = same_line.find(kw)
                    cand_pos = same_line.find(raw_val) if raw_val in same_line else same_line.find(text)
                    if cand_pos > kw_pos:
                        label_score = max(label_score, 1.5)
                    else:
                        label_score = max(label_score, 0.8)
            
            if candidate_line_idx > 0:
                above_line = lines[candidate_line_idx - 1].lower()
                for kw in keywords:
                    if kw in above_line:
                        label_score = max(label_score, 1.5)
                        
            if candidate_line_idx > 1:
                above_two = lines[candidate_line_idx - 2].lower()
                for kw in keywords:
                    if kw in above_two:
                        label_score = max(label_score, 0.7)
                        
            if candidate_line_idx + 1 < len(lines):
                below_line = lines[candidate_line_idx + 1].lower()
                for kw in keywords:
                    if kw in below_line:
                        label_score = max(label_score, 0.5)
                        
        score += label_score
        
        # 4. Regex validity score
        regex_score = 0.0
        if field_name == "aadhaar_number":
            cleaned_num = re.sub(r"\s+", "", text)
            if re.match(r"^\d{12}$", cleaned_num):
                regex_score = 1.5
            elif re.match(r"^[0-9OISZB]{12}$", cleaned_num):
                regex_score = 0.7
            else:
                regex_score = -2.0
        elif field_name == "pan_number":
            if re.match(r"^[A-Z]{5}[0-9]{4}[A-Z]$", text):
                regex_score = 1.5
            elif re.match(r"^[A-Z]{5}[0-9OISZB]{4}[A-Z]$", text):
                regex_score = 0.7
            else:
                regex_score = -2.0
        elif field_name == "passport_number":
            if re.match(r"^[A-Z][0-9]{7}$", text):
                regex_score = 1.5
            elif re.match(r"^[A-Z][0-9OISZB]{7}$", text):
                regex_score = 0.7
            else:
                regex_score = -2.0
        elif field_name == "dl_number":
            if re.match(r"^[A-Z]{2}[0-9]{2}[0-9]{4}[0-9]{7}$", text):
                regex_score = 1.5
            elif re.match(r"^[A-Z]{2}[0-9OISZB]{13}$", text):
                regex_score = 0.7
            else:
                regex_score = -2.0
        elif field_name in ["dob", "expiry", "validity"]:
            if not self.is_valid_calendar_date(text):
                regex_score = -5.0
            elif re.match(r"^\d{4}-\d{2}-\d{2}$", text):
                regex_score = 1.5
            else:
                regex_score = -2.0
        elif field_name in ["name", "father_name"]:
            if any(c.isdigit() for c in text):
                regex_score = -3.0
            elif not is_valid_name(text):
                regex_score = -3.0
            else:
                regex_score = 1.0
                
        score += regex_score
        
        # 5. Document structure / Evidence score
        structure_score = 0.0
        if field_name == "aadhaar_number":
            from validators.checksum_validator import ChecksumValidator
            cleaned_num = re.sub(r"\s+", "", text)
            if ChecksumValidator.validate_verhoeff(cleaned_num):
                structure_score = 2.0
        elif field_name == "passport_number":
            if page_source in ["mrz_line2", "mrz"]:
                structure_score = 2.0
        elif field_name in ["dob", "expiry"] and doc_type == DocumentType.PASSPORT:
            if page_source in ["mrz_line2", "mrz"]:
                if candidate.get("mrz_valid", False):
                    structure_score = 2.0
                else:
                    structure_score = -1.0
                
        score += structure_score
        
        # 6. Field length score
        length_score = 0.0
        text_len = len(text)
        if field_name == "aadhaar_number":
            cleaned_num = re.sub(r"\s+", "", text)
            if len(cleaned_num) == 12:
                length_score = 0.5
            else:
                length_score = -3.0
        elif field_name == "pan_number":
            if text_len == 10:
                length_score = 0.5
            else:
                length_score = -3.0
        elif field_name == "passport_number":
            if text_len == 8:
                length_score = 0.5
            else:
                length_score = -3.0
        elif field_name in ["dob", "expiry", "validity"]:
            if text_len == 10:
                length_score = 0.5
            else:
                length_score = -2.0
        elif field_name in ["name", "father_name"]:
            if 3 <= text_len <= 50:
                length_score = 0.5
            elif text_len < 3:
                length_score = -3.0
            else:
                length_score = -1.5
                
        score += length_score
        score += candidate.get("boost", 0.0)
        
        # Future DOB penalty
        if field_name == "dob":
            if self.is_valid_calendar_date(text):
                from datetime import datetime, date
                try:
                    dt = datetime.strptime(text, "%Y-%m-%d").date()
                    if dt > date.today():
                        score -= 5.0
                except Exception:
                    pass
        
        return score

    def select_best_candidate(
        self,
        field_name: str,
        candidates: List[Dict[str, Any]],
        doc_type: DocumentType,
        word_map: List[Dict[str, Any]],
        raw_text: str
    ) -> FieldResult:
        if not candidates:
            return FieldResult(value="NOT_FOUND", raw_text="", confidence=0.0, bounding_box=None)
            
        # Score all candidates
        scored_candidates = []
        for c in candidates:
            score = self.score_candidate(field_name, c, doc_type, word_map, raw_text)
            scored_candidates.append((score, c))
            
        # Sort candidates by score descending
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Logging in debug mode
        from utils.logger import get_logger
        from utils.string_utils import is_valid_name
        logger = get_logger("base_extractor")
        
        logger.debug(f"Field: {field_name}\n")
        for idx, (score, c) in enumerate(scored_candidates):
            # Check validation result dynamically
            val_res = "PASS"
            if field_name == "aadhaar_number":
                from validators.checksum_validator import ChecksumValidator
                val_res = "PASS" if ChecksumValidator.validate_verhoeff(c["text"]) else "FAIL"
            elif field_name == "passport_number":
                if c.get("page_source") == "mrz":
                    val_res = "PASS" if c.get("mrz_valid", False) else "FAIL"
                else:
                    val_res = "PASS" if re.match(r"^[A-Z][0-9]{7}$", c["text"]) else "FAIL"
            elif field_name == "pan_number":
                val_res = "PASS" if re.match(r"^[A-Z]{5}[0-9]{4}[A-Z]$", c["text"]) else "FAIL"
            elif field_name == "dl_number":
                val_res = "PASS" if re.match(r"^[A-Z]{2}[0-9]{2}[0-9]{4}[0-9]{7}$", c["text"]) else "FAIL"
            elif field_name in ["dob", "expiry", "validity"]:
                val_res = "PASS" if re.match(r"^\d{4}-\d{2}-\d{2}$", c["text"]) else "FAIL"
            elif field_name in ["name", "father_name"]:
                val_res = "PASS" if is_valid_name(c["text"]) and not any(ch.isdigit() for ch in c["text"]) else "FAIL"
                
            logger.debug(f"Candidate {idx + 1}:")
            logger.debug(f"Text={c['text']}")
            logger.debug(f"Score={score:.2f}")
            logger.debug(f"Validation Result={val_res}\n")
            
        best_score, best_candidate = scored_candidates[0]
        
        if best_candidate['text'] == "NOT_FOUND" or best_score < 0.0 or not best_candidate['text'].strip():
            logger.debug("Selected=NOT_FOUND")
            return FieldResult(value="NOT_FOUND", raw_text="", confidence=0.0, bounding_box=None)
            
        logger.debug(f"Selected={best_candidate['text']}")
        
        bbox = best_candidate.get("bbox")
        if not bbox and best_candidate.get("text"):
            bbox = self.merge_bounding_boxes(best_candidate["raw_text"], word_map)
            
        conf = best_candidate.get("ocr_confidence", 0.0)
        if conf == 0.0:
            conf = self.get_field_confidence(best_candidate["text"], word_map)
            
        return FieldResult(
            value=best_candidate["text"],
            raw_text=best_candidate["raw_text"],
            confidence=conf,
            bounding_box=bbox
        )

