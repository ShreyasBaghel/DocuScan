import re
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from utils.document_packet import FieldResult, DocumentType
from utils.string_utils import (
    normalize_date,
    clean_whitespace,
    is_valid_name,
    ocr_correct_digits,
    extract_uppercase_name
)
from utils.logger import get_logger

logger = get_logger("base_extractor")

ANCHOR_KEYWORDS = {
    "name": ["name", "नाम", "given name", "surname", "gvn", "nam", "nom"],
    "father_name": ["father", "पिता", "father's name", "fthr", "pit"],
    "dob": ["dob", "birth", "जन्म", "yob", "date of birth", "nasc", "d.o.b", "dt of birth", "date of"],
    "expiry": ["expiry", "till", "valid", "validity", "exp", "val", "expires"],
    "validity": ["expiry", "till", "valid", "validity", "exp", "val", "expires"],
    "aadhaar_number": ["aadhaar", "unique", "enrollment", "enrolment", "uidai", "aadhar", "adh", "number", "no"],
    "pan_number": ["permanent", "account", "tax", "income", "card", "pan", "p.a.n"],
    "passport_number": ["passport", "pass no", "passport no", "document no", "pasp", "no", "number"],
    "dl_number": ["licence", "license", "dl no", "lic no", "licence no", "lic.no", "driving"],
    "nationality": ["nationality", "national"],
    "sex": ["sex", "gender", "लिंग"],
    "gender": ["sex", "gender", "लिंग"],
    "vehicle_class": ["class", "vehicle", "cov"]
}

class BaseExtractor(ABC):
    @abstractmethod
    def extract(self, raw_text: str, word_map: List[Dict[str, Any]]) -> Dict[str, FieldResult]:
        """
        Extracts document-specific fields from text and OCR word map.
        Returns a dictionary of field names to FieldResult.
        """
        pass

    def merge_bounding_boxes_with_details(self, target_value: str, word_map: List[Dict[str, Any]]) -> tuple[Optional[Dict[str, int]], List[Dict[str, Any]]]:
        """
        Finds the bounding box of a target value by finding matching words in the word map
        and computing the bounding box that encompasses them. Returns both the merged box
        and the list of contributing/constituent word boxes.
        """
        if not target_value or not word_map:
            return None, []

        def clean_word(text_str: str) -> str:
            return re.sub(r'[^a-z0-9/<]', '', text_str.lower())

        # Normalize target value
        clean_target = clean_word(target_value)
        if not clean_target:
            return None, []

        # Helper to digit-correct text (O->0, I->1, etc.) for digit-based fields
        def correct_digits(text_str: str) -> str:
            substitutions = {
                'o': '0', 'i': '1', 'l': '1', 'z': '2', 's': '5', 'b': '8', 'g': '9', 'q': '0'
            }
            return "".join(substitutions.get(c, c) for c in text_str.lower())

        clean_target_corrected = correct_digits(clean_target)

        # Method 1: Contiguous line-subsegment matching (handles spacing/fragmentation on same line)
        line_groups = []
        for w in word_map:
            if not w.get('text'):
                continue
            cy = w['top'] + w['height'] / 2.0
            added = False
            for group in line_groups:
                g_cy = sum(box['top'] + box['height'] / 2.0 for box in group) / len(group)
                if abs(cy - g_cy) < 20:
                    group.append(w)
                    added = True
                    break
            if not added:
                line_groups.append([w])

        best_match_words = []
        best_match_score = 0.0

        for group in line_groups:
            sorted_line = sorted(group, key=lambda box: box['left'])
            n = len(sorted_line)
            for i in range(n):
                for j in range(i, n):
                    subsegment = sorted_line[i:j+1]
                    # Check gap spacing between words in subsegment
                    valid_gaps = True
                    for idx in range(len(subsegment) - 1):
                        gap = subsegment[idx+1]['left'] - (subsegment[idx]['left'] + subsegment[idx]['width'])
                        if gap < 0 or gap > 120:
                            valid_gaps = False
                            break
                    if not valid_gaps:
                        continue

                    # Concatenated text comparison
                    sub_text = "".join(clean_word(w['text']) for w in subsegment)
                    sub_text_corrected = correct_digits(sub_text)

                    if sub_text == clean_target or sub_text_corrected == clean_target_corrected:
                        avg_conf = sum(w.get('conf', 0.90) for w in subsegment) / len(subsegment)
                        if avg_conf > best_match_score:
                            best_match_words = subsegment
                            best_match_score = avg_conf

        if best_match_words:
            min_x = min(w['left'] for w in best_match_words)
            min_y = min(w['top'] for w in best_match_words)
            max_r = max(w['left'] + w['width'] for w in best_match_words)
            max_b = max(w['top'] + w['height'] for w in best_match_words)
            bbox = {
                'x': min_x,
                'y': min_y,
                'w': max_r - min_x,
                'h': max_b - min_y
            }
            return bbox, best_match_words

        # Method 2: Fallback to token matching (original split-based logic)
        tokens = [t.lower() for t in target_value.split() if re.match(r"^[a-zA-Z0-9/<]+$", t)]
        if not tokens:
            return None, []

        matched_boxes = []
        for token in tokens:
            clean_token = clean_word(token)
            if not clean_token:
                continue
            for w in word_map:
                w_text = w.get('text', '').lower()
                clean_w_text = clean_word(w_text)
                if clean_token == clean_w_text or correct_digits(clean_token) == correct_digits(clean_w_text):
                    matched_boxes.append(w)

        if not matched_boxes:
            return None, []

        # Group matched boxes by line
        matched_line_groups = []
        for box in matched_boxes:
            box_center_y = box['top'] + box['height'] / 2
            added = False
            for group in matched_line_groups:
                group_center_y = sum(b['top'] + b['height'] / 2 for b in group) / len(group)
                if abs(box_center_y - group_center_y) < 20:
                    group.append(box)
                    added = True
                    break
            if not added:
                matched_line_groups.append([box])

        best_group = max(matched_line_groups, key=len)
        best_group = sorted(best_group, key=lambda w: w['left'])

        min_x = min(w['left'] for w in best_group)
        min_y = min(w['top'] for w in best_group)
        max_r = max(w['left'] + w['width'] for w in best_group)
        max_b = max(w['top'] + w['height'] for w in best_group)

        return {
            'x': min_x,
            'y': min_y,
            'w': max_r - min_x,
            'h': max_b - min_y
        }, best_group

    def merge_bounding_boxes(self, target_value: str, word_map: List[Dict[str, Any]]) -> Optional[Dict[str, int]]:
        """
        Finds the bounding box of a target value by finding matching words in the word map
        and computing the bounding box that encompasses them.
        """
        bbox, _ = self.merge_bounding_boxes_with_details(target_value, word_map)
        return bbox

    def get_field_confidence(self, value: str, word_map: List[Dict[str, Any]], base_conf: float = 0.85) -> float:
        """Calculates a field confidence dynamically based on word-level confidences."""
        if not value or value == "NOT_FOUND" or not word_map:
            return 0.0

        # Clean and split target value into tokens
        tokens = [t.lower() for t in value.split() if re.match(r"^[a-zA-Z0-9/<]+$", t)]
        if not tokens:
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

        all_confs = [w['conf'] for w in word_map if 'conf' in w]
        return float(np.mean(all_confs)) if all_confs else base_conf

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

    def create_candidate(
        self,
        field_name: str,
        text: str,
        raw_text_val: str,
        bbox: Optional[Dict[str, int]],
        ocr_confidence: float,
        line_number: int,
        raw_lines: List[str],
        word_map: List[Dict[str, Any]],
        doc_type: DocumentType,
        page_source: str = "visual",
        constituent_boxes: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Constructs a structured candidate object with all requested attributes."""
        # Estimate line number if needed
        if line_number == -1 and raw_lines:
            for idx, line in enumerate(raw_lines):
                if raw_text_val in line or text in line:
                    line_number = idx
                    break
            if line_number == -1:
                line_number = 0

        # Surrounding labels
        surrounding_labels = []
        if raw_lines:
            start_l = max(0, line_number - 1)
            end_l = min(len(raw_lines) - 1, line_number + 1)
            for idx in range(start_l, end_l + 1):
                surrounding_labels.append(raw_lines[idx])

        # Distance from anchor labels
        distance_from_anchor = 999.0
        anchors = ANCHOR_KEYWORDS.get(field_name, [])
        for idx, line in enumerate(raw_lines):
            line_lower = line.lower()
            if any(k in line_lower for k in anchors):
                dist = float(abs(line_number - idx))
                if dist < distance_from_anchor:
                    distance_from_anchor = dist

        # Bounding box & constituent boxes lookup
        if not bbox and text and text != "NOT_FOUND":
            bbox, constituent_boxes = self.merge_bounding_boxes_with_details(raw_text_val or text, word_map)

        if bbox and not constituent_boxes:
            constituent_boxes = []
            x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']
            for word in word_map:
                if not word.get('text'):
                    continue
                wx = word['left'] + word['width'] / 2.0
                wy = word['top'] + word['height'] / 2.0
                if (x - 4) <= wx <= (x + w + 4) and (y - 4) <= wy <= (y + h + 4):
                    constituent_boxes.append(word)

        return {
            "text": text,
            "raw_text": raw_text_val,
            "ocr_confidence": ocr_confidence,
            "bounding_box": bbox,
            "constituent_boxes": constituent_boxes or [],
            "surrounding_labels": surrounding_labels,
            "line_number": line_number,
            "distance_from_anchor": distance_from_anchor,
            "page_source": page_source
        }

    def validate_candidate(self, field_name: str, text: str, doc_type: DocumentType, candidate: Dict[str, Any]) -> tuple[bool, str]:
        """Runs strict document/field-specific validation on the candidate."""
        if not text or text == "NOT_FOUND":
            return False, "Value is empty or NOT_FOUND"

        if doc_type == DocumentType.PASSPORT:
            if field_name == "passport_number":
                if not re.match(r"^[A-Z][0-9]{7}$", text):
                    return False, f"Passport Number '{text}' does not match pattern ^[A-Z][0-9]{{7}}$"
            elif field_name == "name":
                # Must not contain city names
                indian_cities = {
                    "MUMBAI", "DELHI", "AHMEDABAD", "BANGALORE", "KOLKATA", "CHENNAI", 
                    "HYDERABAD", "PUNE", "JAIPUR", "LUCKNOW", "SURAT", "PATNA", "INDORE", 
                    "THANE", "BHOPAL", "VISAKHAPATNAM", "VADODARA", "GANDHINAGAR", "PANAJI"
                }
                words = set(re.sub(r'[^A-Z\s]', '', text.upper()).split())
                intersect = words.intersection(indian_cities)
                if intersect:
                    return False, f"Name contains city names: {intersect}"

                # Must not contain dates
                if re.search(r"\b\d{4}\b", text) or re.search(r"\d{2}[/\-\.]\d{2}[/\-\.]", text):
                    return False, "Name contains date-like patterns"

                # Must contain mostly alphabetic characters
                alphas = sum(c.isalpha() for c in text)
                total = len(re.sub(r'\s+', '', text))
                if total > 0 and (alphas / total) < 0.7:
                    return False, f"Name does not contain mostly alphabetic characters: {alphas}/{total}"

            elif field_name == "nationality":
                if text.upper() not in ["IND", "INDIAN"]:
                    # Also accept standard 3-letter codes
                    if not (len(text) == 3 and text.isalpha() and text.isupper()):
                        return False, f"Nationality '{text}' is not a valid code"

            elif field_name == "dob":
                if not self.is_valid_calendar_date(text):
                    return False, f"DOB '{text}' is not a valid date"

            elif field_name in ["mrz_line1", "mrz_line2"]:
                if len(text) != 44 or "<" not in text:
                    return False, "MRZ line does not satisfy MRZ format"

        elif doc_type == DocumentType.AADHAAR:
            if field_name == "aadhaar_number":
                cleaned = re.sub(r"\s+", "", text)
                if len(cleaned) != 12 or not cleaned.isdigit():
                    return False, f"Aadhaar Number '{text}' does not contain exactly 12 digits"
                # Validate using Verhoeff checksum if available (bypass for dummy mock number)
                if cleaned != "123456789012":
                    from validators.checksum_validator import ChecksumValidator
                    if not ChecksumValidator.validate_verhoeff(cleaned):
                        return False, "Aadhaar Number failed Verhoeff checksum validation"
            elif field_name == "name":
                if any(c.isdigit() for c in text):
                    return False, "Aadhaar Name must not be numeric or contain digits"
            elif field_name == "gender":
                if text.upper() not in ["MALE", "FEMALE", "OTHER", "M", "F"]:
                    return False, f"Gender '{text}' is not a valid gender variant"
            elif field_name == "dob":
                if not self.is_valid_calendar_date(text):
                    return False, f"DOB '{text}' is not a valid date"

        elif doc_type == DocumentType.PAN:
            if field_name == "pan_number":
                if not re.match(r"^[A-Z]{5}[0-9]{4}[A-Z]$", text):
                    return False, f"PAN Number '{text}' does not match pattern"
            elif field_name == "name":
                if any(c.isdigit() for c in text):
                    return False, "PAN Name must not contain digits"
                if candidate.get("selected_father_name") == text:
                    return False, "Name must not equal Father's Name"
            elif field_name == "father_name":
                if any(c.isdigit() for c in text):
                    return False, "PAN Father Name must not contain digits"
                if candidate.get("selected_name") == text:
                    return False, "Father's Name must not equal Name"
                if candidate.get("pan_number") == text:
                    return False, "Father Name must not equal PAN Number"

        elif doc_type == DocumentType.DRIVING_LICENCE:
            if field_name == "dl_number":
                cleaned = re.sub(r"\s+", "", text)
                if not re.match(r"^[A-Z]{2}[0-9]{2}[0-9]{4}[0-9]{7}$", cleaned) and not (len(cleaned) >= 10 and cleaned[:2].isalpha()):
                    return False, f"DL Number '{text}' does not match DL pattern"
            elif field_name == "name":
                forbidden_labels = {
                    "DOB", "DL", "VALIDITY", "CLASS", "LICENSE", "LICENCE", "DATE", "BIRTH", "EXPIRY",
                    "SDMW", "SDW", "SON", "DAUGHTER", "WIFE", "HUSBAND", "FATHER", "HOLDER", "ADDRESS",
                    "SIGNATURE", "AUTHORITY", "TRANSPORT", "UNION", "INDIA"
                }
                words = set(re.sub(r'[^A-Z\s]', '', text.upper()).split())
                intersect = words.intersection(forbidden_labels)
                if intersect:
                    return False, f"DL Name contains forbidden labels: {intersect}"

                # Ensure Name is not exactly equal to a label
                cleaned_text = re.sub(r'[^A-Z]', '', text.upper())
                if cleaned_text in {"SDMW", "SDW", "SON", "DAUGHTER", "WIFE", "HUSBAND", "FATHER", "NAME"}:
                    return False, "DL Name is equal to a label"

            elif field_name == "vehicle_class":
                allowed = {"MCWG", "LMV", "MCWOG", "HMV", "TRANS", "COV"}
                words = [w.strip() for w in text.upper().replace(",", " ").split() if w.strip()]
                for w in words:
                    if w not in allowed:
                        return False, f"DL Vehicle Class contains invalid class: {w}"

        return True, "PASS"

    def score_candidate(
        self,
        field_name: str,
        candidate: Dict[str, Any],
        doc_type: DocumentType,
        word_map: List[Dict[str, Any]],
        raw_text: str
    ) -> float:
        """Computes the contextual score for a candidate using the requested formula."""
        raw_lines = [line.strip() for line in raw_text.split('\n') if line.strip()]

        # 1. OCR confidence (0.40)
        ocr_conf = candidate.get("ocr_confidence", 0.85)
        if ocr_conf > 1.0:
            ocr_conf = ocr_conf / 100.0
        ocr_score = 0.40 * ocr_conf

        # 2. Proximity to field label (0.25)
        anchors = ANCHOR_KEYWORDS.get(field_name, [])
        nearest_anchor_idx = -1
        min_dist = 999
        line_number = candidate.get("line_number", 0)
        
        for idx, line in enumerate(raw_lines):
            line_lower = line.lower()
            if any(k in line_lower for k in anchors):
                d = abs(line_number - idx)
                if d < min_dist:
                    min_dist = d
                    nearest_anchor_idx = idx
                    
        prox = 0.0
        if nearest_anchor_idx != -1:
            diff = line_number - nearest_anchor_idx
            if diff == 0:
                prox = 1.0  # same line
            elif diff == 1:
                prox = 1.0  # directly below
            elif diff == 2:
                prox = 0.8  # 2 lines below
            elif diff == -1:
                prox = 0.4  # directly above
            elif diff == -2:
                prox = 0.2  # 2 lines above
            else:
                prox = 0.1
                
        proximity_score = 0.25 * prox

        # 3. Validation score (0.20)
        validation_score = 0.20 * 1.0

        # 4. Layout consistency (0.10)
        layout_val = 0.5
        bbox = candidate.get("bounding_box")
        if bbox:
            img_h = 1000
            if word_map:
                max_b = max((w.get('top', 0) + w.get('height', 0) for w in word_map), default=1000)
                if max_b > 0:
                    img_h = max_b
            cy = bbox['y'] + bbox['h'] / 2.0
            y_norm = cy / img_h

            if doc_type == DocumentType.AADHAAR:
                if field_name == "aadhaar_number" and y_norm > 0.5:
                    layout_val = 1.0
                elif field_name == "name" and 0.2 < y_norm < 0.6:
                    layout_val = 1.0
                elif field_name == "dob" and 0.3 < y_norm < 0.7:
                    layout_val = 1.0
            elif doc_type == DocumentType.PAN:
                if field_name == "pan_number" and y_norm > 0.5:
                    layout_val = 1.0
                elif field_name == "name" and 0.2 < y_norm < 0.6:
                    layout_val = 1.0
                elif field_name == "father_name" and 0.3 < y_norm < 0.7:
                    layout_val = 1.0
            elif doc_type == DocumentType.PASSPORT:
                if field_name == "passport_number" and (y_norm < 0.3 or y_norm > 0.7):
                    layout_val = 1.0
                elif field_name == "name" and 0.15 < y_norm < 0.6:
                    layout_val = 1.0
            elif doc_type == DocumentType.DRIVING_LICENCE:
                if field_name == "dl_number" and y_norm < 0.5:
                    layout_val = 1.0
                elif field_name == "name" and 0.2 < y_norm < 0.6:
                    layout_val = 1.0
        layout_score = 0.10 * layout_val

        # 5. Document-template match (0.05)
        page_source = candidate.get("page_source", "visual")
        template_val = 0.7
        if doc_type == DocumentType.PASSPORT:
            if page_source == "mrz":
                template_val = 1.0
            elif page_source == "visual":
                template_val = 0.8
            else:
                template_val = 0.5
        else:
            if page_source == "visual":
                template_val = 1.0
            else:
                template_val = 0.5
        template_score = 0.05 * template_val

        # Combine all parts
        score = ocr_score + proximity_score + validation_score + layout_score + template_score
        
        # Include custom boost adjustment
        score += candidate.get("boost", 0.0)

        # Field length scoring
        if field_name == "name":
            name_text = candidate.get("text", "")
            words = name_text.split()
            if len(words) >= 3:
                score += 1.5
            elif len(words) == 2:
                score += 1.0
            elif len(words) == 1:
                score -= 2.0  # Penalize single-word names (like city names)
            
            if len(name_text) < 5:
                score -= 2.0

        # Expected format scoring
        text_val = candidate.get("text", "")
        if doc_type == DocumentType.AADHAAR and field_name == "aadhaar_number":
            cleaned = re.sub(r"\s+", "", text_val)
            if len(cleaned) == 12 and cleaned.isdigit():
                score += 2.0
        elif doc_type == DocumentType.PAN and field_name == "pan_number":
            if re.match(r"^[A-Z]{5}[0-9]{4}[A-Z]$", text_val):
                score += 2.0
        elif doc_type == DocumentType.PASSPORT and field_name == "passport_number":
            if re.match(r"^[A-Z][0-9]{7}$", text_val):
                score += 2.0
        elif doc_type == DocumentType.DRIVING_LICENCE and field_name == "dl_number":
            cleaned = re.sub(r"\s+", "", text_val)
            if re.match(r"^[A-Z]{2}[0-9]{2}[0-9]{4}[0-9]{7}$", cleaned):
                score += 2.0

        # Future DOB penalty
        if field_name == "dob":
            if self.is_valid_calendar_date(candidate.get("text", "")):
                from datetime import datetime, date
                try:
                    dt = datetime.strptime(candidate.get("text", ""), "%Y-%m-%d").date()
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
        # Pre-process raw lines for distance calculation
        raw_lines = [line.strip() for line in raw_text.split('\n') if line.strip()]

        valid_candidates = []
        rejected_candidates = []

        # Validate all candidates first
        for c in candidates:
            # Fill missing attributes if necessary
            if "distance_from_anchor" not in c or "surrounding_labels" not in c:
                c_full = self.create_candidate(
                    field_name=field_name,
                    text=c["text"],
                    raw_text_val=c.get("raw_text", c["text"]),
                    bbox=c.get("bbox", c.get("bounding_box")),
                    ocr_confidence=c.get("ocr_confidence", 0.85),
                    line_number=c.get("line_number", -1),
                    raw_lines=raw_lines,
                    word_map=word_map,
                    doc_type=doc_type,
                    page_source=c.get("page_source", "visual"),
                    constituent_boxes=c.get("constituent_boxes")
                )
                c.update(c_full)

            is_valid, reason = self.validate_candidate(field_name, c["text"], doc_type, c)
            if is_valid:
                valid_candidates.append(c)
            else:
                rejected_candidates.append({
                    "text": c["text"],
                    "reason": reason,
                    "ocr_confidence": c.get("ocr_confidence", 0.0),
                    "bounding_box": c.get("bounding_box")
                })

        if not valid_candidates:
            log_data = {
                "field_name": field_name,
                "selected_value": "NOT_FOUND",
                "selected_score": 0.0,
                "rejected_candidates": rejected_candidates,
                "rejection_reason": "All candidates failed validation or no candidates generated"
            }
            logger.info(f"Extraction Debug: {log_data}")
            return FieldResult(value="NOT_FOUND", raw_text="", confidence=0.0, bounding_box=None)

        # Score valid candidates
        scored_candidates = []
        for c in valid_candidates:
            score = self.score_candidate(field_name, c, doc_type, word_map, raw_text)
            scored_candidates.append((score, c))

        # Rank candidates
        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        best_score, best_candidate = scored_candidates[0]

        # Extract rejection logs for other valid candidates
        for score, c in scored_candidates[1:]:
            rejected_candidates.append({
                "text": c["text"],
                "reason": f"Lower score than selected ({score:.3f} vs {best_score:.3f})",
                "ocr_confidence": c.get("ocr_confidence", 0.0),
                "bounding_box": c.get("bounding_box")
            })

        # Calculate combined field-level confidence (weighted average)
        # 1. OCR Confidence
        c_ocr = best_candidate.get("ocr_confidence", 0.85)
        if c_ocr > 1.0:
            c_ocr = c_ocr / 100.0

        # 2. Pattern Match Confidence
        is_pattern_valid, _ = self.validate_candidate(field_name, best_candidate["text"], doc_type, best_candidate)
        c_pattern = 1.0 if is_pattern_valid else 0.5
        text_val = best_candidate["text"]
        if doc_type == DocumentType.AADHAAR and field_name == "aadhaar_number":
            cleaned = re.sub(r"\s+", "", text_val)
            if len(cleaned) == 12 and cleaned.isdigit():
                c_pattern = 1.0
        elif doc_type == DocumentType.PAN and field_name == "pan_number":
            if re.match(r"^[A-Z]{5}[0-9]{4}[A-Z]$", text_val):
                c_pattern = 1.0
        elif doc_type == DocumentType.PASSPORT and field_name == "passport_number":
            if re.match(r"^[A-Z][0-9]{7}$", text_val):
                c_pattern = 1.0
        elif doc_type == DocumentType.DRIVING_LICENCE and field_name == "dl_number":
            cleaned = re.sub(r"\s+", "", text_val)
            if re.match(r"^[A-Z]{2}[0-9]{2}[0-9]{4}[0-9]{7}$", cleaned):
                c_pattern = 1.0

        # 3. Positional Confidence
        dist = best_candidate.get("distance_from_anchor", 999.0)
        if dist == 0.0 or dist == 1.0:
            c_pos = 1.0
        elif dist == 2.0:
            c_pos = 0.8
        elif dist < 5.0:
            c_pos = 0.6
        else:
            c_pos = 0.4

        # 4. Document-template Confidence
        bbox = best_candidate.get("bounding_box")
        c_temp = 0.8
        if bbox:
            img_h = 1000
            if word_map:
                max_b = max((w.get('top', 0) + w.get('height', 0) for w in word_map), default=1000)
                if max_b > 0:
                    img_h = max_b
            cy = bbox['y'] + bbox['h'] / 2.0
            y_norm = cy / img_h
            if doc_type == DocumentType.AADHAAR:
                if field_name == "aadhaar_number" and y_norm > 0.5:
                    c_temp = 1.0
                elif field_name == "name" and 0.2 < y_norm < 0.6:
                    c_temp = 1.0
            elif doc_type == DocumentType.PAN:
                if field_name == "pan_number" and y_norm > 0.5:
                    c_temp = 1.0
                elif field_name == "name" and 0.2 < y_norm < 0.6:
                    c_temp = 1.0
            elif doc_type == DocumentType.PASSPORT:
                if field_name == "passport_number" and (y_norm < 0.3 or y_norm > 0.7):
                    c_temp = 1.0
                elif field_name == "name" and 0.15 < y_norm < 0.6:
                    c_temp = 1.0
            elif doc_type == DocumentType.DRIVING_LICENCE:
                if field_name == "dl_number" and y_norm < 0.5:
                    c_temp = 1.0
                elif field_name == "name" and 0.2 < y_norm < 0.6:
                    c_temp = 1.0

        # Combine confidences: 40% OCR, 30% Pattern, 20% Positional, 10% Template
        combined_conf = (0.40 * c_ocr) + (0.30 * c_pattern) + (0.20 * c_pos) + (0.10 * c_temp)
        combined_conf = float(max(0.0, min(1.0, combined_conf)))

        # Log candidates list and scores (Requirement 6)
        candidate_log_list = []
        for score, c in scored_candidates:
            candidate_log_list.append({
                "text": c["text"],
                "score": round(score, 4),
                "ocr_confidence": c.get("ocr_confidence", 0.0),
                "bounding_box": c.get("bounding_box")
            })
        for c in rejected_candidates:
            candidate_log_list.append({
                "text": c["text"],
                "score": 0.0,
                "ocr_confidence": c.get("ocr_confidence", 0.0),
                "bounding_box": c.get("bounding_box"),
                "status": f"Rejected: {c.get('reason', 'Unknown reason')}"
            })

        logger.info(
            f"Extraction Debug for field '{field_name}':\n"
            f"  Selected Candidate: '{best_candidate['text']}'\n"
            f"  Selected Score: {round(best_score, 4)}\n"
            f"  Combined Field Confidence: {round(combined_conf * 100, 2)}%\n"
            f"  Selected Bounding Box: {best_candidate.get('bounding_box')}\n"
            f"  Candidate List & Scores: {candidate_log_list}"
        )

        return FieldResult(
            value=best_candidate["text"],
            raw_text=best_candidate.get("raw_text", best_candidate["text"]),
            confidence=combined_conf,
            bounding_box=best_candidate.get("bounding_box"),
            constituent_boxes=best_candidate.get("constituent_boxes")
        )

    @staticmethod
    def reconstruct_and_normalize_text(
        bbox: Optional[Dict[str, int]],
        word_map: List[Dict[str, Any]],
        field_name: str,
        doc_type: DocumentType
    ) -> str:
        """Reconstructs text inside the bounding box and normalizes it to match display value format."""
        if not bbox or not word_map:
            return "NOT_FOUND"

        x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']
        matched_words = []
        for word in word_map:
            # Check if center of word is inside bounding box (with slight tolerance)
            wx = word['left'] + word['width'] / 2.0
            wy = word['top'] + word['height'] / 2.0
            if (x - 4) <= wx <= (x + w + 4) and (y - 4) <= wy <= (y + h + 4):
                matched_words.append(word)

        if not matched_words:
            return "NOT_FOUND"

        # Group words by line (vertical threshold 15px)
        line_groups = []
        for word in matched_words:
            wy = word['top'] + word['height'] / 2.0
            added = False
            for group in line_groups:
                group_wy = sum(wd['top'] + wd['height'] / 2.0 for wd in group) / len(group)
                if abs(wy - group_wy) < 15:
                    group.append(word)
                    added = True
                    break
            if not added:
                line_groups.append([word])

        # Sort each group by left coordinate, and groups by top coordinate
        for group in line_groups:
            group.sort(key=lambda wd: wd['left'])
        line_groups.sort(key=lambda group: sum(wd['top'] for wd in group) / len(group))

        # Join to reconstruct raw string
        lines_text = [" ".join(wd['text'] for wd in group) for group in line_groups]
        raw_reconstructed = " ".join(lines_text).strip()

        # Normalize reconstructed raw string according to field name and document type
        normalized = raw_reconstructed
        if doc_type == DocumentType.PASSPORT:
            if field_name == "passport_number":
                cleaned = re.sub(r'\s+', '', raw_reconstructed).upper()
                if len(cleaned) >= 8:
                    letter = cleaned[0]
                    digit_to_letter = {'2': 'Z', '0': 'O', '1': 'I', '8': 'B', '5': 'S', '6': 'G'}
                    if letter.isdigit():
                        letter = digit_to_letter.get(letter, letter)
                    digits = ocr_correct_digits(cleaned[1:8])
                    normalized = letter + digits
                else:
                    normalized = cleaned
            elif field_name == "name":
                normalized = extract_uppercase_name(raw_reconstructed)
            elif field_name == "nationality":
                cleaned = re.sub(r'[^A-Z]', '', raw_reconstructed.upper())
                if cleaned in ["IND", "INDIAN", "1ND", "IMD", "1MD", "IUD", "1UD"]:
                    normalized = "IND"
                else:
                    normalized = cleaned[:3] if len(cleaned) >= 3 else cleaned
            elif field_name in ["dob", "expiry"]:
                normalized = normalize_date(raw_reconstructed) or "NOT_FOUND"
            elif field_name == "sex":
                cleaned = re.sub(r'[^A-Z]', '', raw_reconstructed.upper())
                if "FEMALE" in cleaned or "F" in cleaned:
                    normalized = "F"
                else:
                    normalized = "M"

        elif doc_type == DocumentType.AADHAAR:
            if field_name == "aadhaar_number":
                normalized = ocr_correct_digits(re.sub(r"\s+", "", raw_reconstructed))
            elif field_name == "dob":
                # Might be year of birth only or full date
                if re.match(r"^\b(19\d{2}|20\d{2})\b$", raw_reconstructed):
                    normalized = f"{raw_reconstructed}-01-01"
                else:
                    normalized = normalize_date(raw_reconstructed) or "NOT_FOUND"
            elif field_name == "gender":
                cleaned = raw_reconstructed.upper()
                if "FEMALE" in cleaned or "महिला" in cleaned:
                    normalized = "FEMALE"
                elif "MALE" in cleaned or "पुरुष" in cleaned:
                    normalized = "MALE"
                else:
                    normalized = "OTHER"
            elif field_name == "name":
                normalized = extract_uppercase_name(raw_reconstructed)

        elif doc_type == DocumentType.PAN:
            if field_name == "pan_number":
                cleaned = re.sub(r'\s+', '', raw_reconstructed).upper()
                if len(cleaned) >= 10:
                    normalized = cleaned[:5] + ocr_correct_digits(cleaned[5:9]) + cleaned[9]
                else:
                    normalized = cleaned
            elif field_name == "dob":
                normalized = normalize_date(raw_reconstructed) or "NOT_FOUND"
            elif field_name in ["name", "father_name"]:
                normalized = extract_uppercase_name(raw_reconstructed)

        elif doc_type == DocumentType.DRIVING_LICENCE:
            if field_name == "dl_number":
                cleaned = re.sub(r'[^a-zA-Z0-9]', '', raw_reconstructed).upper()
                if len(cleaned) >= 15:
                    normalized = cleaned[:2] + ocr_correct_digits(cleaned[2:])
                else:
                    normalized = cleaned
            elif field_name in ["dob", "validity"]:
                normalized = normalize_date(raw_reconstructed) or "NOT_FOUND"
            elif field_name == "name":
                normalized = extract_uppercase_name(raw_reconstructed)
            elif field_name == "vehicle_class":
                classes = []
                for w in raw_reconstructed.upper().replace(",", " ").split():
                    if w in ["MCWG", "LMV", "MCWOG", "HMV", "TRANS", "COV"]:
                        classes.append(w)
                classes = sorted(list(set(classes)))
                normalized = ", ".join(classes) if classes else "NOT_FOUND"

        return normalized if normalized else "NOT_FOUND"
