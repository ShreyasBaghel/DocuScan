import re
from typing import Dict, List, Any
from extractors.base_extractor import BaseExtractor
from utils.document_packet import FieldResult, DocumentType
from utils.string_utils import normalize_date, clean_whitespace, extract_uppercase_name, is_valid_name, ocr_correct_digits
from utils.logger import get_logger

logger = get_logger("dl_extractor")

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
                
        results["validity"] = self.select_best_candidate("validity", cands_val_uniq, DocumentType.DRIVING_LICENCE, word_map, raw_text)
        
        # 4. Name candidates
        
        # Identify first address line index to partition layout regions
        first_address_line_idx = len(lines)
        for idx, line in enumerate(lines):
            l_lower = line.lower()
            if (any(k in l_lower for k in ["add:", "address:", "add ", "address ", "a/p", "at/post", "at post", "at & post", "at/po", "at po"]) or
                re.search(r'\b(tal|taluka|dist|district|pincode|pin|village|town|road|street|lane)\b', l_lower)):
                first_address_line_idx = idx
                break

        name_lbl_indices = []
        for idx, line in enumerate(lines):
            l_lower = line.lower()
            # Do not treat parent name labels or document labels as name labels
            if any(k in l_lower for k in ["name", "नाम", "holder"]):
                if not any(k in l_lower for k in ["father", "husband", "licence", "license", "sdmw", "s/d/w", "s/o", "d/o", "w/o", "son", "daughter", "wife"]):
                    name_lbl_indices.append(idx)
                
        cands_name = []
        # Restrict Name candidate generation to the name region (above the address boundary)
        for idx in range(min(first_address_line_idx, len(lines))):
            line = lines[idx]
            
            # Exclude headers, ID numbers, parent/spouse names, and other document labels
            if any(h in line.lower() for h in [
                "driving", "licence", "license", "authority", "union", "india", "transport", "dl no", "lic no", "licence no", "lic.no",
                "sdmw", "s/d/w", "s/o", "d/o", "w/o", "father", "husband", "wife", "mother", "son of", "daughter of", "wife of", "parents"
            ]):
                continue
                
            cand_name = extract_uppercase_name(line)
            if cand_name == "NOT_FOUND" or not is_valid_name(cand_name, dl_num):
                continue
                
            bbox = self.merge_bounding_boxes(line, word_map)
            ocr_conf = self.get_field_confidence(cand_name, word_map)
            
            # Distance boost from the nearest name label
            boost = 0.0
            if name_lbl_indices:
                min_lbl_dist = min(abs(idx - lbl_idx) for lbl_idx in name_lbl_indices)
                if min_lbl_dist == 0:
                    boost += 1.5
                elif min_lbl_dist == 1:
                    boost += 2.0
                elif min_lbl_dist == 2:
                    boost += 1.0

            cands_name.append({
                "text": cand_name,
                "raw_text": line,
                "bbox": bbox,
                "ocr_confidence": ocr_conf,
                "page_source": "visual",
                "boost": boost,
                "line_number": idx
            })
            
            # Also try combining adjacent uppercase lines (Multi-line OCR reconstruction)
            if idx + 1 < first_address_line_idx:
                next_line = lines[idx+1]
                if not any(h in next_line.lower() for h in [
                    "driving", "licence", "license", "authority", "union", "india", "transport", "dl no", "lic no", "licence no", "lic.no",
                    "sdmw", "s/d/w", "s/o", "d/o", "w/o", "father", "husband", "wife", "mother", "son of", "daughter of", "wife of", "parents"
                ]):
                    cand_next = extract_uppercase_name(next_line)
                    if cand_next != "NOT_FOUND" and is_valid_name(cand_name + " " + cand_next, dl_num):
                        cands_name.append({
                            "text": f"{cand_name} {cand_next}",
                            "raw_text": f"{line} {next_line}",
                            "bbox": self.merge_bounding_boxes(f"{line} {next_line}", word_map),
                            "ocr_confidence": (ocr_conf + self.get_field_confidence(cand_next, word_map)) / 2.0,
                            "page_source": "visual",
                            "boost": boost,
                            "line_number": idx
                        })
            
        seen = set()
        cands_name_uniq = []
        for c in cands_name:
            if c["text"] not in seen:
                seen.add(c["text"])
                cands_name_uniq.append(c)
                
        results["name"] = self.select_best_candidate("name", cands_name_uniq, DocumentType.DRIVING_LICENCE, word_map, raw_text)
        
        # 5. Vehicle class candidates
        cands_vc = []
        for idx, line in enumerate(lines):
            line_upper = line.upper()
            found_classes = []
            for c_cls in ["MCWG", "LMV", "MCWOG", "HMV", "TRANS", "COV"]:
                if c_cls in line_upper:
                    found_classes.append(c_cls)
            if found_classes:
                found_classes = sorted(list(set(found_classes)))
                val = ", ".join(found_classes)
                cands_vc.append({
                    "text": val,
                    "raw_text": line,
                    "bbox": self.merge_bounding_boxes(found_classes[0], word_map),
                    "ocr_confidence": 0.90,
                    "page_source": "visual",
                    "line_number": idx
                })
                
        seen = set()
        cands_vc_uniq = []
        for c in cands_vc:
            if c["text"] not in seen:
                seen.add(c["text"])
                cands_vc_uniq.append(c)
                
        results["vehicle_class"] = self.select_best_candidate("vehicle_class", cands_vc_uniq, DocumentType.DRIVING_LICENCE, word_map, raw_text)
        
        return results

    def validate_candidate(self, field_name: str, text: str, doc_type: DocumentType, candidate: Dict[str, Any]) -> tuple[bool, str]:
        # Run super validations first
        is_valid, reason = super().validate_candidate(field_name, text, doc_type, candidate)
        if not is_valid:
            return False, reason
            
        if field_name == "name":
            text_upper = text.upper()
            # Strict blacklist keywords rejection
            blacklist_pattern = re.compile(
                r'\b(add|address|a/p|ajp|at|post|tal|taluka|dist|district|road|street|lane|village|pin|pincode)\b', 
                re.IGNORECASE
            )
            if blacklist_pattern.search(text_upper):
                return False, f"DL Name contains blacklisted address keyword"
                
            # Reject if name contains digits
            if any(c.isdigit() for c in text):
                return False, "DL Name contains digits"
                
        return True, "PASS"

    def score_candidate(
        self,
        field_name: str,
        candidate: Dict[str, Any],
        doc_type: DocumentType,
        word_map: List[Dict[str, Any]],
        raw_text: str
    ) -> float:
        # Get base score first
        score = super().score_candidate(field_name, candidate, doc_type, word_map, raw_text)
        
        if field_name == "name":
            cand_text = candidate.get("text", "")
            cand_text_upper = cand_text.upper()
            line_number = candidate.get("line_number", 0)
            
            # Positive signals:
            # 1. 2-5 words length
            words = cand_text.split()
            if 2 <= len(words) <= 5:
                score += 2.0
            else:
                score -= 3.0
                
            # Negative signals (blacklist and location names):
            blacklist_pattern = re.compile(
                r'\b(add|address|a/p|ajp|at|post|tal|taluka|dist|district|road|street|lane|village|pin|pincode)\b', 
                re.IGNORECASE
            )
            if blacklist_pattern.search(cand_text_upper):
                score -= 50.0
                
            location_names = {
                "MUMBAI", "DELHI", "AHMEDABAD", "BANGALORE", "KOLKATA", "CHENNAI", 
                "HYDERABAD", "PUNE", "JAIPUR", "LUCKNOW", "SURAT", "PATNA", "INDORE", 
                "THANE", "BHOPAL", "VISAKHAPATNAM", "VADODARA", "GANDHINAGAR", "PANAJI",
                "MAHARASHTRA", "GUJARAT", "KARNATAKA", "BHOR", "MAZGAON", "INDIA", "STATE"
            }
            cand_words = set(re.sub(r'[^A-Z\s]', '', cand_text_upper).split())
            if cand_words.intersection(location_names):
                score -= 20.0
                
            if any(c.isdigit() for c in cand_text):
                score -= 20.0
                
            # Appears below Add/Address field
            lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
            first_address_line_idx = len(lines)
            for idx, line in enumerate(lines):
                l_lower = line.lower()
                if (any(k in l_lower for k in ["add:", "address:", "add ", "address ", "a/p", "at/post", "at post", "at & post", "at/po", "at po"]) or
                    re.search(r'\b(tal|taluka|dist|district|pincode|pin|village|town|road|street|lane)\b', l_lower)):
                    first_address_line_idx = idx
                    break
            if line_number >= first_address_line_idx:
                score -= 50.0
                
            # Appears near PIN code (within 2 lines of 6-digit number)
            pin_line_idx = -1
            for idx, line in enumerate(lines):
                if re.search(r'\b\d{6}\b', line):
                    pin_line_idx = idx
                    break
            if pin_line_idx != -1 and abs(line_number - pin_line_idx) <= 2:
                score -= 20.0
                
        return score

    def select_best_candidate(
        self,
        field_name: str,
        candidates: List[Dict[str, Any]],
        doc_type: DocumentType,
        word_map: List[Dict[str, Any]],
        raw_text: str
    ) -> FieldResult:
        # Call base select_best_candidate to perform the actual selection
        selected_result = super().select_best_candidate(field_name, candidates, doc_type, word_map, raw_text)
        
        # Calculate confidences and log debugging details for requirement 6 and 7
        candidate_scores = []
        candidate_coords = []
        candidate_texts = []
        candidate_ocr_confs = []
        candidate_layout_confs = []
        candidate_pattern_confs = []
        
        lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
        
        # Find first address line and PIN line for layout/pattern calculations
        first_address_line_idx = len(lines)
        for idx, line in enumerate(lines):
            l_lower = line.lower()
            if (any(k in l_lower for k in ["add:", "address:", "add ", "address ", "a/p", "at/post", "at post", "at & post", "at/po", "at po"]) or
                re.search(r'\b(tal|taluka|dist|district|pincode|pin|village|town|road|street|lane)\b', l_lower)):
                first_address_line_idx = idx
                break
                
        pin_line_idx = -1
        for idx, line in enumerate(lines):
            if re.search(r'\b\d{6}\b', line):
                pin_line_idx = idx
                break
                
        for c in candidates:
            score = self.score_candidate(field_name, c, doc_type, word_map, raw_text)
            candidate_scores.append(round(score, 4))
            candidate_coords.append(c.get("bounding_box") or c.get("bbox"))
            candidate_texts.append(c.get("text"))
            
            # OCR confidence
            c_ocr = c.get("ocr_confidence", 0.85)
            if c_ocr > 1.0:
                c_ocr = c_ocr / 100.0
            candidate_ocr_confs.append(round(c_ocr, 4))
            
            # Layout confidence
            line_num = c.get("line_number", 0)
            if field_name == "name":
                in_region = 1.0 if line_num < first_address_line_idx else 0.0
                dist_pin = 1.0 if pin_line_idx == -1 or abs(line_num - pin_line_idx) > 2 else 0.2
                layout_conf = 0.7 * in_region + 0.3 * dist_pin
            else:
                # Default layout confidence estimation
                layout_conf = 0.8
            candidate_layout_confs.append(round(layout_conf, 4))
            
            # Pattern confidence
            text_val = c.get("text", "")
            if field_name == "name":
                words = text_val.split()
                has_digits = any(char.isdigit() for char in text_val)
                blacklist_pattern = re.compile(r'\b(add|address|a/p|ajp|at|post|tal|taluka|dist|district|road|street|lane|village|pin|pincode)\b', re.IGNORECASE)
                has_blacklist = bool(blacklist_pattern.search(text_val))
                if 2 <= len(words) <= 5 and not has_digits and not has_blacklist:
                    pattern_conf = 1.0
                elif len(words) == 1 and not has_digits and not has_blacklist:
                    pattern_conf = 0.5
                else:
                    pattern_conf = 0.1
            elif field_name == "dl_number":
                cleaned = re.sub(r"\s+", "", text_val)
                pattern_conf = 1.0 if re.match(r"^[A-Z]{2}[0-9]{2}[0-9]{4}[0-9]{7}$", cleaned) else 0.5
            else:
                pattern_conf = 0.9
            candidate_pattern_confs.append(round(pattern_conf, 4))
            
        logger.info(
            f"Driving Licence Field Interpretation Debug:\n"
            f"  field name: {field_name}\n"
            f"  candidate texts: {candidate_texts}\n"
            f"  candidate coordinates: {candidate_coords}\n"
            f"  candidate scores: {candidate_scores}\n"
            f"  ocr confidence: {candidate_ocr_confs}\n"
            f"  layout confidence: {candidate_layout_confs}\n"
            f"  pattern confidence: {candidate_pattern_confs}\n"
            f"  selected value: {selected_result.value}"
        )
        
        return selected_result
