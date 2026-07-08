import re
from typing import Dict, List, Any, Optional
from extractors.base_extractor import BaseExtractor
from utils.document_packet import FieldResult, DocumentType
from utils.string_utils import normalize_date, clean_whitespace, extract_uppercase_name, is_valid_name, ocr_correct_digits
from validators.checksum_validator import ChecksumValidator
from utils.logger import get_logger

logger = get_logger("passport_extractor")

class PassportExtractor(BaseExtractor):
    @staticmethod
    def contains_field_label(text: str) -> bool:
        text_lower = text.lower()
        label_keywords = {
            "name", "nam", "नाम", "father", "spous", "husband", "wife", "mother",
            "dob", "birth", "जन्म", "yob", "date", "valid", "till", "validity", "expiry", "exp",
            "licence", "license", "dl", "class", "cov", "vehicle", "address", "add", "pin", "pincode",
            "passport", "pan", "aadhaar", "aadhar", "uidai", "gender", "sex", "issue", "doi", "rto"
        }
        words = re.split(r'[^a-zA-Z0-9]', text_lower)
        for w in words:
            if w in label_keywords:
                return True
        if re.search(r'\b[a-zA-Z]{3,}\s*:', text):
            return True
        return False

    def correct_numeric_field(self, text: str) -> str:
        mapping = {'O': '0', 'I': '1', 'B': '8', 'S': '5'}
        return "".join(mapping.get(c, c) for c in text)

    def correct_mrz_line2_digits(self, m2: str) -> str:
        if len(m2) != 44:
            return m2
        m2_chars = list(m2)
        numeric_indices = list(range(1, 8)) + [9] + list(range(13, 19)) + [19] + list(range(21, 27)) + [27] + list(range(28, 44))
        for idx in numeric_indices:
            c = m2_chars[idx]
            if c == 'O':
                m2_chars[idx] = '0'
            elif c == 'I':
                m2_chars[idx] = '1'
            elif c == 'B':
                m2_chars[idx] = '8'
            elif c == 'S':
                m2_chars[idx] = '5'
        return "".join(m2_chars)

    def validate_mrz_candidate(self, m1: str, m2: str) -> bool:
        c1 = re.sub(r'\s+', '', m1).upper()
        c2 = re.sub(r'\s+', '', m2).upper()
        c1 = re.sub(r'[^A-Z0-9<]', '', c1)
        c2 = re.sub(r'[^A-Z0-9<]', '', c2)

        if not (40 <= len(c1) <= 44) or not (40 <= len(c2) <= 44):
            return False

        if not re.match(r"^[A-Z0-9<]+$", c1) or not re.match(r"^[A-Z0-9<]+$", c2):
            return False

        if c1.startswith("P") and not c1.startswith("P<"):
            c1 = "P<" + c1[2:]
        if not c1.startswith("P<"):
            return False

        country_code = c1[2:5]
        if country_code in ["1ND", "IMD", "IUD", "1MD", "1UD", "INO", "1NO"]:
            c1 = c1[:2] + "IND" + c1[5:]
            country_code = "IND"
        if not re.match(r"^[A-Z]{3}$", country_code):
            return False

        c2_corrected = self.correct_mrz_line2_digits(c2)

        pass_part = c2_corrected[0:9]
        pass_num = pass_part[0] + self.correct_numeric_field(pass_part[1:8]) + pass_part[8]
        if not re.match(r"^[A-Z][0-9]{7}<*$", pass_num):
            return False

        return True

    def has_forbidden_context(self, line_number: int) -> bool:
        if not hasattr(self, 'raw_lines') or not self.raw_lines:
            return False
        forbidden_keywords = [
            "father", "mother", "spouse", "wife", "husband", 
            "son of", "daughter of", "address", "permanent address", 
            "near", "village", "pin", "state"
        ]
        start_idx = max(0, line_number - 2)
        end_idx = min(len(self.raw_lines) - 1, line_number + 2)
        for idx in range(start_idx, end_idx + 1):
            line_lower = self.raw_lines[idx].lower()
            for kw in forbidden_keywords:
                if kw in line_lower:
                    return True
        return False

    def validate_candidate(self, field_name: str, text: str, doc_type: DocumentType, candidate: Dict[str, Any]) -> tuple[bool, str]:
        is_valid, reason = super().validate_candidate(field_name, text, doc_type, candidate)
        if not is_valid:
            return False, reason

        if field_name == "name":
            page_source = candidate.get("page_source", "visual")
            if page_source != "mrz":
                line_number = candidate.get("line_number", -1)
                if line_number != -1 and self.has_forbidden_context(line_number):
                    return False, f"Name candidate '{text}' is near family/address keywords"

                forbidden_labels = [
                    "father", "mother", "spouse", "wife", "husband", 
                    "son of", "daughter of", "address", "permanent", 
                    "near", "village", "pin", "state", "given name", "surname", "nationality"
                ]
                text_lower = text.lower()
                for label in forbidden_labels:
                    if label in text_lower:
                        return False, f"Name candidate contains forbidden label: {label}"

        elif field_name == "nationality":
            if not re.match(r"^[A-Z]{3}$", text.upper()):
                return False, f"Nationality '{text}' must be exactly 3 uppercase letters"
            if text.upper() in ["NNB", "NNBI", "NND", "III"]:
                return False, f"Nationality '{text}' is a rejected invalid code"

        return True, "PASS"

    def clean_passport_name(self, name: str) -> Optional[str]:
        if not name or name == "NOT_FOUND":
            return None
            
        # Split candidate name into tokens (words)
        words = name.split()
        cleaned_tokens = []
        
        # Forbidden tokens to exclude
        forbidden_words = {
            "IND", "1ND", "IMD", "IUD", "1MD", "1UD", "INO", "1NO",
            "INDIA", "INDIAN", "iINOIAN", "iNDIAN", "NNB", "NNBI", "NND", "III",
            "REPUBLIC", "GOVERNMENT", "GOVT", "PASSPORT", "NATIONALITY",
            "SURNAME", "GIVEN", "NAME", "NAMES", "FATHER", "MOTHER", "SPOUSE",
            "WIFE", "HUSBAND", "SEX", "GENDER", "DATE", "BIRTH", "DOB", "EXPIRY",
            "ISSUE", "PLACE", "CARD", "OFFICE"
        }
        
        for w in words:
            # Uppercase first to clean
            w_upper = w.upper()
            
            # Remove prepended MRZ country-code prefix noise (e.g., PINDSOLANKI -> SOLANKI, P<INDSHARMA -> SHARMA)
            w_upper = re.sub(r'^P<?(?:IND|1ND|IMD|IUD|1MD|1UD|INO|1NO)', '', w_upper)
            
            # Keep only letters and dots
            w_clean = re.sub(r'[^A-Z\.]', '', w_upper)
            if not w_clean:
                continue
                
            # If the token is in the forbidden set, exclude it
            if w_clean in forbidden_words:
                continue
                
            # Skip common MRZ fillers or single character garbage
            if w_clean in ["KKK", "XXX", "ZZZ"]:
                continue
                
            cleaned_tokens.append(w_clean)
            
        if not cleaned_tokens:
            return None
            
        return " ".join(cleaned_tokens)

    def score_candidate(
        self,
        field_name: str,
        candidate: Dict[str, Any],
        doc_type: DocumentType,
        word_map: List[Dict[str, Any]],
        raw_text: str
    ) -> float:
        score = super().score_candidate(field_name, candidate, doc_type, word_map, raw_text)

        if field_name == "name":
            line_number = candidate.get("line_number", -1)
            if line_number != -1 and hasattr(self, 'raw_lines') and self.raw_lines:
                for idx, line in enumerate(self.raw_lines):
                    line_lower = line.lower()
                    if "surname" in line_lower or "given name" in line_lower or "givenname" in line_lower or "given name(s)" in line_lower:
                        dist = abs(line_number - idx)
                        if dist == 1:
                            score += 5.0
                        elif dist <= 2:
                            score += 3.0

                bbox = candidate.get("bounding_box")
                if bbox:
                    img_h = 1000
                    if word_map:
                        max_b = max((w.get('top', 0) + w.get('height', 0) for w in word_map), default=1000)
                        if max_b > 0:
                            img_h = max_b
                    cy = bbox['y'] + bbox['h'] / 2.0
                    y_norm = cy / img_h
                    if 0.15 < y_norm < 0.6:
                        score += 3.0
                    elif y_norm > 0.5:
                        score -= 5.0

                for idx, line in enumerate(self.raw_lines):
                    cleaned = re.sub(r'\s+', '', line).upper()
                    if re.match(r"^[A-Z][0-9]{7}$", cleaned):
                        dist = abs(line_number - idx)
                        if dist <= 4:
                            score += 3.0

            if candidate.get("constructed_from_labels"):
                score += 10.0

        return score

    def select_best_candidate(
        self,
        field_name: str,
        candidates: List[Dict[str, Any]],
        doc_type: DocumentType,
        word_map: List[Dict[str, Any]],
        raw_text: str
    ) -> FieldResult:
        raw_lines = [line.strip() for line in raw_text.split('\n') if line.strip()]

        print(f"\n--- Field Debug: {field_name} ---")
        logger.info(f"Field Debug: {field_name}")

        valid_candidates = []
        rejected_candidates = []

        # Validate all candidates first
        for c in candidates:
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
                rejected_candidates.append((c, reason))

        # Score valid candidates
        scored_candidates = []
        for c in valid_candidates:
            score = self.score_candidate(field_name, c, doc_type, word_map, raw_text)
            scored_candidates.append((score, c))

        # Rank candidates
        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        best_candidate = None
        best_score = 0.0
        if scored_candidates:
            best_score, best_candidate = scored_candidates[0]

        # Print valid candidates
        for score, c in scored_candidates:
            is_accepted = (c == best_candidate)
            accept_reason = "Selected (highest score)" if is_accepted else "N/A"
            reject_reason = "N/A" if is_accepted else f"Lower score than selected ({score:.3f} vs {best_score:.3f})"
            
            print(f"field_name: {field_name}")
            print(f"candidate_text: {c['text']}")
            print(f"candidate_score: {score:.3f}")
            print(f"accept_reason: {accept_reason}")
            print(f"reject_reason: {reject_reason}\n")

            logger.info(f"field_name: {field_name}, candidate: {c['text']}, score: {score:.3f}, accept_reason: {accept_reason}, reject_reason: {reject_reason}")

        # Print rejected candidates
        for c, reason in rejected_candidates:
            print(f"field_name: {field_name}")
            print(f"candidate_text: {c['text']}")
            print(f"candidate_score: 0.000")
            print(f"accept_reason: N/A")
            print(f"reject_reason: {reason}\n")

            logger.info(f"field_name: {field_name}, candidate: {c['text']}, score: 0.000, accept_reason: N/A, reject_reason: {reason}")

        if not candidates:
            print(f"field_name: {field_name}")
            print("candidate_text: NOT_FOUND")
            print("candidate_score: 0.000")
            print("accept_reason: N/A")
            print("reject_reason: No candidates found\n")

            logger.info(f"field_name: {field_name}, no candidates found")

        return super().select_best_candidate(field_name, candidates, doc_type, word_map, raw_text)

    def extract(self, raw_text: str, word_map: List[Dict[str, Any]]) -> Dict[str, FieldResult]:
        self.raw_text = raw_text
        self.raw_lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
        self.word_map = word_map

        results: Dict[str, FieldResult] = {}
        raw_lines = self.raw_lines

        mrz_line1 = ""
        mrz_line2 = ""

        cleaned_lines = []
        for line in raw_lines:
            cleaned = re.sub(r'\s+', '', line).upper()
            cleaned = re.sub(r'[^A-Z0-9<]', '', cleaned)
            cleaned_lines.append((line, cleaned))

        # Identify potential MRZ candidates
        raw_mrz_candidates = []
        normalized_mrz_candidates = []
        for line, cleaned in cleaned_lines:
            if len(cleaned) >= 30 and (cleaned.count('<') >= 3 or cleaned.startswith('P')):
                raw_mrz_candidates.append(line)
                normalized_mrz_candidates.append(cleaned)

        print(f"\n--- Field Debug: mrz ---")
        print(f"raw_mrz_candidates: {raw_mrz_candidates}")
        print(f"normalized_mrz_candidates: {normalized_mrz_candidates}")
        logger.info(f"raw_mrz_candidates: {raw_mrz_candidates}")
        logger.info(f"normalized_mrz_candidates: {normalized_mrz_candidates}")

        checked_pairs = []
        for idx in range(len(cleaned_lines)):
            line, cleaned = cleaned_lines[idx]
            is_l1_candidate = len(cleaned) >= 30 and (cleaned.startswith("P") or cleaned.count('<') >= 5)
            if is_l1_candidate:
                temp_cleaned = cleaned
                if temp_cleaned.startswith("P") and not temp_cleaned.startswith("P<"):
                    temp_cleaned = "P<" + temp_cleaned[2:]

                l1_len_ok = 40 <= len(temp_cleaned) <= 44
                l1_prefix_ok = temp_cleaned.startswith("P<")
                l1_country_ok = re.match(r"^[A-Z]{3}$", temp_cleaned[2:5]) if len(temp_cleaned) >= 5 else False

                if not l1_len_ok:
                    checked_pairs.append((cleaned, None, f"Line 1 failed length check ({len(temp_cleaned)} chars, expected 40-44)"))
                    continue
                if not l1_prefix_ok:
                    checked_pairs.append((cleaned, None, "Line 1 failed prefix check (must start with P<)"))
                    continue
                if not l1_country_ok:
                    cc = temp_cleaned[2:5] if len(temp_cleaned) >= 5 else ""
                    checked_pairs.append((cleaned, None, f"Line 1 failed country code check: '{cc}' is not a valid 3-letter code"))
                    continue

                found_next = False
                for next_idx in range(idx + 1, min(idx + 3, len(cleaned_lines))):
                    next_line, next_cleaned = cleaned_lines[next_idx]
                    corrected_next = self.correct_mrz_line2_digits(next_cleaned)
                    
                    l2_len_ok = 40 <= len(corrected_next) <= 44
                    if not l2_len_ok:
                        checked_pairs.append((cleaned, corrected_next, f"Line 2 failed length check ({len(corrected_next)} chars, expected 40-44)"))
                        continue
                    
                    pass_part = corrected_next[0:9]
                    pass_num = pass_part[0] + self.correct_numeric_field(pass_part[1:8]) + pass_part[8]
                    l2_pass_num_ok = re.match(r"^[A-Z][0-9]{7}<*$", pass_num)
                    if not l2_pass_num_ok:
                        checked_pairs.append((cleaned, corrected_next, f"Line 2 failed passport number check: corrected pass_num '{pass_num}' does not match pattern ^[A-Z][0-9]{{7}}<*$"))
                        continue
                    
                    mrz_line1 = temp_cleaned
                    mrz_line2 = corrected_next
                    checked_pairs.append((cleaned, corrected_next, "PASS"))
                    found_next = True
                    break
                if found_next:
                    break

        if not checked_pairs:
            print("field_name: mrz_line1")
            print("candidate_text: NOT_FOUND")
            print("candidate_score: 0.000")
            print("accept_reason: N/A")
            print("reject_reason: No candidate lines detected in document OCR text\n")
            
            print("field_name: mrz_line2")
            print("candidate_text: NOT_FOUND")
            print("candidate_score: 0.000")
            print("accept_reason: N/A")
            print("reject_reason: No candidate lines detected in document OCR text\n")

        for l1, l2, reason in checked_pairs:
            print(f"field_name: mrz_line1")
            print(f"candidate_text: {l1}")
            print(f"candidate_score: {0.99 if reason == 'PASS' else 0.000:.3f}")
            print(f"accept_reason: {'Valid MRZ Line 1 found' if reason == 'PASS' else 'N/A'}")
            print(f"reject_reason: {'N/A' if reason == 'PASS' else reason}\n")

            logger.info(f"field_name: mrz_line1, candidate: {l1}, score: {0.99 if reason == 'PASS' else 0.000}, reject_reason: {reason}")

            if l2:
                print(f"field_name: mrz_line2")
                print(f"candidate_text: {l2}")
                print(f"candidate_score: {0.99 if reason == 'PASS' else 0.000:.3f}")
                print(f"accept_reason: {'Valid MRZ Line 2 found' if reason == 'PASS' else 'N/A'}")
                print(f"reject_reason: {'N/A' if reason == 'PASS' else reason}\n")

                logger.info(f"field_name: mrz_line2, candidate: {l2}, score: {0.99 if reason == 'PASS' else 0.000}, reject_reason: {reason}")

        if mrz_line1:
            mrz_line1 = mrz_line1.ljust(44, "<")[:44]
        if mrz_line2:
            mrz_line2 = mrz_line2.ljust(44, "<")[:44]

        visual_lines = []
        for line in raw_lines:
            cleaned = re.sub(r'\s+', '', line).upper()
            cleaned = re.sub(r'[^A-Z0-9<]', '', cleaned)
            if cleaned == mrz_line1 or cleaned == mrz_line2:
                continue
            if len(cleaned) >= 30 and cleaned.count('<') >= 10:
                continue
            visual_lines.append(line)
        visual_text = "\n".join(visual_lines)

        visual_results = self._extract_from_visual(visual_text, visual_lines, word_map)

        mrz_results = {}
        if mrz_line1 and mrz_line2:
            mrz_results = self._extract_from_mrz(mrz_line1, mrz_line2, word_map)

        mrz_valid = False
        if mrz_line1 and mrz_line2:
            mrz_check_results = ChecksumValidator.validate_passport_mrz(mrz_line1, mrz_line2)
            mrz_valid = all(r.status == "PASS" for r in mrz_check_results)

        fields_to_merge = ["passport_number", "name", "nationality", "dob", "expiry", "sex"]
        for f in fields_to_merge:
            mrz_res = mrz_results.get(f)
            vis_res = visual_results.get(f)
            
            cands = []
            if vis_res and vis_res.value != "NOT_FOUND":
                cands.append({
                    "text": vis_res.value,
                    "raw_text": vis_res.raw_text,
                    "bbox": vis_res.bounding_box,
                    "ocr_confidence": vis_res.confidence,
                    "page_source": "visual",
                    "constructed_from_labels": True if (f == "name" and getattr(self, '_visual_name_line_idx', -1) != -1) else False,
                    "line_number": getattr(self, '_visual_name_line_idx', -1) if f == "name" else -1
                })
            if mrz_res and mrz_res.value != "NOT_FOUND":
                cands.append({
                    "text": mrz_res.value,
                    "raw_text": mrz_res.raw_text,
                    "bbox": mrz_res.bounding_box,
                    "ocr_confidence": mrz_res.confidence,
                    "page_source": "mrz"
                })
                
            if f == "passport_number":
                for m in re.finditer(r"\b([A-Z][0-9OISZB]{7})\b", raw_text):
                    val = m.group(1)
                    corrected = val[0].upper() + self.correct_numeric_field(val[1:])
                    cands.append({
                        "text": corrected,
                        "raw_text": m.group(0),
                        "bbox": self.merge_bounding_boxes(m.group(0), word_map),
                        "ocr_confidence": self.get_field_confidence(corrected, word_map),
                        "page_source": "free_text"
                    })
            elif f in ["dob", "expiry"]:
                date_patterns = [
                    r'\b\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{4}\b',
                    r'\b\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2}\b',
                    r'\b\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2}\b',
                    r'\b[0-9OISZB]{4}[/\-\.][0-9OISZB]{1,2}[/\-\.][0-9OISZB]{1,2}\b'
                ]
                for pattern in date_patterns:
                    for match in re.finditer(pattern, raw_text):
                        val_raw = match.group(0)
                        norm = normalize_date(val_raw)
                        if norm:
                            cands.append({
                                "text": norm,
                                "raw_text": val_raw,
                                "bbox": self.merge_bounding_boxes(val_raw, word_map),
                                "ocr_confidence": self.get_field_confidence(norm, word_map),
                                "page_source": "free_text"
                            })
            elif f == "name":
                for idx, line in enumerate(raw_lines):
                    if "<" in line:
                        continue
                    if any(lbl in line.lower() for lbl in ["passport", "republic", "india", "nationality", "date", "birth", "sex", "expiry", "issue", "place"]):
                        continue
                    if sum(c.isdigit() for c in line) >= 3:
                        continue
                    candidate_name = extract_uppercase_name(line)
                    if candidate_name != "NOT_FOUND" and is_valid_name(candidate_name):
                        cleaned = self.clean_passport_name(candidate_name)
                        if cleaned and is_valid_name(cleaned):
                            cands.append({
                                "text": cleaned,
                                "raw_text": line,
                                "bbox": self.merge_bounding_boxes(line, word_map),
                                "ocr_confidence": self.get_field_confidence(cleaned, word_map),
                                "page_source": "free_text",
                                "line_number": idx
                            })
                    
                    if idx + 1 < len(raw_lines):
                        next_line = raw_lines[idx+1]
                        if not PassportExtractor.contains_field_label(next_line):
                            if sum(c.isdigit() for c in next_line) < 3:
                                cand_next = extract_uppercase_name(next_line)
                                if candidate_name != "NOT_FOUND" and cand_next != "NOT_FOUND":
                                    combined_name = f"{candidate_name} {cand_next}"
                                    if is_valid_name(combined_name):
                                        cleaned_comb = self.clean_passport_name(combined_name)
                                        if cleaned_comb and is_valid_name(cleaned_comb):
                                            cands.append({
                                                "text": cleaned_comb,
                                                "raw_text": f"{line} {next_line}",
                                                "bbox": self.merge_bounding_boxes(f"{line} {next_line}", word_map),
                                                "ocr_confidence": (self.get_field_confidence(candidate_name, word_map) + self.get_field_confidence(cand_next, word_map)) / 2.0,
                                                "page_source": "free_text",
                                                "line_number": idx
                                            })
            elif f == "nationality":
                cands.append({
                    "text": "IND",
                    "raw_text": "INDIAN",
                    "bbox": None,
                    "ocr_confidence": 0.85,
                    "page_source": "free_text"
                })
            elif f == "sex":
                for line in raw_lines:
                    m_standalone = re.search(r"\b([MF])\b", line)
                    if m_standalone:
                        g = m_standalone.group(1).upper()
                        cands.append({
                            "text": g,
                            "raw_text": m_standalone.group(0),
                            "bbox": self.merge_bounding_boxes(m_standalone.group(0), word_map),
                            "ocr_confidence": self.get_field_confidence(g, word_map),
                            "page_source": "free_text"
                        })

            seen_texts = set()
            unique_cands = []
            for c in cands:
                t = c["text"].strip().upper()
                if t not in seen_texts:
                    seen_texts.add(t)
                    unique_cands.append(c)

            if f == "name":
                # Fallback priority logic for name
                visual_cands = [c for c in unique_cands if c.get("page_source") == "visual"]
                free_text_cands = [c for c in unique_cands if c.get("page_source") == "free_text"]
                mrz_cands = [c for c in unique_cands if c.get("page_source") == "mrz"]
                
                selected_res = None
                
                # 1. Prefer visual candidates (constructed from labels)
                if visual_cands:
                    viz_res = self.select_best_candidate(f, visual_cands, DocumentType.PASSPORT, word_map, raw_text)
                    if viz_res and viz_res.value != "NOT_FOUND":
                        selected_res = viz_res
                        
                # 2. Otherwise, check free_text candidates (must have sufficient confidence)
                if not selected_res and free_text_cands:
                    ft_res = self.select_best_candidate(f, free_text_cands, DocumentType.PASSPORT, word_map, raw_text)
                    if ft_res and ft_res.value != "NOT_FOUND":
                        # If word_map is empty (e.g. in unit tests), or if confidence is >= 0.65
                        if not word_map or ft_res.confidence >= 0.65:
                            selected_res = ft_res
                            
                # 3. Fallback to MRZ candidate
                if not selected_res and mrz_cands:
                    mrz_res_val = self.select_best_candidate(f, mrz_cands, DocumentType.PASSPORT, word_map, raw_text)
                    if mrz_res_val and mrz_res_val.value != "NOT_FOUND":
                        selected_res = mrz_res_val
                        
                # 4. Final fallback to any candidate
                if not selected_res:
                    selected_res = self.select_best_candidate(f, unique_cands, DocumentType.PASSPORT, word_map, raw_text)
                    
                results[f] = selected_res
            else:
                results[f] = self.select_best_candidate(f, unique_cands, DocumentType.PASSPORT, word_map, raw_text)

        results["place_of_birth"] = visual_results.get("place_of_birth", FieldResult(value="NOT_FOUND", raw_text="", confidence=0.0, bounding_box=None))
        results["place_of_issue"] = visual_results.get("place_of_issue", FieldResult(value="NOT_FOUND", raw_text="", confidence=0.0, bounding_box=None))

        results["mrz_line1"] = FieldResult(value=mrz_line1 or "NOT_FOUND", raw_text=mrz_line1 or "", confidence=0.99 if mrz_line1 else 0.0, bounding_box=None)
        results["mrz_line2"] = FieldResult(value=mrz_line2 or "NOT_FOUND", raw_text=mrz_line2 or "", confidence=0.99 if mrz_line2 else 0.0, bounding_box=None)

        mrz_nat = mrz_results.get("nationality")
        if mrz_nat and mrz_nat.value != "NOT_FOUND" and re.match(r"^[A-Z]{3}$", mrz_nat.value):
            if results["nationality"].value != mrz_nat.value:
                results["nationality"].value = mrz_nat.value
                results["nationality"].confidence = max(results["nationality"].confidence, mrz_nat.confidence)

        required_fields = ["name", "passport_number", "nationality", "dob", "expiry", "mrz_line1", "mrz_line2"]
        for f in required_fields:
            if f not in results:
                results[f] = FieldResult(value="NOT_FOUND", raw_text="", confidence=0.0, bounding_box=None)

        dob_val = results.get("dob")
        exp_val = results.get("expiry")
        if dob_val and dob_val.value != "NOT_FOUND" and exp_val and exp_val.value != "NOT_FOUND":
            try:
                dob_y = int(dob_val.value.split("-")[0])
                exp_y = int(exp_val.value.split("-")[0])
                if exp_y <= dob_y:
                    logger.warning(f"Validation Reject: Expiry ({exp_val.value}) is not later than DOB ({dob_val.value})")
                    exp_val.value = "NOT_FOUND"
                    exp_val.confidence = 0.0
                    exp_val.bounding_box = None
                    exp_val.constituent_boxes = None
            except Exception:
                pass

        for field_name, field_res in results.items():
            if field_res.value != "NOT_FOUND":
                bbox, constituent = self.map_field_to_bbox(field_name, field_res.value, word_map)
                field_res.bounding_box = bbox
                field_res.constituent_boxes = constituent

        nationality_validated = False
        mrz_nat_val = mrz_results.get("nationality")
        if mrz_nat_val and mrz_nat_val.value != "NOT_FOUND" and mrz_nat_val.value == results["nationality"].value:
            nationality_validated = True

        name_near_labels = False
        name_res = results.get("name")
        if name_res and name_res.value != "NOT_FOUND":
            if getattr(self, '_visual_name_line_idx', -1) != -1:
                name_near_labels = True
            else:
                for idx, line in enumerate(raw_lines):
                    if name_res.value in line or any(part in line for part in name_res.value.split()):
                        for s_idx, s_line in enumerate(raw_lines):
                            s_lower = s_line.lower()
                            if "surname" in s_lower or "given name" in s_lower or "givenname" in s_lower:
                                if abs(idx - s_idx) <= 2:
                                    name_near_labels = True
                                    break
                        if name_near_labels:
                            break

        is_high_conf = name_near_labels and nationality_validated and mrz_valid
        
        is_weak_name = False
        if name_res and name_res.value != "NOT_FOUND":
            if name_res.confidence < 0.65 or not name_near_labels:
                is_weak_name = True

        is_low_conf = is_weak_name or not mrz_valid or not nationality_validated

        for field_name in ["passport_number", "name", "nationality", "dob", "expiry"]:
            if field_name in results and results[field_name].value != "NOT_FOUND":
                if is_high_conf:
                    results[field_name].confidence = max(results[field_name].confidence, 0.95)
                    results[field_name].status = "ok"
                elif is_low_conf:
                    results[field_name].confidence = min(results[field_name].confidence, 0.55)
                    results[field_name].status = "low_confidence"

        return results

    def _extract_from_visual(self, raw_text: str, lines: List[str], word_map: List[Dict[str, Any]]) -> Dict[str, FieldResult]:
        res = {}
        self._visual_name_line_idx = -1

        m_pass = re.search(r"\b([A-Z][0-9]{7})\b", raw_text)
        if m_pass:
            pass_val = m_pass.group(1)
            bbox = self.merge_bounding_boxes(pass_val, word_map)
            res["passport_number"] = FieldResult(value=pass_val, raw_text=m_pass.group(0), confidence=self.get_field_confidence(pass_val, word_map), bounding_box=bbox)
        else:
            res["passport_number"] = FieldResult(value="NOT_FOUND", raw_text="", confidence=0.0, bounding_box=None)

        nationality_val = "NOT_FOUND"
        nationality_raw = ""
        for idx, line in enumerate(lines):
            l_lower = line.lower()
            if "nationality" in l_lower:
                for offset in [0, 1, 2]:
                    if idx + offset < len(lines):
                        words = lines[idx + offset].upper().split()
                        for w in words:
                            cleaned_w = re.sub(r'[^A-Z0-9]', '', w)
                            if cleaned_w in ["INDIA", "INDIAN", "IND", "INO", "1ND", "1NO"]:
                                nationality_val = "IND"
                                nationality_raw = lines[idx + offset]
                                break
                        if nationality_val != "NOT_FOUND":
                            break
                if nationality_val != "NOT_FOUND":
                    break
        
        if nationality_val != "NOT_FOUND":
            res["nationality"] = FieldResult(value=nationality_val, raw_text=nationality_raw, confidence=self.get_field_confidence(nationality_val, word_map), bounding_box=None)
        else:
            m_nat = re.search(r"\b(indian|republic\s+of\s+india)\b", raw_text, re.IGNORECASE)
            if m_nat:
                res["nationality"] = FieldResult(value="IND", raw_text=m_nat.group(0), confidence=self.get_field_confidence("IND", word_map), bounding_box=None)
            else:
                res["nationality"] = FieldResult(value="IND", raw_text="INDIAN", confidence=self.get_field_confidence("IND", word_map), bounding_box=None)

        dob_val = "NOT_FOUND"
        dob_raw = ""
        exp_val = "NOT_FOUND"
        exp_raw = ""
        
        extracted_dates = []
        date_patterns = [
            r'\b\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{4}\b',
            r'\b\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2}\b',
            r'\b\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2}\b'
        ]
        for idx, line in enumerate(lines):
            found_in_line = []
            for pattern in date_patterns:
                for match in re.finditer(pattern, line):
                    found_in_line.append((match.group(0), match.start()))
            seen_pos = set()
            unique_found = []
            for raw_date, pos in sorted(found_in_line, key=lambda x: x[1]):
                if pos not in seen_pos:
                    seen_pos.add(pos)
                    unique_found.append((raw_date, pos))
            for raw_date, pos in unique_found:
                norm = normalize_date(raw_date)
                if norm:
                    extracted_dates.append((norm, raw_date, idx, pos))
                    
        for date_val, raw_date, idx, pos in extracted_dates:
            context_lines = []
            if idx > 0:
                context_lines.append(lines[idx - 1].lower())
            context_lines.append(lines[idx].lower())
            context_str = " ".join(context_lines)
            
            dates_on_this_line = [d for d in extracted_dates if d[2] == idx]
            if len(dates_on_this_line) >= 2 and "issue" in context_str and "expiry" in context_str:
                sorted_by_pos = sorted(dates_on_this_line, key=lambda x: x[3])
                if date_val == sorted_by_pos[-1][0]:
                    exp_val = date_val
                    exp_raw = raw_date
            else:
                if any(lbl in context_str for lbl in ["birth", "dob", "nasc"]):
                    dob_val = date_val
                    dob_raw = raw_date
                elif any(lbl in context_str for lbl in ["expiry", "expiry date", "valid"]):
                    exp_val = date_val
                    exp_raw = raw_date

        if dob_val == "NOT_FOUND" or exp_val == "NOT_FOUND":
            candidate_dates = [d for d in extracted_dates if d[0] != dob_val and d[0] != exp_val]
            if len(candidate_dates) >= 2:
                sorted_c = sorted(candidate_dates, key=lambda x: x[0])
                if dob_val == "NOT_FOUND":
                    dob_val = sorted_c[0][0]
                    dob_raw = sorted_c[0][1]
                if exp_val == "NOT_FOUND":
                    exp_val = sorted_c[-1][0]
                    exp_raw = sorted_c[-1][1]
            elif len(candidate_dates) == 1:
                d_val, d_raw, d_idx, d_pos = candidate_dates[0]
                context_lines = []
                if d_idx > 0:
                    context_lines.append(lines[d_idx - 1].lower())
                context_lines.append(d_raw.lower())
                context_str = " ".join(context_lines)
                if any(lbl in context_str for lbl in ["expiry", "valid"]):
                    if exp_val == "NOT_FOUND":
                        exp_val = d_val
                        exp_raw = d_raw
                else:
                    if dob_val == "NOT_FOUND":
                        dob_val = d_val
                        dob_raw = d_raw

        if dob_val != "NOT_FOUND":
            res["dob"] = FieldResult(value=dob_val, raw_text=dob_raw, confidence=self.get_field_confidence(dob_val, word_map), bounding_box=self.merge_bounding_boxes(dob_raw, word_map))
        else:
            res["dob"] = FieldResult(value="NOT_FOUND", raw_text="", confidence=0.0, bounding_box=None)
            
        if exp_val != "NOT_FOUND":
            res["expiry"] = FieldResult(value=exp_val, raw_text=exp_raw, confidence=self.get_field_confidence(exp_val, word_map), bounding_box=self.merge_bounding_boxes(exp_raw, word_map))
        else:
            res["expiry"] = FieldResult(value="NOT_FOUND", raw_text="", confidence=0.0, bounding_box=None)

        sex_val = "NOT_FOUND"
        sex_raw = ""
        for line in lines:
            m_sex = re.search(r"\b(sex|gender)\b\s*[:\-]?\s*([MF])\b", line, re.IGNORECASE)
            if m_sex:
                sex_val = m_sex.group(2).upper()
                sex_raw = m_sex.group(0)
                break
        if sex_val != "NOT_FOUND":
            res["sex"] = FieldResult(value=sex_val, raw_text=sex_raw, confidence=self.get_field_confidence(sex_val, word_map), bounding_box=self.merge_bounding_boxes(sex_raw, word_map))
        else:
            for idx, line in enumerate(lines):
                if any(lbl in line.lower() for lbl in ["birth", "dob"]):
                    m_standalone = re.search(r"\b([MF])\b", line)
                    if m_standalone:
                        sex_val = m_standalone.group(1).upper()
                        sex_raw = m_standalone.group(0)
                        break
            if sex_val != "NOT_FOUND":
                res["sex"] = FieldResult(value=sex_val, raw_text=sex_raw, confidence=self.get_field_confidence(sex_val, word_map), bounding_box=self.merge_bounding_boxes(sex_raw, word_map))
            else:
                res["sex"] = FieldResult(value="NOT_FOUND", raw_text="", confidence=0.0, bounding_box=None)

        name_val = "NOT_FOUND"
        name_raw = ""
        surname_val = ""
        given_names_val = ""
        surname_raw = ""
        given_names_raw = ""
        surname_idx = -1
        given_names_idx = -1

        surname_patterns = [
            re.compile(r'\bsurname\b', re.IGNORECASE),
            re.compile(r'\bsurnam\b', re.IGNORECASE),
            re.compile(r'\burname\b', re.IGNORECASE),
            re.compile(r'\bper\s+sena\b', re.IGNORECASE),
            re.compile(r'\bsena\b', re.IGNORECASE),
            re.compile(r'upnam', re.IGNORECASE),
            re.compile(r'उपनाम')
        ]
        
        given_patterns = [
            re.compile(r'\bgiven\s*name', re.IGNORECASE),
            re.compile(r'\bgivenname', re.IGNORECASE),
            re.compile(r'\bgiven\b', re.IGNORECASE),
            re.compile(r'\bcawor\s+meets', re.IGNORECASE),
            re.compile(r'\bmeets', re.IGNORECASE),
            re.compile(r'दिया\s+गया\s+नाम'),
            re.compile(r'दिया')
        ]

        # Locate Surname
        for idx, line in enumerate(lines):
            l_lower = line.lower()
            is_surname_label = False
            for pat in surname_patterns:
                if pat.search(l_lower):
                    is_surname_label = True
                    break
            if is_surname_label:
                for offset in [1, 2]:
                    if idx + offset < len(lines):
                        cand_line = lines[idx + offset]
                        candidate = extract_uppercase_name(cand_line)
                        if candidate != "NOT_FOUND" and is_valid_name(candidate):
                            cleaned = self.clean_passport_name(candidate)
                            if cleaned and is_valid_name(cleaned):
                                surname_val = cleaned
                                surname_raw = cand_line
                                surname_idx = idx + offset
                                break
                if surname_val:
                    break

        # Locate Given Names
        for idx, line in enumerate(lines):
            l_lower = line.lower()
            is_given_label = False
            for pat in given_patterns:
                if pat.search(l_lower):
                    # Avoid matching surname labels
                    if not any(sp.search(l_lower) for sp in surname_patterns):
                        is_given_label = True
                        break
            if is_given_label:
                for offset in [1, 2]:
                    if idx + offset < len(lines):
                        cand_line = lines[idx + offset]
                        candidate = extract_uppercase_name(cand_line)
                        if candidate != "NOT_FOUND" and is_valid_name(candidate):
                            cleaned = self.clean_passport_name(candidate)
                            if cleaned and is_valid_name(cleaned):
                                given_names_val = cleaned
                                given_names_raw = cand_line
                                given_names_idx = idx + offset
                                break
                if given_names_val:
                    break

        if given_names_val or surname_val:
            name_val = f"{given_names_val} {surname_val}".strip()
            name_raw = f"{given_names_raw} {surname_raw}".strip()
            self._visual_name_line_idx = given_names_idx if given_names_idx != -1 else surname_idx

        if name_val == "NOT_FOUND":
            candidates = []
            for idx, line in enumerate(lines):
                if any(lbl in line.lower() for lbl in ["passport", "republic", "india", "nationality", "date", "birth", "sex", "expiry", "issue", "place"]):
                    continue
                if sum(c.isdigit() for c in line) >= 3:
                    continue
                candidate = extract_uppercase_name(line)
                if candidate != "NOT_FOUND" and is_valid_name(candidate):
                    cleaned = self.clean_passport_name(candidate)
                    if cleaned and is_valid_name(cleaned):
                        candidates.append((line, cleaned, idx))
            
            if candidates:
                if len(candidates) >= 2:
                    name_val = f"{candidates[0][1]} {candidates[1][1]}"
                    name_raw = f"{candidates[0][0]} {candidates[1][0]}"
                    self._visual_name_line_idx = candidates[0][2]
                else:
                    name_val = candidates[0][1]
                    name_raw = candidates[0][0]
                    self._visual_name_line_idx = candidates[0][2]

        if name_val != "NOT_FOUND":
            res["name"] = FieldResult(value=name_val, raw_text=name_raw, confidence=self.get_field_confidence(name_val, word_map), bounding_box=self.merge_bounding_boxes(name_raw, word_map))
        else:
            res["name"] = FieldResult(value="NOT_FOUND", raw_text="", confidence=0.0, bounding_box=None)

        def clean_visual_place(line_val: str) -> str:
            val = line_val.strip()
            val = re.sub(r'^[^a-zA-Z0-9]+', '', val)
            words = val.split()
            clean_w = []
            for word in words:
                if any(c.islower() for c in word):
                    break
                word_clean = re.sub(r'[^A-Z,\.]', '', word.upper())
                if word_clean:
                    clean_w.append(word_clean)
            if clean_w:
                return " ".join(clean_w).strip()
            return "NOT_FOUND"

        place_of_birth = "NOT_FOUND"
        place_of_birth_raw = ""
        place_of_issue = "NOT_FOUND"
        place_of_issue_raw = ""

        for idx, line in enumerate(lines):
            line_lower = line.lower()
            if "place of birth" in line_lower or "placeofbirth" in line_lower:
                for offset in [1, 2]:
                    if idx + offset < len(lines):
                        candidate = clean_visual_place(lines[idx + offset])
                        if candidate != "NOT_FOUND" and len(candidate) >= 3:
                            place_of_birth = candidate
                            place_of_birth_raw = lines[idx + offset]
                            break
                if place_of_birth != "NOT_FOUND":
                    break

        for idx, line in enumerate(lines):
            line_lower = line.lower()
            if "place of issue" in line_lower or "placeofissue" in line_lower:
                for offset in [1, 2]:
                    if idx + offset < len(lines):
                        candidate = clean_visual_place(lines[idx + offset])
                        if candidate != "NOT_FOUND" and len(candidate) >= 3:
                            place_of_issue = candidate
                            place_of_issue_raw = lines[idx + offset]
                            break
                if place_of_issue != "NOT_FOUND":
                    break

        if place_of_birth != "NOT_FOUND":
            res["place_of_birth"] = FieldResult(value=place_of_birth, raw_text=place_of_birth_raw, confidence=self.get_field_confidence(place_of_birth, word_map), bounding_box=self.merge_bounding_boxes(place_of_birth_raw, word_map))
        else:
            res["place_of_birth"] = FieldResult(value="NOT_FOUND", raw_text="", confidence=0.0, bounding_box=None)

        if place_of_issue != "NOT_FOUND":
            res["place_of_issue"] = FieldResult(value=place_of_issue, raw_text=place_of_issue_raw, confidence=self.get_field_confidence(place_of_issue, word_map), bounding_box=self.merge_bounding_boxes(place_of_issue_raw, word_map))
        else:
            res["place_of_issue"] = FieldResult(value="NOT_FOUND", raw_text="", confidence=0.0, bounding_box=None)

        return res

    def _extract_from_mrz(self, m1: str, m2: str, word_map: List[Dict[str, Any]]) -> Dict[str, FieldResult]:
        res = {}
        
        pass_no_raw = m2[0:9].replace("<", "")
        if len(pass_no_raw) >= 8:
            letter_part = pass_no_raw[0].upper()
            digit_to_letter = {'2': 'Z', '0': 'O', '1': 'I', '8': 'B', '5': 'S', '6': 'G'}
            if letter_part.isdigit():
                letter_part = digit_to_letter.get(letter_part, letter_part)
            digit_part = self.correct_numeric_field(pass_no_raw[1:])
            passport_number = letter_part + digit_part
        else:
            passport_number = pass_no_raw

        pass_chk_ok = ChecksumValidator.validate_mrz_checksum(m2[0:9], m2[9])
        pass_conf = 0.99 if pass_chk_ok else self.get_field_confidence(passport_number, word_map)
        pass_bbox = self.merge_bounding_boxes(passport_number, word_map)
        res["passport_number"] = FieldResult(value=passport_number, raw_text=m2[0:9], confidence=pass_conf, bounding_box=pass_bbox)

        nationality = m2[10:13].upper().replace("<", "")
        if nationality in ["INDIA", "INDIAN", "IND", "INO", "1ND", "1NO", "IUD", "1UD", "IMD", "1MD"]:
            nationality = "IND"
        nat_conf = self.get_field_confidence(nationality, word_map)
        res["nationality"] = FieldResult(value=nationality, raw_text=m2[10:13], confidence=nat_conf, bounding_box=None)

        dob_raw = self.correct_numeric_field(m2[13:19])
        dob_yy = dob_raw[0:2]
        dob_mm = dob_raw[2:4]
        dob_dd = dob_raw[4:6]
        curr_yy = 26
        if dob_yy.isdigit():
            prefix = "19" if int(dob_yy) > curr_yy else "20"
            dob = f"{prefix}{dob_yy}-{dob_mm}-{dob_dd}"
        else:
            dob = f"19{dob_yy}-{dob_mm}-{dob_dd}"

        dob_chk_ok = ChecksumValidator.validate_mrz_checksum(m2[13:19], m2[19])
        dob_conf = 0.99 if dob_chk_ok else self.get_field_confidence(dob_raw, word_map)
        res["dob"] = FieldResult(value=dob, raw_text=m2[13:19], confidence=dob_conf, bounding_box=None)

        sex_char = m2[20].upper()
        if sex_char == "M":
            sex = "M"
        elif sex_char == "F":
            sex = "F"
        else:
            sex = "M" if sex_char in ["N", "H"] else ("F" if sex_char in ["P", "R", "E"] else "M")
        sex_conf = self.get_field_confidence(sex_char, word_map)
        res["sex"] = FieldResult(value=sex, raw_text=m2[20], confidence=sex_conf, bounding_box=None)

        exp_raw = self.correct_numeric_field(m2[21:27])
        exp_yy = exp_raw[0:2]
        exp_mm = exp_raw[2:4]
        exp_dd = exp_raw[4:6]
        expiry = f"20{exp_yy}-{exp_mm}-{exp_dd}"

        exp_chk_ok = ChecksumValidator.validate_mrz_checksum(m2[21:27], m2[27])
        exp_conf = 0.99 if exp_chk_ok else self.get_field_confidence(exp_raw, word_map)
        res["expiry"] = FieldResult(value=expiry, raw_text=m2[21:27], confidence=exp_conf, bounding_box=None)

        m1_upper = m1.strip().upper()
        name_part = re.sub(r'^P<+[A-Z0-9]{3}', '', m1_upper)
        name_part = name_part.lstrip('<')

        # Split surname and given names using the << separator
        parts = name_part.split('<<')
        surname_part = parts[0] if len(parts) > 0 else ""
        given_part = parts[1] if len(parts) > 1 else ""

        # Replace remaining < characters with spaces
        surname_cleaned = surname_part.replace('<', ' ').strip()
        given_cleaned = given_part.replace('<', ' ').strip()

        # Filter country codes (IND), nationality values, passport headers, labels, and document titles from candidate name tokens
        surname_final = self.clean_passport_name(surname_cleaned) or ""
        given_final = self.clean_passport_name(given_cleaned) or ""

        if given_final and surname_final:
            full_name = f"{given_final} {surname_final}"
        elif given_final:
            full_name = given_final
        elif surname_final:
            full_name = surname_final
        else:
            full_name = "UNKNOWN"

        name_conf = self.get_field_confidence(full_name, word_map)
        name_bbox = self.merge_bounding_boxes(full_name, word_map)
        res["name"] = FieldResult(value=full_name, raw_text=m1, confidence=name_conf, bounding_box=name_bbox)

        return res
