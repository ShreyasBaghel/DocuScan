import re
from typing import Dict, List, Any
from extractors.base_extractor import BaseExtractor
from utils.document_packet import FieldResult, DocumentType
from utils.string_utils import normalize_date, clean_whitespace, extract_uppercase_name, is_valid_name, ocr_correct_digits
from validators.checksum_validator import ChecksumValidator

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

    def extract(self, raw_text: str, word_map: List[Dict[str, Any]]) -> Dict[str, FieldResult]:
        results: Dict[str, FieldResult] = {}

        # 1. Split raw text into lines
        raw_lines = [line.strip() for line in raw_text.split('\n') if line.strip()]

        # 2. Identify MRZ Lines (using sequential top-down and bottom-up search)
        mrz_line1 = ""
        mrz_line2 = ""

        # Search for a line starting with P< or containing P<< (after cleaning)
        for idx, line in enumerate(raw_lines):
            cleaned = re.sub(r'\s+', '', line).upper()
            cleaned = re.sub(r'[^A-Z0-9<]', '', cleaned)
            # Line 1 usually starts with P< or P<< or is a P-line of length >= 30
            if len(cleaned) >= 30 and (cleaned.startswith("P<") or (cleaned.startswith("P") and cleaned.count("<") >= 5)):
                mrz_line1 = cleaned
                # The next line is typically MRZ Line 2
                if idx + 1 < len(raw_lines):
                    next_cleaned = re.sub(r'\s+', '', raw_lines[idx + 1]).upper()
                    next_cleaned = re.sub(r'[^A-Z0-9<]', '', next_cleaned)
                    if len(next_cleaned) >= 30:
                        mrz_line2 = next_cleaned
                break

        # Fallback: if we didn't find MRZ lines that way, try bottom-up matching
        if not mrz_line1 or not mrz_line2:
            mrz_candidates = []
            for idx, line in enumerate(raw_lines):
                cleaned = re.sub(r'\s+', '', line).upper()
                cleaned = re.sub(r'[^A-Z0-9<]', '', cleaned)
                if len(cleaned) >= 30 and cleaned.count('<') >= 4:
                    mrz_candidates.append((idx, cleaned))
            if len(mrz_candidates) >= 2:
                selected = sorted(mrz_candidates[-2:], key=lambda x: x[0])
                mrz_line1 = selected[0][1]
                mrz_line2 = selected[1][1]

        # Standardize MRZ length to 44
        if mrz_line1:
            if len(mrz_line1) < 44:
                mrz_line1 = mrz_line1.ljust(44, "<")
            elif len(mrz_line1) > 44:
                mrz_line1 = mrz_line1[:44]
        if mrz_line2:
            if len(mrz_line2) < 44:
                mrz_line2 = mrz_line2.ljust(44, "<")
            elif len(mrz_line2) > 44:
                mrz_line2 = mrz_line2[:44]

        # 3. Separate MRZ text from normal visual text
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

        # 4. Extract from Visual Zone
        visual_results = self._extract_from_visual(visual_text, visual_lines, word_map)

        # 5. Extract from MRZ Zone if available
        mrz_results = {}
        if mrz_line1 and mrz_line2:
            mrz_results = self._extract_from_mrz(mrz_line1, mrz_line2, word_map)

        # Validate MRZ check digits before accepting
        mrz_valid = False
        if mrz_line1 and mrz_line2:
            mrz_check_results = ChecksumValidator.validate_passport_mrz(mrz_line1, mrz_line2)
            mrz_valid = all(r.status == "PASS" for r in mrz_check_results)

        # 6. Merge/Select best results prioritizing MRZ (falling back to visual)
        fields_to_merge = ["passport_number", "name", "nationality", "dob", "expiry", "sex"]
        for f in fields_to_merge:
            mrz_res = mrz_results.get(f)
            vis_res = visual_results.get(f)
            
            cands = []
            if mrz_res and mrz_res.value != "NOT_FOUND":
                cands.append({
                    "text": mrz_res.value,
                    "raw_text": mrz_res.raw_text,
                    "bbox": mrz_res.bounding_box,
                    "ocr_confidence": mrz_res.confidence,
                    "page_source": "mrz"
                })
            if vis_res and vis_res.value != "NOT_FOUND":
                cands.append({
                    "text": vis_res.value,
                    "raw_text": vis_res.raw_text,
                    "bbox": vis_res.bounding_box,
                    "ocr_confidence": vis_res.confidence,
                    "page_source": "visual"
                })
                
            # Free text OCR candidates
            if f == "passport_number":
                for m in re.finditer(r"\b([A-Z][0-9OISZB]{7})\b", raw_text):
                    val = m.group(1)
                    corrected = val[0].upper() + ocr_correct_digits(val[1:])
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
                # Add free text candidates (all uppercase lines and adjacent combinations)
                for idx, line in enumerate(raw_lines):
                    if any(lbl in line.lower() for lbl in ["passport", "republic", "india", "nationality", "date", "birth", "sex", "expiry", "issue", "place"]):
                        continue
                    if sum(c.isdigit() for c in line) >= 3:
                        continue
                    candidate_name = extract_uppercase_name(line)
                    if candidate_name != "NOT_FOUND" and is_valid_name(candidate_name):
                        cands.append({
                            "text": candidate_name,
                            "raw_text": line,
                            "bbox": self.merge_bounding_boxes(line, word_map),
                            "ocr_confidence": self.get_field_confidence(candidate_name, word_map),
                            "page_source": "free_text",
                            "line_number": idx
                        })
                    
                    # Try combining adjacent uppercase lines (e.g., Given Name + Surname)
                    if idx + 1 < len(raw_lines):
                        next_line = raw_lines[idx+1]
                        # Rule 5: Name extraction must stop at next field label
                        if not PassportExtractor.contains_field_label(next_line):
                            if sum(c.isdigit() for c in next_line) < 3:
                                cand_next = extract_uppercase_name(next_line)
                                if candidate_name != "NOT_FOUND" and cand_next != "NOT_FOUND":
                                    combined_name = f"{candidate_name} {cand_next}"
                                    if is_valid_name(combined_name):
                                        cands.append({
                                            "text": combined_name,
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

            # Deduplicate by text
            seen_texts = set()
            unique_cands = []
            for c in cands:
                t = c["text"].strip().upper()
                if t not in seen_texts:
                    seen_texts.add(t)
                    unique_cands.append(c)

            results[f] = self.select_best_candidate(f, unique_cands, DocumentType.PASSPORT, word_map, raw_text)

        # Non-MRZ visual fields
        results["place_of_birth"] = visual_results.get("place_of_birth", FieldResult(value="NOT_FOUND", raw_text="", confidence=0.0, bounding_box=None))
        results["place_of_issue"] = visual_results.get("place_of_issue", FieldResult(value="NOT_FOUND", raw_text="", confidence=0.0, bounding_box=None))

        # Always include MRZ lines in results
        results["mrz_line1"] = FieldResult(value=mrz_line1 or "NOT_FOUND", raw_text=mrz_line1 or "", confidence=0.99 if mrz_line1 else 0.0, bounding_box=None)
        results["mrz_line2"] = FieldResult(value=mrz_line2 or "NOT_FOUND", raw_text=mrz_line2 or "", confidence=0.99 if mrz_line2 else 0.0, bounding_box=None)

        # Ensure all required fields exist in results
        required_fields = ["name", "passport_number", "nationality", "dob", "expiry", "mrz_line1", "mrz_line2"]
        for f in required_fields:
            if f not in results:
                results[f] = FieldResult(value="NOT_FOUND", raw_text="", confidence=0.0, bounding_box=None)

        # Cross-Field Validations (Additional Requirement 5)
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

        # Field-to-BoundingBox Mapping (Single Source of Truth)
        for field_name, field_res in results.items():
            if field_res.value != "NOT_FOUND":
                bbox, constituent = self.map_field_to_bbox(field_name, field_res.value, word_map)
                field_res.bounding_box = bbox
                field_res.constituent_boxes = constituent

        return results

    def _extract_from_mrz(self, m1: str, m2: str, word_map: List[Dict[str, Any]]) -> Dict[str, FieldResult]:
        res = {}
        
        # Passport Number: chars 0-9 of Line 2 (9 chars, usually ends in <)
        pass_no_raw = m2[0:9].replace("<", "")
        if len(pass_no_raw) >= 8:
            letter_part = pass_no_raw[0].upper()
            digit_to_letter = {'2': 'Z', '0': 'O', '1': 'I', '8': 'B', '5': 'S', '6': 'G'}
            if letter_part.isdigit():
                letter_part = digit_to_letter.get(letter_part, letter_part)
            digit_part = ocr_correct_digits(pass_no_raw[1:])
            passport_number = letter_part + digit_part
        else:
            passport_number = pass_no_raw

        pass_chk_ok = ChecksumValidator.validate_mrz_checksum(m2[0:9], m2[9])
        pass_conf = 0.99 if pass_chk_ok else self.get_field_confidence(passport_number, word_map)
        pass_bbox = self.merge_bounding_boxes(passport_number, word_map)
        res["passport_number"] = FieldResult(value=passport_number, raw_text=m2[0:9], confidence=pass_conf, bounding_box=pass_bbox)

        # Nationality: chars 10-13 of Line 2
        nationality = m2[10:13].upper().replace("<", "")
        if nationality in ["IND", "1ND", "IMD", "1MD", "IUD", "1UD"]:
            nationality = "IND"
        nat_conf = self.get_field_confidence(nationality, word_map)
        res["nationality"] = FieldResult(value=nationality, raw_text=m2[10:13], confidence=nat_conf, bounding_box=None)

        # DOB: chars 13-19 of Line 2 (YYMMDD)
        dob_raw = ocr_correct_digits(m2[13:19])
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

        # Sex: char 20 of Line 2
        sex_char = m2[20].upper()
        if sex_char == "M":
            sex = "M"
        elif sex_char == "F":
            sex = "F"
        else:
            sex = "M" if sex_char in ["N", "H"] else ("F" if sex_char in ["P", "R", "E"] else "M")
        sex_conf = self.get_field_confidence(sex_char, word_map)
        res["sex"] = FieldResult(value=sex, raw_text=m2[20], confidence=sex_conf, bounding_box=None)

        # Expiry: chars 21-27 of Line 2 (YYMMDD)
        exp_raw = ocr_correct_digits(m2[21:27])
        exp_yy = exp_raw[0:2]
        exp_mm = exp_raw[2:4]
        exp_dd = exp_raw[4:6]
        expiry = f"20{exp_yy}-{exp_mm}-{exp_dd}"

        exp_chk_ok = ChecksumValidator.validate_mrz_checksum(m2[21:27], m2[27])
        exp_conf = 0.99 if exp_chk_ok else self.get_field_confidence(exp_raw, word_map)
        res["expiry"] = FieldResult(value=expiry, raw_text=m2[21:27], confidence=exp_conf, bounding_box=None)

        # Name: From line 1, after country code (chars 5-44)
        country_code_area = m1[2:5]
        if "<" not in country_code_area:
            name_part = m1[5:]
        else:
            start_idx = 2
            while start_idx < len(m1) and m1[start_idx] == "<":
                start_idx += 1
            name_part = m1[start_idx:]

        parts = [p.replace("<", " ").strip() for p in name_part.split("<<") if p.replace("<", " ").strip()]
        clean_parts = []
        for p in parts:
            p_clean = re.sub(r'\s+', ' ', p).strip()
            if any(c.isdigit() for c in p_clean):
                continue
            if len(p_clean) < 2:
                continue
            clean_parts.append(p_clean)
            
        if len(clean_parts) >= 2:
            surname = clean_parts[0]
            given_names = clean_parts[1]
            full_name = f"{given_names} {surname}".strip()
        elif len(clean_parts) == 1:
            full_name = clean_parts[0]
        else:
            full_name = "UNKNOWN"

        name_conf = self.get_field_confidence(full_name, word_map)
        name_bbox = self.merge_bounding_boxes(full_name, word_map)
        res["name"] = FieldResult(value=full_name, raw_text=name_part, confidence=name_conf, bounding_box=name_bbox)

        return res

    def _extract_from_visual(self, raw_text: str, lines: List[str], word_map: List[Dict[str, Any]]) -> Dict[str, FieldResult]:
        res = {}

        # 1. Passport Number
        m_pass = re.search(r"\b([A-Z][0-9]{7})\b", raw_text)
        if m_pass:
            pass_val = m_pass.group(1)
            bbox = self.merge_bounding_boxes(pass_val, word_map)
            res["passport_number"] = FieldResult(value=pass_val, raw_text=m_pass.group(0), confidence=self.get_field_confidence(pass_val, word_map), bounding_box=bbox)
        else:
            res["passport_number"] = FieldResult(value="NOT_FOUND", raw_text="", confidence=0.0, bounding_box=None)

        # 2. Nationality
        m_nat = re.search(r"\b(indian|republic\s+of\s+india)\b", raw_text, re.IGNORECASE)
        if m_nat:
            res["nationality"] = FieldResult(value="IND", raw_text=m_nat.group(0), confidence=self.get_field_confidence("IND", word_map), bounding_box=None)
        else:
            res["nationality"] = FieldResult(value="IND", raw_text="INDIAN", confidence=self.get_field_confidence("IND", word_map), bounding_box=None)

        # 3. DOB and Expiry (via multiple date detection and label matching)
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

        # 4. Sex
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

        # 5. Name
        name_val = "NOT_FOUND"
        name_raw = ""
        surname_val = ""
        given_names_val = ""
        surname_raw = ""
        given_names_raw = ""
        for idx, line in enumerate(lines):
            l_lower = line.lower()
            if "surname" in l_lower and idx + 1 < len(lines):
                candidate = extract_uppercase_name(lines[idx + 1])
                if candidate != "NOT_FOUND":
                    surname_val = candidate
                    surname_raw = lines[idx + 1]
            if ("givenname" in l_lower or ("given" in l_lower and "name" in l_lower)) and idx + 1 < len(lines):
                candidate = extract_uppercase_name(lines[idx + 1])
                if candidate != "NOT_FOUND":
                    given_names_val = candidate
                    given_names_raw = lines[idx + 1]

        if given_names_val or surname_val:
            name_val = f"{given_names_val} {surname_val}".strip()
            name_raw = f"{given_names_raw} {surname_raw}".strip()

        if name_val == "NOT_FOUND":
            candidates = []
            for line in lines:
                if any(lbl in line.lower() for lbl in ["passport", "republic", "india", "nationality", "date", "birth", "sex", "expiry", "issue", "place"]):
                    continue
                if sum(c.isdigit() for c in line) >= 3:
                    continue
                candidate = extract_uppercase_name(line)
                if candidate != "NOT_FOUND" and is_valid_name(candidate):
                    candidates.append((line, candidate))
            
            if candidates:
                if len(candidates) >= 2:
                    name_val = f"{candidates[0][1]} {candidates[1][1]}"
                    name_raw = f"{candidates[0][0]} {candidates[1][0]}"
                else:
                    name_val = candidates[0][1]
                    name_raw = candidates[0][0]

        if name_val != "NOT_FOUND":
            res["name"] = FieldResult(value=name_val, raw_text=name_raw, confidence=self.get_field_confidence(name_val, word_map), bounding_box=self.merge_bounding_boxes(name_raw, word_map))
        else:
            res["name"] = FieldResult(value="NOT_FOUND", raw_text="", confidence=0.0, bounding_box=None)

        # 6. Place of Birth / Place of Issue
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
