import re
from typing import Tuple, List, Dict, Any
from classifiers.base_classifier import BaseClassifier
from utils.document_packet import DocumentType
from utils.string_utils import ocr_correct_digits
from validators.checksum_validator import ChecksumValidator

class RegexClassifier(BaseClassifier):
    def __init__(self):
        # Strict regular expressions for ID numbering formats
        self.strict_patterns = {
            DocumentType.AADHAAR: [
                r"\b(\d{4}\s+\d{4}\s+\d{4})\b",
                r"\b(\d{12})\b"
            ],
            DocumentType.PAN: [
                r"\b([A-Z]{5}[0-9]{4}[A-Z])\b"
            ],
            DocumentType.PASSPORT: [
                r"\b([A-Z][0-9]{7})\b"
            ],
            DocumentType.DRIVING_LICENCE: [
                r"\b([A-Z]{2}\s*[-/]?\s*[0-9]{2}\s*[-/]?\s*[0-9]{4}\s*[-/]?\s*[0-9]{7})\b",
                r"\b([A-Z]{2}[0-9]{2}[A-Z][0-9]{11})\b"
            ]
        }

        # OCR-tolerant "soft" regular expressions allowing typical digit/letter confusions
        # [0-9OISZB] matches digits and characters commonly misread by OCR as digits
        self.soft_patterns = {
            DocumentType.AADHAAR: [
                r"\b([0-9OISZB]{4}\s+[0-9OISZB]{4}\s+[0-9OISZB]{4})\b",
                r"\b([0-9OISZB]{12})\b"
            ],
            DocumentType.PAN: [
                r"\b([A-Z]{5}[0-9OISZB]{4}[A-Z])\b"
            ],
            DocumentType.PASSPORT: [
                r"\b([A-Z][0-9OISZB]{7})\b"
            ],
            DocumentType.DRIVING_LICENCE: [
                r"\b([A-Z]{2}\s*[-/]?\s*[0-9OISZB]{2}\s*[-/]?\s*[0-9OISZB]{4}\s*[-/]?\s*[0-9OISZB]{7})\b"
            ]
        }

    def get_match_confidence(self, matched_str: str, word_map: List[Dict[str, Any]]) -> float:
        """Calculates dynamic OCR confidence for matched regex string from word_map."""
        if not matched_str or not word_map:
            return 0.85 # sensible default
        
        # Clean and split matched string into tokens
        import numpy as np
        tokens = [t.lower() for t in matched_str.split() if re.match(r"^[a-zA-Z0-9/<]+$", t)]
        if not tokens:
            all_confs = [w['conf'] for w in word_map if 'conf' in w]
            return float(np.mean(all_confs)) if all_confs else 0.85

        matched_confs = []
        for token in tokens:
            clean_token = re.sub(r'[^a-z0-9/<]', '', token)
            if not clean_token:
                continue
            for w in word_map:
                w_text = w.get('text', '').lower()
                clean_w_text = re.sub(r'[^a-z0-9/<]', '', w_text)
                if clean_token == clean_w_text and 'conf' in w:
                    matched_confs.append(w['conf'])

        if matched_confs:
            return float(np.mean(matched_confs))

        all_confs = [w['conf'] for w in word_map if 'conf' in w]
        return float(np.mean(all_confs)) if all_confs else 0.85

    def classify(self, raw_text: str, word_map: List[Dict[str, Any]]) -> Tuple[DocumentType, float]:
        if not raw_text:
            return DocumentType.UNKNOWN, 0.0

        # Standardize spaces for easier regex matching
        flat_text = " ".join(raw_text.split())

        best_doc = DocumentType.UNKNOWN
        best_score = 0.0

        # 1. First Pass: Check strict matches
        for doc_type, regex_list in self.strict_patterns.items():
            for pattern in regex_list:
                m = re.search(pattern, flat_text, re.IGNORECASE)
                if m:
                    matched_str = m.group(1)
                    ocr_conf = self.get_match_confidence(matched_str, word_map)
                    
                    # Special validation for Aadhaar strict matches
                    if doc_type == DocumentType.AADHAAR:
                        cleaned = re.sub(r"\s+", "", matched_str)
                        if ChecksumValidator.validate_verhoeff(cleaned):
                            score = 0.90 + 0.10 * ocr_conf
                        else:
                            # Strict match failed Verhoeff (might be a false match)
                            score = 0.60 + 0.10 * ocr_conf
                    else:
                        is_passport = doc_type == DocumentType.PASSPORT and "p<ind" in flat_text.lower()
                        score_base = 0.90 if (doc_type != DocumentType.PASSPORT or is_passport) else 0.80
                        score = score_base + 0.10 * ocr_conf
                    
                    if score > best_score:
                        best_score = score
                        best_doc = doc_type

        # Passport MRZ check
        if "p<ind" in flat_text.lower() and best_score < 0.95:
            ocr_conf = self.get_match_confidence("p<ind", word_map)
            best_score = 0.88 + 0.10 * ocr_conf
            best_doc = DocumentType.PASSPORT

        # 2. Second Pass: Check soft matches if no confident strict match was found
        if best_score < 0.90:
            for doc_type, regex_list in self.soft_patterns.items():
                for pattern in regex_list:
                    m = re.search(pattern, flat_text, re.IGNORECASE)
                    if m:
                        matched_str = m.group(1)
                        ocr_conf = self.get_match_confidence(matched_str, word_map)
                        
                        if doc_type == DocumentType.AADHAAR:
                            # Autocorrect substitutions and check Verhoeff
                            cleaned = re.sub(r"\s+", "", matched_str)
                            corrected = ocr_correct_digits(cleaned)
                            if ChecksumValidator.validate_verhoeff(corrected):
                                score = 0.85 + 0.10 * ocr_conf
                                if score > best_score:
                                    best_score = score
                                    best_doc = doc_type
                        elif doc_type == DocumentType.PAN:
                            score = 0.75 + 0.10 * ocr_conf
                            if score > best_score:
                                best_score = score
                                best_doc = doc_type
                        elif doc_type == DocumentType.PASSPORT:
                            score = 0.70 + 0.10 * ocr_conf
                            if score > best_score:
                                best_score = score
                                best_doc = doc_type
                        elif doc_type == DocumentType.DRIVING_LICENCE:
                            score = 0.70 + 0.10 * ocr_conf
                            if score > best_score:
                                best_score = score
                                best_doc = doc_type

        return best_doc, best_score
