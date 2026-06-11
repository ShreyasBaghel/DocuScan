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
                    # Special validation for Aadhaar strict matches
                    if doc_type == DocumentType.AADHAAR:
                        cleaned = re.sub(r"\s+", "", m.group(1))
                        if ChecksumValidator.validate_verhoeff(cleaned):
                            return DocumentType.AADHAAR, 1.0
                        else:
                            # Strict match failed Verhoeff (might be a false match)
                            # We still score it but with lower confidence
                            score = 0.70
                    else:
                        score = 0.95 if doc_type != DocumentType.PASSPORT or "p<ind" in flat_text.lower() else 0.90
                    
                    if score > best_score:
                        best_score = score
                        best_doc = doc_type

        # Passport MRZ check
        if "p<ind" in flat_text.lower() and best_score < 0.95:
            best_score = 0.98
            best_doc = DocumentType.PASSPORT

        # 2. Second Pass: Check soft matches if no confident strict match was found
        if best_score < 0.90:
            for doc_type, regex_list in self.soft_patterns.items():
                for pattern in regex_list:
                    m = re.search(pattern, flat_text, re.IGNORECASE)
                    if m:
                        matched_str = m.group(1)
                        if doc_type == DocumentType.AADHAAR:
                            # Autocorrect substitutions and check Verhoeff
                            cleaned = re.sub(r"\s+", "", matched_str)
                            corrected = ocr_correct_digits(cleaned)
                            if ChecksumValidator.validate_verhoeff(corrected):
                                return DocumentType.AADHAAR, 0.95
                        elif doc_type == DocumentType.PAN:
                            # Soft PAN match
                            score = 0.85
                            if score > best_score:
                                best_score = score
                                best_doc = doc_type
                        elif doc_type == DocumentType.PASSPORT:
                            # Soft Passport match
                            score = 0.80
                            if score > best_score:
                                best_score = score
                                best_doc = doc_type
                        elif doc_type == DocumentType.DRIVING_LICENCE:
                            # Soft DL match
                            score = 0.80
                            if score > best_score:
                                best_score = score
                                best_doc = doc_type

        return best_doc, best_score
