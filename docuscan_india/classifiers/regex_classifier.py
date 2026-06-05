import re
from typing import Tuple, List, Dict, Any
from classifiers.base_classifier import BaseClassifier
from utils.document_packet import DocumentType

class RegexClassifier(BaseClassifier):
    def __init__(self):
        # Strict regular expressions for ID numbering formats
        self.patterns = {
            DocumentType.AADHAAR: [
                r"\b\d{4}\s+\d{4}\s+\d{4}\b",       # Aadhaar format: 1234 5678 9012
                r"\b\d{12}\b"                         # 12 consecutive digits
            ],
            DocumentType.PAN: [
                r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"         # PAN: 5 letters, 4 digits, 1 letter
            ],
            DocumentType.PASSPORT: [
                r"\b[A-Z][0-9]{7}\b",                # Indian Passport: 1 letter, 7 digits
                r"p<ind[a-z0-9<]*"                   # MRZ Line 1 starts with P<IND
            ],
            DocumentType.DRIVING_LICENCE: [
                # DL format: State (2 letters) + RTO (2 digits) + Year (4 digits) + LicNo (7 digits)
                r"\b[A-Z]{2}\s*[-/]?\s*[0-9]{2}\s*[-/]?\s*[0-9]{4}\s*[-/]?\s*[0-9]{7}\b",
                r"\b[A-Z]{2}[0-9]{2}[A-Z][0-9]{11}\b" # Alternative formats
            ]
        }

    def classify(self, raw_text: str, word_map: List[Dict[str, Any]]) -> Tuple[DocumentType, float]:
        if not raw_text:
            return DocumentType.UNKNOWN, 0.0

        # Replace standard newlines/multiple spaces with single space for easier regex matching
        flat_text = " ".join(raw_text.split())

        best_doc = DocumentType.UNKNOWN
        best_score = 0.0

        for doc_type, regex_list in self.patterns.items():
            matches = 0
            for pattern in regex_list:
                # Passport MRZ case-insensitive or raw flat text
                if re.search(pattern, flat_text, re.IGNORECASE):
                    matches += 1
            
            if matches > 0:
                # If a strict ID pattern matches, give high confidence
                score = 0.95 if matches == 1 else 1.0
                if score > best_score:
                    best_score = score
                    best_doc = doc_type

        return best_doc, best_score
