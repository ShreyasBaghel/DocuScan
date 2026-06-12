import re
from typing import Tuple, List, Dict, Any
from classifiers.base_classifier import BaseClassifier
from utils.document_packet import DocumentType

class KeywordClassifier(BaseClassifier):
    def __init__(self):
        # Broaden keyword list to include common OCR errors and partial matches
        self.keywords = {
            DocumentType.AADHAAR: [
                r"unique\s+identification",
                r"uidai",
                r"aadhaar",
                r"adhar",
                r"bharat\s+sarkar",
                r"भारत\s+सरकार",
                r"विशिष्ट\s+पहचान",
                r"mera\s+aadhaar",
                r"yob\s*:",
                r"enrollment\s+no",
                r"enrolment\s+no"
            ],
            DocumentType.PAN: [
                r"income\s+tax",
                r"permanent\s+account",
                r"perm{1,2}anent\s+acc",
                r"आयकर\s+विभाग",
                r"pan\s+card",
                r"father\'s\s+name",
                r"govt\.\s+of\s+india"
            ],
            DocumentType.PASSPORT: [
                r"republic\s+of\s+india",
                r"passport\s+no",
                r"भारत\s+गणराज्य",
                r"passport",
                r"nationality",
                r"place\s+of\s+birth",
                r"type\s+code",
                r"sur\s*name",
                r"given\s+name"
            ],
            DocumentType.DRIVING_LICENCE: [
                r"driving\s+licen",
                r"driving\s+lic",
                r"licence\s+to\s+drive",
                r"licensing\s+auth",
                r"transport\s+dep",
                r"class\s+of\s+veh",
                r"lmv",
                r"mcwg",
                r"badge\s+no"
            ]
        }

    def classify(self, raw_text: str, word_map: List[Dict[str, Any]]) -> Tuple[DocumentType, float]:
        if not raw_text:
            return DocumentType.UNKNOWN, 0.0

        # Normalise input text: lowercase, strip punctuation except spaces, clean multiple spaces
        clean_text = raw_text.lower()
        clean_text = re.sub(r'[^\w\s\u0900-\u097F]', ' ', clean_text)  # Keep devanagari characters
        clean_text = " ".join(clean_text.split())

        scores: Dict[DocumentType, float] = {}

        for doc_type, regex_list in self.keywords.items():
            matches = 0
            for regex in regex_list:
                if re.search(regex, clean_text, re.IGNORECASE):
                    matches += 1
            # Calculate absolute score based on number of keyword matches
            if matches >= 2:
                scores[doc_type] = 0.90
            elif matches == 1:
                scores[doc_type] = 0.75
            else:
                scores[doc_type] = 0.0

        # Find the document type with the highest score
        best_doc = DocumentType.UNKNOWN
        best_score = 0.0

        for doc_type, score in scores.items():
            if score > best_score:
                best_score = score
                best_doc = doc_type

        # If best score is negligible, classify as UNKNOWN
        if best_score < 0.50:
            return DocumentType.UNKNOWN, 0.0

        return best_doc, best_score
