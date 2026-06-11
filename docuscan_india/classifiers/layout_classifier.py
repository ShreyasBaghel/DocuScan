from typing import Tuple, List, Dict, Any
from classifiers.base_classifier import BaseClassifier
from utils.document_packet import DocumentType
import re

class LayoutClassifier(BaseClassifier):
    def classify(self, raw_text: str, word_map: List[Dict[str, Any]]) -> Tuple[DocumentType, float]:
        if not word_map:
            # Fallback to checking raw text for Passport MRZ `<` characters
            if raw_text:
                count_brackets = raw_text.count("<")
                if count_brackets >= 15:
                    return DocumentType.PASSPORT, 0.70
            return DocumentType.UNKNOWN, 0.0

        # Sort the word map: first top-to-bottom, then left-to-right on similar lines (vertical tolerance 15px)
        sorted_words = []
        unsorted = list(word_map)
        
        while unsorted:
            # Pick first word
            first = min(unsorted, key=lambda w: w['top'])
            # Get all words on the same line (within 15px top difference)
            line_words = [w for w in unsorted if abs(w['top'] - first['top']) < 15]
            # Sort line words left-to-right
            line_words.sort(key=lambda w: w['left'])
            sorted_words.extend(line_words)
            # Remove from unsorted
            for w in line_words:
                unsorted.remove(w)

        # Calculate bounding dimensions of the document
        max_x = max(w['left'] + w['width'] for w in sorted_words)
        max_y = max(w['top'] + w['height'] for w in sorted_words)
        
        if max_y == 0 or max_x == 0:
            return DocumentType.UNKNOWN, 0.0

        passport_score = 0.0
        aadhaar_score = 0.0
        pan_score = 0.0
        dl_score = 0.0

        # 1. PASSPORT check:
        # A. Look for MRZ-like structures at the bottom
        bottom_words = [w for w in sorted_words if w['top'] > 0.70 * max_y]
        lines: Dict[int, List[Dict[str, Any]]] = {}
        for w in bottom_words:
            matched_line = False
            for line_y in lines.keys():
                if abs(w['top'] - line_y) < 15:
                    lines[line_y].append(w)
                    matched_line = True
                    break
            if not matched_line:
                lines[w['top']] = [w]

        mrz_like_lines = 0
        for line_y, line_words in lines.items():
            line_words.sort(key=lambda x: x['left'])
            line_text = "".join(w['text'] for w in line_words)
            line_text_clean = re.sub(r'[^a-zA-Z0-9<]', '', line_text)
            if len(line_text_clean) >= 25 and '<' in line_text_clean:
                mrz_like_lines += 1

        # B. Density of `<` character in general
        brackets_count = raw_text.count("<")
        if mrz_like_lines >= 1 or brackets_count >= 15:
            passport_score = 0.90 if mrz_like_lines == 1 or brackets_count < 20 else 1.0

        # 2. AADHAAR check: Look for 12-digit Aadhaar number horizontally aligned in lower 40%
        # Use our sorted word list to reliably locate three consecutive 4-digit groups on the same line
        aadhaar_number_found = False
        for i in range(len(sorted_words) - 2):
            w1, w2, w3 = sorted_words[i], sorted_words[i+1], sorted_words[i+2]
            # Check if they are on the same line
            if abs(w1['top'] - w2['top']) < 15 and abs(w2['top'] - w3['top']) < 15:
                # Check if all are 4-digit strings
                if re.match(r"^\d{4}$", w1['text']) and re.match(r"^\d{4}$", w2['text']) and re.match(r"^\d{4}$", w3['text']):
                    # Check if they flow left-to-right
                    if w1['left'] < w2['left'] < w3['left']:
                        aadhaar_number_found = True
                        break
        
        if aadhaar_number_found:
            aadhaar_score = 0.85

        # 3. PAN check: Name and Father's Name labels
        father_name_label = False
        name_label = False
        for w in sorted_words:
            text = w['text'].lower()
            if "father" in text or "पिता" in text:
                father_name_label = True
            if "name" in text or "नाम" in text:
                name_label = True

        if father_name_label and name_label and not aadhaar_number_found:
            pan_score = 0.60

        # 4. DRIVING LICENCE check: MCWG / LMV keywords
        vehicle_classes = 0
        for w in sorted_words:
            text = w['text'].upper()
            if text in ["LMV", "MCWG", "MCWOG", "HMV", "TRANS"]:
                vehicle_classes += 1
        
        if vehicle_classes >= 2:
            dl_score = 0.80
        elif vehicle_classes == 1:
            dl_score = 0.40

        scores = {
            DocumentType.PASSPORT: passport_score,
            DocumentType.AADHAAR: aadhaar_score,
            DocumentType.PAN: pan_score,
            DocumentType.DRIVING_LICENCE: dl_score
        }

        best_doc = DocumentType.UNKNOWN
        best_score = 0.0

        for doc_type, score in scores.items():
            if score > best_score:
                best_score = score
                best_doc = doc_type

        if best_score < 0.20:
            return DocumentType.UNKNOWN, 0.0

        return best_doc, best_score
