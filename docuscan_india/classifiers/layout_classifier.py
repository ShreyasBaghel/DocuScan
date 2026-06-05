from typing import Tuple, List, Dict, Any
from classifiers.base_classifier import BaseClassifier
from utils.document_packet import DocumentType
import re

class LayoutClassifier(BaseClassifier):
    def classify(self, raw_text: str, word_map: List[Dict[str, Any]]) -> Tuple[DocumentType, float]:
        if not word_map:
            return DocumentType.UNKNOWN, 0.0

        # Calculate bounding dimensions of the document
        max_x = max(w['left'] + w['width'] for w in word_map)
        max_y = max(w['top'] + w['height'] for w in word_map)
        
        if max_y == 0 or max_x == 0:
            return DocumentType.UNKNOWN, 0.0

        passport_score = 0.0
        aadhaar_score = 0.0
        pan_score = 0.0
        dl_score = 0.0

        # 1. PASSPORT check: Look for MRZ-like structures at the bottom
        # MRZ consists of two lines of 44 characters containing '<' at the bottom 25% of height
        bottom_words = [w for w in word_map if w['top'] > 0.70 * max_y]
        
        # Group bottom words by line (similar top coordinates, e.g. within 20px)
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
            # Sort words by left coordinate
            line_words.sort(key=lambda x: x['left'])
            line_text = "".join(w['text'] for w in line_words)
            # Clean text but keep '<'
            line_text_clean = re.sub(r'[^a-zA-Z0-9<]', '', line_text)
            if len(line_text_clean) >= 30 and '<' in line_text_clean:
                mrz_like_lines += 1

        if mrz_like_lines >= 1:
            passport_score = 0.90 if mrz_like_lines == 1 else 1.0

        # 2. AADHAAR check: Look for 12-digit Aadhaar number horizontally aligned in the lower 40% of the card
        lower_words = [w for w in word_map if w['top'] > 0.50 * max_y]
        
        # Check if we have three groups of 4 digits close to each other on the same line
        # We can scan the word map for digit groups
        aadhaar_number_found = False
        for i in range(len(word_map) - 2):
            w1, w2, w3 = word_map[i], word_map[i+1], word_map[i+2]
            if abs(w1['top'] - w2['top']) < 10 and abs(w2['top'] - w3['top']) < 10:
                # Check if all are 4-digit strings
                if re.match(r"^\d{4}$", w1['text']) and re.match(r"^\d{4}$", w2['text']) and re.match(r"^\d{4}$", w3['text']):
                    # Check if they are sequential horizontally
                    if w1['left'] < w2['left'] < w3['left'] and (w3['left'] - w1['left']) < max_x * 0.5:
                        aadhaar_number_found = True
                        break
        
        if aadhaar_number_found:
            aadhaar_score = 0.85

        # 3. PAN check: Usually has Permanent Account Number label and signature box area
        # Check for stacked layout: Name label, Father's Name label
        father_name_label = False
        name_label = False
        for w in word_map:
            text = w['text'].lower()
            if "father" in text or "पिता" in text:
                father_name_label = True
            if "name" in text or "नाम" in text:
                name_label = True

        if father_name_label and name_label and not aadhaar_number_found:
            pan_score = 0.60

        # 4. DRIVING LICENCE check: Check for vehicle classes (LMV, MCWG, TRANS) layout
        vehicle_classes = 0
        for w in word_map:
            text = w['text'].upper()
            if text in ["LMV", "MCWG", "MCWOG", "HMV", "TRANS"]:
                vehicle_classes += 1
        
        if vehicle_classes >= 2:
            dl_score = 0.80
        elif vehicle_classes == 1:
            dl_score = 0.40

        # Output the best match
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
