import re
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from utils.document_packet import FieldResult, DocumentType
from utils.string_utils import normalize_date, clean_whitespace

class BaseExtractor(ABC):
    @abstractmethod
    def extract(self, raw_text: str, word_map: List[Dict[str, Any]]) -> Dict[str, FieldResult]:
        """
        Extracts document-specific fields from text and OCR word map.
        Returns a dictionary of field names to FieldResult.
        """
        pass

    def merge_bounding_boxes(self, target_value: str, word_map: List[Dict[str, Any]]) -> Optional[Dict[str, int]]:
        """
        Finds the bounding box of a target value by finding matching words in the word map
        and computing the bounding box that encompasses them.
        """
        if not target_value or not word_map:
            return None

        # Clean and split target value into tokens
        tokens = [t.lower() for t in target_value.split() if re.match(r"^[a-zA-Z0-9/<]+$", t)]
        if not tokens:
            return None

        matched_boxes = []

        # Find words matching tokens
        for token in tokens:
            for w in word_map:
                w_text = w['text'].lower()
                # Check for exact or partial matches
                if token in w_text or w_text in token:
                    matched_boxes.append(w)

        if not matched_boxes:
            return None

        # Compute bounding box encompassing all matched words
        min_x = min(w['left'] for w in matched_boxes)
        min_y = min(w['top'] for w in matched_boxes)
        max_r = max(w['left'] + w['width'] for w in matched_boxes)
        max_b = max(w['top'] + w['height'] for w in matched_boxes)

        return {
            'x': min_x,
            'y': min_y,
            'w': max_r - min_x,
            'h': max_b - min_y
        }

    def extract_field_with_regex(self, pattern: str, text: str, word_map: List[Dict[str, Any]], group_name: str = None) -> FieldResult:
        """Helper to run regex on raw text, return FieldResult with bounding box."""
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            val = m.group(group_name) if group_name else m.group(0)
            val_clean = clean_whitespace(val)
            bbox = self.merge_bounding_boxes(val_clean, word_map)
            # Estimate confidence from matching words (or default to 0.90)
            conf = 0.90
            matched_words = [w for w in word_map if w['text'] in val_clean]
            if matched_words:
                conf = sum(w['conf'] for w in matched_words) / len(matched_words)
            return FieldResult(value=val_clean, raw_text=m.group(0), confidence=conf, bounding_box=bbox)
        
        return FieldResult(value="NOT_FOUND", raw_text="", confidence=0.0, bounding_box=None)
