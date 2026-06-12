import re
import numpy as np
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
            clean_token = re.sub(r'[^a-z0-9/<]', '', token)
            if not clean_token:
                continue
            for w in word_map:
                w_text = w['text'].lower()
                clean_w_text = re.sub(r'[^a-z0-9/<]', '', w_text)
                if clean_token == clean_w_text:
                    matched_boxes.append(w)

        if not matched_boxes:
            return None

        # Group matched boxes by line (y-coordinate)
        line_groups = []
        for box in matched_boxes:
            box_center_y = box['top'] + box['height'] / 2
            added = False
            for group in line_groups:
                group_center_y = sum(b['top'] + b['height'] / 2 for b in group) / len(group)
                if abs(box_center_y - group_center_y) < 20:  # 20 pixels vertical tolerance
                    group.append(box)
                    added = True
                    break
            if not added:
                line_groups.append([box])

        # Pick the line group that has the most matched boxes
        best_group = max(line_groups, key=len)
        matched_boxes = best_group

        # Compute bounding box encompassing all matched words in the best group
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

    def get_field_confidence(self, value: str, word_map: List[Dict[str, Any]], base_conf: float = 0.85) -> float:
        """Calculates a field confidence dynamically based on word-level confidences."""
        if not value or value == "NOT_FOUND" or not word_map:
            return 0.0

        # Clean and split target value into tokens
        tokens = [t.lower() for t in value.split() if re.match(r"^[a-zA-Z0-9/<]+$", t)]
        if not tokens:
            # Fallback to mean of all word confidences
            all_confs = [w['conf'] for w in word_map if 'conf' in w]
            return float(np.mean(all_confs)) if all_confs else base_conf

        matched_confs = []
        for token in tokens:
            clean_token = re.sub(r'[^a-z0-9/<]', '', token)
            if not clean_token:
                continue
            for w in word_map:
                w_text = w['text'].lower()
                clean_w_text = re.sub(r'[^a-z0-9/<]', '', w_text)
                if clean_token == clean_w_text:
                    matched_confs.append(w['conf'])

        if matched_confs:
            return float(np.mean(matched_confs))

        # Dynamic fallback: mean of all word confidences
        all_confs = [w['conf'] for w in word_map if 'conf' in w]
        return float(np.mean(all_confs)) if all_confs else base_conf

    def extract_field_with_regex(self, pattern: str, text: str, word_map: List[Dict[str, Any]], group_name: str = None) -> FieldResult:
        """Helper to run regex on raw text, return FieldResult with bounding box."""
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            val = m.group(group_name) if group_name else m.group(0)
            val_clean = clean_whitespace(val)
            bbox = self.merge_bounding_boxes(val_clean, word_map)
            # Estimate confidence dynamically from matching words
            conf = self.get_field_confidence(val_clean, word_map)
            return FieldResult(value=val_clean, raw_text=m.group(0), confidence=conf, bounding_box=bbox)
        
        return FieldResult(value="NOT_FOUND", raw_text="", confidence=0.0, bounding_box=None)

