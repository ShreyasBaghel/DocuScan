import re
from typing import Tuple, List, Dict, Any, Optional
from PIL import Image
from utils.document_packet import DocumentPacket, FieldResult
from utils.logger import get_logger

logger = get_logger("passport_splitter")

class PassportSplitter:
    @staticmethod
    def count_mrz_indicators(text: str) -> int:
        """
        Calculates a score indicating the presence of MRZ patterns in the text.
        """
        if not text:
            return 0
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        score = 0
        for line in lines:
            cleaned = re.sub(r'\s+', '', line).upper()
            cleaned = re.sub(r'[^A-Z0-9<]', '', cleaned)
            if len(cleaned) >= 30:
                if cleaned.startswith("P<") or (cleaned.startswith("P") and cleaned.count("<") >= 5):
                    score += 15
                elif cleaned.count("<") >= 10:
                    score += 10
                elif cleaned.count("<") >= 5:
                    score += 5
        score += text.count("<")
        return score

    @staticmethod
    def detect_split(img: Image.Image, word_map: List[Dict[str, Any]]) -> Tuple[Optional[str], Optional[int]]:
        """
        Analyzes the image and word coordinates to detect if there is a vertical or horizontal split.
        Returns:
            - split_type: "vertical", "horizontal", or None
            - split_coord: coordinate index of the split, or None
        """
        W, H = img.size
        aspect_ratio_w_h = W / H
        aspect_ratio_h_w = H / W

        can_split_vertical = aspect_ratio_w_h >= 1.15
        can_split_horizontal = aspect_ratio_h_w >= 1.15

        x_split = None
        y_split = None
        min_x_cost = float('inf')
        min_y_cost = float('inf')

        # Find best vertical split (pages side-by-side)
        if can_split_vertical:
            if word_map:
                start_x = int(0.35 * W)
                end_x = int(0.65 * W)
                for x in range(start_x, end_x, 2):
                    cost = 0
                    for w in word_map:
                        w_left = w['left']
                        w_right = w_left + w['width']
                        if w_left < x < w_right:
                            cost += 1
                    if cost < min_x_cost:
                        min_x_cost = cost
                        x_split = x
                    elif cost == min_x_cost and x_split is not None:
                        # Tie-breaker: closer to center
                        if abs(x - W/2) < abs(x_split - W/2):
                            x_split = x
            else:
                # Fallback if no word map: split at center
                if aspect_ratio_w_h >= 1.4:
                    x_split = W // 2
                    min_x_cost = 0

        # Find best horizontal split (pages stacked top/bottom)
        if can_split_horizontal:
            if word_map:
                start_y = int(0.35 * H)
                end_y = int(0.65 * H)
                for y in range(start_y, end_y, 2):
                    cost = 0
                    for w in word_map:
                        w_top = w['top']
                        w_bottom = w_top + w['height']
                        if w_top < y < w_bottom:
                            cost += 1
                    if cost < min_y_cost:
                        min_y_cost = cost
                        y_split = y
                    elif cost == min_y_cost and y_split is not None:
                        # Tie-breaker: closer to center
                        if abs(y - H/2) < abs(y_split - H/2):
                            y_split = y
            else:
                # Fallback if no word map: split at center
                if aspect_ratio_h_w >= 1.4:
                    y_split = H // 2
                    min_y_cost = 0

        # Determine if we should split
        # Threshold: cost <= 3 or <= 3% of total words
        max_allowed_cost = max(3, int(len(word_map) * 0.03)) if word_map else 0

        is_v_valid = x_split is not None and min_x_cost <= max_allowed_cost and aspect_ratio_w_h >= 1.15
        is_h_valid = y_split is not None and min_y_cost <= max_allowed_cost and aspect_ratio_h_w >= 1.15

        if is_v_valid and is_h_valid:
            # If both are valid, prioritize the one matching the dominant aspect ratio
            if W >= H:
                return "vertical", x_split
            else:
                return "horizontal", y_split
        elif is_v_valid:
            return "vertical", x_split
        elif is_h_valid:
            return "horizontal", y_split

        return None, None

    @classmethod
    def split_and_extract(cls, packet: DocumentPacket, ocr_engine, extractor) -> bool:
        """
        Performs page splitting, independent OCR and field extraction on each page,
        maps coordinates back, and merges the fields.
        Returns True if a split was processed, False otherwise.
        """
        img = packet.preprocessed_image
        if img is None:
            logger.warning("No preprocessed image in packet, cannot split.")
            return False

        W, H = img.size
        split_type, split_coord = cls.detect_split(img, packet.ocr_word_map)

        if not split_type or split_coord is None:
            logger.info("No passport split detected. Treating as single-page.")
            return False

        logger.info(f"Passport dual-page layout detected. Type: {split_type}, Coord: {split_coord}")

        # Crop regions and define offsets
        if split_type == "vertical":
            region_A = img.crop((0, 0, split_coord, H))
            region_B = img.crop((split_coord, 0, W, H))
            offset_A = (0, 0)
            offset_B = (split_coord, 0)
        else:  # horizontal
            region_A = img.crop((0, 0, W, split_coord))
            region_B = img.crop((0, split_coord, W, H))
            offset_A = (0, 0)
            offset_B = (0, split_coord)

        # Run OCR on each page independently
        raw_text_A, conf_A, word_map_A = ocr_engine.extract(region_A, "PASSPORT")
        raw_text_B, conf_B, word_map_B = ocr_engine.extract(region_B, "PASSPORT")

        # Classify which region is Page 1 (bio-data page) vs Page 2 (secondary)
        score_A = cls.count_mrz_indicators(raw_text_A)
        score_B = cls.count_mrz_indicators(raw_text_B)

        logger.info(f"MRZ indicator scores: Region A = {score_A}, Region B = {score_B}")

        if score_A >= score_B:
            page_1_raw_text = raw_text_A
            page_1_word_map = word_map_A
            p1_x_offset, p1_y_offset = offset_A

            page_2_raw_text = raw_text_B
            page_2_word_map = word_map_B
            p2_x_offset, p2_y_offset = offset_B
            logger.info("Assigned Region A -> Page 1, Region B -> Page 2")
        else:
            page_1_raw_text = raw_text_B
            page_1_word_map = word_map_B
            p1_x_offset, p1_y_offset = offset_B

            page_2_raw_text = raw_text_A
            page_2_word_map = word_map_A
            p2_x_offset, p2_y_offset = offset_A
            logger.info("Assigned Region B -> Page 1, Region A -> Page 2")

        # Extract fields from both pages
        fields_1 = extractor.extract(page_1_raw_text, page_1_word_map)
        fields_2 = extractor.extract(page_2_raw_text, page_2_word_map)

        # Map field bounding boxes back to full image coordinates
        for field_name, field_res in fields_1.items():
            if field_res.bounding_box:
                field_res.bounding_box['x'] += p1_x_offset
                field_res.bounding_box['y'] += p1_y_offset

        for field_name, field_res in fields_2.items():
            if field_res.bounding_box:
                field_res.bounding_box['x'] += p2_x_offset
                field_res.bounding_box['y'] += p2_y_offset

        # Merge extracted fields prioritizing Page 1 for identity/MRZ fields
        identity_fields = ["passport_number", "name", "nationality", "dob", "expiry", "sex", "mrz_line1", "mrz_line2"]
        merged_fields = {}
        all_keys = set(fields_1.keys()).union(fields_2.keys())

        for field in all_keys:
            val1 = fields_1.get(field)
            val2 = fields_2.get(field)

            if field in identity_fields:
                if val1 and val1.value != "NOT_FOUND":
                    merged_fields[field] = val1
                elif val2 and val2.value != "NOT_FOUND":
                    merged_fields[field] = val2
                else:
                    merged_fields[field] = val1 or val2
            else:
                # Secondary fields (e.g. place_of_birth, place_of_issue)
                # Prioritize Page 2 if present, otherwise Page 1
                if val2 and val2.value != "NOT_FOUND":
                    merged_fields[field] = val2
                elif val1 and val1.value != "NOT_FOUND":
                    merged_fields[field] = val1
                else:
                    merged_fields[field] = val2 or val1

        # Map word maps back to full image and combine them
        word_map_1_mapped = []
        for w in page_1_word_map:
            w_copy = w.copy()
            w_copy['left'] += p1_x_offset
            w_copy['top'] += p1_y_offset
            word_map_1_mapped.append(w_copy)

        word_map_2_mapped = []
        for w in page_2_word_map:
            w_copy = w.copy()
            w_copy['left'] += p2_x_offset
            w_copy['top'] += p2_y_offset
            word_map_2_mapped.append(w_copy)

        # Update packet contents
        packet.extracted_fields = merged_fields
        packet.ocr_raw_text = page_1_raw_text + "\n" + page_2_raw_text
        packet.ocr_word_map = word_map_1_mapped + word_map_2_mapped
        packet.ocr_confidence = (conf_A + conf_B) / 2.0

        # Store metadata
        packet.pipeline_metadata["split_occurred"] = True
        packet.pipeline_metadata["split_type"] = split_type
        packet.pipeline_metadata["split_coordinate"] = split_coord
        packet.pipeline_metadata["page_1_offset"] = (p1_x_offset, p1_y_offset)
        packet.pipeline_metadata["page_2_offset"] = (p2_x_offset, p2_y_offset)

        logger.info("Passport fields merged and packet updated successfully.")
        return True
