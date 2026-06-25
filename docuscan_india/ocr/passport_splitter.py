import re
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import pytesseract
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

    @staticmethod
    def _find_widest_gap(profile: np.ndarray, start_idx: int, end_idx: int, threshold: float) -> Tuple[Optional[int], int]:
        """
        Finds the center and width of the widest contiguous block in profile[start_idx:end_idx]
        where the profile values are <= threshold.
        """
        widest_center = None
        max_width = 0
        
        current_start = None
        for i in range(start_idx, end_idx):
            if profile[i] <= threshold:
                if current_start is None:
                    current_start = i
            else:
                if current_start is not None:
                    width = i - current_start
                    if width > max_width:
                        max_width = width
                        widest_center = (current_start + i - 1) // 2
                    current_start = None
        
        if current_start is not None:
            width = end_idx - current_start
            if width > max_width:
                max_width = width
                widest_center = (current_start + end_idx - 1) // 2
                
        return widest_center, max_width

    @classmethod
    def detect_hybrid_split(cls, img: Image.Image, ocr_engine=None, word_map: List[Dict[str, Any]] = None) -> Tuple[Optional[str], Optional[int]]:
        """
        Performs hybrid layout analysis using text density and OCR word bounding box clustering.
        Returns:
            - split_type: "vertical", "horizontal", or None
            - split_coord: coordinate index of the split, or None
        """
        W, H = img.size
        
        # --- Step 1: Convert to grayscale and density analysis ---
        gray = img.convert("L")
        arr = np.array(gray)
        # Assuming text is darker than background (usually binarised 0/255)
        binary = arr < 127
        
        col_density = np.sum(binary, axis=0)
        row_density = np.sum(binary, axis=1)
        
        avg_col_density = np.mean(col_density)
        avg_row_density = np.mean(row_density)
        
        col_threshold = max(2, int(0.02 * avg_col_density)) if avg_col_density > 0 else 2
        row_threshold = max(2, int(0.02 * avg_row_density)) if avg_row_density > 0 else 2
        
        # Density-based vertical candidate in [0.35 * W, 0.65 * W]
        x_density, x_gap_width = cls._find_widest_gap(col_density, int(0.35 * W), int(0.65 * W), col_threshold)
        
        # Density-based horizontal candidate in [0.35 * H, 0.85 * H]
        y_density, y_gap_height = cls._find_widest_gap(row_density, int(0.35 * H), int(0.85 * H), row_threshold)
        
        # --- Step 2: Lightweight OCR metadata extraction ---
        if word_map is not None:
            words = word_map
            logger.info(f"Reusing provided word map with {len(words)} words.")
        else:
            lang = "eng"
            psm_mode = 3
            if ocr_engine and ocr_engine.config:
                tess_config = ocr_engine.config.get("tesseract", {})
                lang = tess_config.get("default_lang", "eng")
                psm_map = ocr_engine.config.get("ocr_psm", {})
                psm_mode = psm_map.get("PASSPORT", 3)
                
            config_str = f"--psm {psm_mode}"
            
            try:
                import pytesseract
                data_dict = pytesseract.image_to_data(img, lang=lang, config=config_str, output_type=pytesseract.Output.DICT)
            except Exception as e:
                logger.error(f"Lightweight image_to_data failed: {e}")
                return None, None
                
            words = []
            n_boxes = len(data_dict['level']) if 'level' in data_dict else 0
            for i in range(n_boxes):
                level = data_dict['level'][i]
                text = data_dict['text'][i].strip() if 'text' in data_dict else ""
                if level == 5 and text:
                    words.append({
                        'left': int(data_dict['left'][i]),
                        'top': int(data_dict['top'][i]),
                        'width': int(data_dict['width'][i]),
                        'height': int(data_dict['height'][i]),
                    })
                
        # Safety Check: OCR metadata is insufficient
        if len(words) < 2:
            logger.info("OCR metadata is insufficient (fewer than 2 words). Fallback to standard OCR.")
            return None, None
            
        # --- Step 3: Cluster OCR text regions ---
        # Find best vertical split (x_ocr)
        min_x_cost = float('inf')
        x_ocr = None
        start_x = int(0.35 * W)
        end_x = int(0.65 * W)
        
        # Step through x coordinates
        x_coords_search = list(range(start_x, end_x, 2))
        x_costs = []
        for x in x_coords_search:
            cost = sum(1 for w in words if w['left'] < x < w['left'] + w['width'])
            x_costs.append((x, cost))
            
        if x_costs:
            min_x_cost = min(cost for x, cost in x_costs)
            # Find center of widest contiguous x with min_x_cost
            best_x_ocr = None
            max_x_gap_width = 0
            current_x_start = None
            for x, cost in x_costs:
                if cost == min_x_cost:
                    if current_x_start is None:
                        current_x_start = x
                else:
                    if current_x_start is not None:
                        width = x - current_x_start
                        if width > max_x_gap_width:
                            max_x_gap_width = width
                            best_x_ocr = (current_x_start + x - 2) // 2
                        current_x_start = None
            if current_x_start is not None:
                width = end_x - current_x_start
                if width > max_x_gap_width:
                    max_x_gap_width = width
                    best_x_ocr = (current_x_start + end_x - 2) // 2
            x_ocr = best_x_ocr
            
        # Find best horizontal split (y_ocr)
        min_y_cost = float('inf')
        y_ocr = None
        start_y = int(0.35 * H)
        end_y = int(0.85 * H)
        
        # Step through y coordinates
        y_coords_search = list(range(start_y, end_y, 2))
        y_costs = []
        for y in y_coords_search:
            cost = sum(1 for w in words if w['top'] < y < w['top'] + w['height'])
            y_costs.append((y, cost))
            
        if y_costs:
            min_y_cost = min(cost for y, cost in y_costs)
            # Find center of widest contiguous y with min_y_cost
            best_y_ocr = None
            max_y_gap_height = 0
            current_y_start = None
            for y, cost in y_costs:
                if cost == min_y_cost:
                    if current_y_start is None:
                        current_y_start = y
                else:
                    if current_y_start is not None:
                        width = y - current_y_start
                        if width > max_y_gap_height:
                            max_y_gap_height = width
                            best_y_ocr = (current_y_start + y - 2) // 2
                        current_y_start = None
            if current_y_start is not None:
                width = end_y - current_y_start
                if width > max_y_gap_height:
                    max_y_gap_height = width
                    best_y_ocr = (current_y_start + end_y - 2) // 2
            y_ocr = best_y_ocr

        # --- Step 4: Combine density and OCR split candidates ---
        max_allowed_cost = max(3, int(len(words) * 0.03))
        col_tolerance = int(0.05 * W)
        row_tolerance = int(0.05 * H)
        
        is_v_valid = False
        if x_density is not None and x_ocr is not None:
            # Check agreement within tolerance, cost constraint, significant gap width, and aspect ratio
            if (abs(x_density - x_ocr) <= col_tolerance and 
                min_x_cost <= max_allowed_cost and 
                x_gap_width >= max(10, int(0.01 * W)) and 
                W / H >= 1.15):
                is_v_valid = True
                
        is_h_valid = False
        if y_density is not None and y_ocr is not None:
            # Check agreement within tolerance, cost constraint, and significant gap height
            if (abs(y_density - y_ocr) <= row_tolerance and 
                min_y_cost <= max_allowed_cost and 
                y_gap_height >= max(10, int(0.01 * H))):
                is_h_valid = True

        if is_v_valid and is_h_valid:
            # Prioritize the one matching dominant aspect ratio
            if W >= H:
                return "vertical", x_ocr
            else:
                return "horizontal", y_ocr
        elif is_v_valid:
            return "vertical", x_ocr
        elif is_h_valid:
            return "horizontal", y_ocr
            
        logger.info("Hybrid split check did not accept any split candidate.")
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
        
        # Check if the split already occurred during Stage 2 pre-OCR preprocessing
        if packet.pipeline_metadata.get("split_occurred"):
            split_type = packet.pipeline_metadata.get("split_type")
            split_coord = packet.pipeline_metadata.get("split_coordinate")
            
            raw_text_A = packet.pipeline_metadata.get("raw_text_A")
            conf_A = packet.pipeline_metadata.get("conf_A", 0.0)
            word_map_A = packet.pipeline_metadata.get("word_map_A")
            
            raw_text_B = packet.pipeline_metadata.get("raw_text_B")
            conf_B = packet.pipeline_metadata.get("conf_B", 0.0)
            word_map_B = packet.pipeline_metadata.get("word_map_B")
            
            if split_type == "vertical":
                offset_A = (0, 0)
                offset_B = (split_coord, 0)
            else:
                offset_A = (0, 0)
                offset_B = (0, split_coord)
                
            logger.info(f"Reusing pre-OCR split results. Type: {split_type}, Coord: {split_coord}")
        else:
            split_type, split_coord = cls.detect_hybrid_split(img, ocr_engine, word_map=packet.ocr_word_map)

            if not split_type or split_coord is None:
                logger.info("No passport split detected. Treating as single-page.")
                return False

            logger.info(f"Passport dual-page layout detected via hybrid detection. Type: {split_type}, Coord: {split_coord}")

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
