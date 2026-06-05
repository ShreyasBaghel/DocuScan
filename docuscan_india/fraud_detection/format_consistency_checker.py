from typing import List, Dict
from utils.document_packet import FraudSignal, FieldResult, DocumentType
from utils.logger import get_logger

logger = get_logger("format_consistency_checker")

class FormatConsistencyChecker:
    def check(self, doc_type: DocumentType, fields: Dict[str, FieldResult], word_map: List[Dict]) -> List[FraudSignal]:
        signals: List[FraudSignal] = []

        if not fields or not word_map:
            return signals

        # 1. Determine bounding dimensions from word map
        max_y = max(w['top'] + w['height'] for w in word_map)
        if max_y == 0:
            return signals

        # 2. Check position anomalies
        # Passport MRZ must be in the lower part of the document
        if doc_type == DocumentType.PASSPORT:
            mrz1 = fields.get("mrz_line1")
            mrz2 = fields.get("mrz_line2")
            for mrz_f, label in [(mrz1, "mrz_line1"), (mrz2, "mrz_line2")]:
                if mrz_f and mrz_f.bounding_box:
                    y_center = mrz_f.bounding_box['y'] + (mrz_f.bounding_box['h'] / 2)
                    # If MRZ center is in the top 50% of the page
                    if y_center < 0.50 * max_y:
                        signals.append(FraudSignal(
                            name="layout_anomaly_mrz_position",
                            score=25,
                            description=f"Passport {label} detected in the upper half of the page (Center Y: {y_center:.1f} / Max Y: {max_y})",
                            source="FormatConsistencyChecker"
                        ))

        # Aadhaar number should not be at the very top (usually lower part of the front side)
        if doc_type == DocumentType.AADHAAR:
            aadhaar_f = fields.get("aadhaar_number")
            if aadhaar_f and aadhaar_f.bounding_box:
                y_center = aadhaar_f.bounding_box['y'] + (aadhaar_f.bounding_box['h'] / 2)
                # If Aadhaar number is at the very top 20% of card
                if y_center < 0.20 * max_y:
                    signals.append(FraudSignal(
                        name="layout_anomaly_aadhaar_position",
                        score=20,
                        description=f"Aadhaar number detected at the very top of the card (Center Y: {y_center:.1f} / Max Y: {max_y})",
                        source="FormatConsistencyChecker"
                    ))

        # 3. Check for overlapping bounding boxes of key fields
        # (Overlapping text boxes are a common artifact of poorly edited fake templates)
        boxes_to_check = []
        for name, field in fields.items():
            if field and field.bounding_box and field.value != "NOT_FOUND":
                boxes_to_check.append((name, field.bounding_box))

        for i in range(len(boxes_to_check)):
            for j in range(i + 1, len(boxes_to_check)):
                n1, b1 = boxes_to_check[i]
                n2, b2 = boxes_to_check[j]

                # Check intersection
                x_overlap = max(0, min(b1['x'] + b1['w'], b2['x'] + b2['w']) - max(b1['x'], b2['x']))
                y_overlap = max(0, min(b1['y'] + b1['h'], b2['y'] + b2['h']) - max(b1['y'], b2['y']))
                
                # If there's substantial intersection (e.g. overlap area is > 30% of either box area)
                if x_overlap > 0 and y_overlap > 0:
                    overlap_area = x_overlap * y_overlap
                    a1 = b1['w'] * b1['h']
                    a2 = b2['w'] * b2['h']
                    
                    if a1 > 0 and a2 > 0:
                        pct1 = overlap_area / a1
                        pct2 = overlap_area / a2
                        if pct1 > 0.30 or pct2 > 0.30:
                            signals.append(FraudSignal(
                                name="overlapping_fields_detected",
                                score=25,
                                description=f"Extracted fields '{n1}' and '{n2}' overlap by {max(pct1, pct2)*100:.1f}%. Possible card forgery.",
                                source="FormatConsistencyChecker"
                            ))

        return signals
