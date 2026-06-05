import json
from dataclasses import asdict
from typing import Dict, Any
from utils.document_packet import DocumentPacket, DocumentType

class DocumentPacketEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, DocumentType):
            return obj.value
        # If it's a numpy array, ignore or convert to list (we skip binary fields)
        return super().default(obj)

class JsonSerializer:
    @staticmethod
    def serialize(packet: DocumentPacket) -> str:
        """Converts DocumentPacket into a structured JSON string, omitting raw images."""
        data = {
            "image_path": packet.image_path,
            "ocr_raw_text": packet.ocr_raw_text,
            "ocr_confidence": packet.ocr_confidence,
            "ocr_word_map": packet.ocr_word_map,
            "document_type": packet.document_type.value,
            "classification_confidence": packet.classification_confidence,
            "extracted_fields": {
                k: {
                    "value": v.value,
                    "raw_text": v.raw_text,
                    "confidence": v.confidence,
                    "bounding_box": v.bounding_box
                } for k, v in packet.extracted_fields.items()
            },
            "validation_results": [
                {
                    "status": r.status,
                    "field_name": r.field_name,
                    "expected": r.expected,
                    "actual": r.actual
                } for r in packet.validation_results
            ],
            "fraud_signals": [
                {
                    "name": s.name,
                    "score": s.score,
                    "description": s.description,
                    "source": s.source
                } for s in packet.fraud_signals
            ],
            "fraud_risk_score": packet.fraud_risk_score,
            "report_path": packet.report_path,
            "pipeline_metadata": packet.pipeline_metadata
        }

        return json.dumps(data, indent=4, cls=DocumentPacketEncoder)
