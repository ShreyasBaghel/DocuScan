from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional
import numpy as np

class DocumentType(Enum):
    AADHAAR = "AADHAAR"
    PAN = "PAN"
    PASSPORT = "PASSPORT"
    DRIVING_LICENCE = "DRIVING_LICENCE"
    UNKNOWN = "UNKNOWN"

@dataclass
class FieldResult:
    value: str
    raw_text: str
    confidence: float
    bounding_box: Optional[Dict[str, int]] = None  # e.g., {'x': 0, 'y': 0, 'w': 0, 'h': 0}
    constituent_boxes: Optional[List[Dict[str, Any]]] = None
    status: str = "ok"

    @property
    def word_boxes(self) -> Optional[List[Dict[str, Any]]]:
        return self.constituent_boxes

    @word_boxes.setter
    def word_boxes(self, value: Optional[List[Dict[str, Any]]]):
        self.constituent_boxes = value

    @property
    def combined_bbox(self) -> Optional[Dict[str, int]]:
        return self.bounding_box

    @combined_bbox.setter
    def combined_bbox(self, value: Optional[Dict[str, int]]):
        self.bounding_box = value



@dataclass
class ValidationResult:
    status: str  # PASS, WARN, FAIL
    field_name: str
    expected: str
    actual: str
    score: int = 0  # Dynamic model-driven risk contribution


@dataclass
class FraudSignal:
    name: str
    score: int  # risk score contribution
    description: str
    source: str  # e.g., "MetadataAnalyser", "ChecksumValidator", etc.

@dataclass
class DocumentPacket:
    image_path: str
    raw_image: Optional[np.ndarray] = None
    preprocessed_image: Optional[np.ndarray] = None
    ocr_raw_text: str = ""
    ocr_confidence: float = 0.0
    ocr_word_map: List[Dict[str, Any]] = field(default_factory=list)
    document_type: DocumentType = DocumentType.UNKNOWN
    classification_confidence: float = 0.0
    extracted_fields: Dict[str, FieldResult] = field(default_factory=dict)
    validation_results: List[ValidationResult] = field(default_factory=list)
    fraud_signals: List[FraudSignal] = field(default_factory=list)
    fraud_risk_score: int = 0
    authenticity_score: int = 0
    extraction_reliability: int = 0
    final_decision: str = "Needs Manual Review"
    report_path: str = ""
    pipeline_metadata: Dict[str, Any] = field(default_factory=dict)

