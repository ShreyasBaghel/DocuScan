from typing import Tuple, List, Dict, Any
from classifiers.base_classifier import BaseClassifier
from utils.document_packet import DocumentType

class MLClassifierStub(BaseClassifier):
    def classify(self, raw_text: str, word_map: List[Dict[str, Any]]) -> Tuple[DocumentType, float]:
        """
        Stub representing a future machine learning layout classifier (e.g., PyTorch or ONNX-based).
        Currently returns UNKNOWN with 0.0 confidence as a placeholder.
        """
        # Phase 2/3 Placeholder:
        # model = load_onnx_model("models/layout_net.onnx")
        # pred = model.predict(word_map)
        return DocumentType.UNKNOWN, 0.0
