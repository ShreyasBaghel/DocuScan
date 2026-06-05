from typing import Dict
from utils.document_packet import DocumentType
from extractors.base_extractor import BaseExtractor
from extractors.aadhaar_extractor import AadhaarExtractor
from extractors.pan_extractor import PANExtractor
from extractors.passport_extractor import PassportExtractor
from extractors.dl_extractor import DLExtractor

class ExtractorRegistry:
    # Pre-register extractor instances
    _registry: Dict[DocumentType, BaseExtractor] = {}

    @classmethod
    def initialize(cls):
        """Registers all document extractors."""
        cls._registry = {
            DocumentType.AADHAAR: AadhaarExtractor(),
            DocumentType.PAN: PANExtractor(),
            DocumentType.PASSPORT: PassportExtractor(),
            DocumentType.DRIVING_LICENCE: DLExtractor()
        }

    @classmethod
    def get_extractor(cls, doc_type: DocumentType) -> BaseExtractor:
        """
        Retrieves the registered extractor for a given DocumentType.
        Raises ValueError if type is unsupported or UNKNOWN.
        """
        if not cls._registry:
            cls.initialize()

        if doc_type not in cls._registry:
            raise ValueError(f"No extractor registered for document type: {doc_type}")

        return cls._registry[doc_type]
