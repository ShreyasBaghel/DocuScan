import os
import yaml
from utils.document_packet import DocumentPacket, DocumentType, ValidationResult
from ocr.image_loader import ImageLoader
from ocr.preprocessor import Preprocessor
from ocr.ocr_engine import OCREngine
from classifiers.classifier_ensemble import ClassifierEnsemble
from extractors.extractor_registry import ExtractorRegistry
from validators.format_validator import FormatValidator
from validators.checksum_validator import ChecksumValidator
from validators.date_validator import DateValidator
from validators.cross_field_validator import CrossFieldValidator
from fraud_detection.metadata_analyser import MetadataAnalyser
from fraud_detection.format_consistency_checker import FormatConsistencyChecker
from fraud_detection.risk_score_engine import RiskScoreEngine
from reports.export_manager import ExportManager
from utils.logger import get_logger

logger = get_logger("pipeline")

class VerificationPipeline:
    def __init__(self, config_path: str = None):
        """Initializes all pipeline stages with configuration."""
        if config_path is None:
            # Default to config.yaml in application directory
            app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(app_dir, "config.yaml")

        self.config = {}
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    self.config = yaml.safe_load(f) or {}
                logger.info(f"Pipeline loaded configuration: {config_path}")
            except Exception as e:
                logger.error(f"Failed to parse config: {e}")

        # Initialize engines
        self.preprocessor = Preprocessor(self.config.get("preprocessing", {}))
        self.ocr_engine = OCREngine(self.config)
        self.classifier = ClassifierEnsemble(self.config)
        self.metadata_analyser = MetadataAnalyser()
        self.format_checker = FormatConsistencyChecker()
        self.risk_engine = RiskScoreEngine(self.config)
        self.export_manager = ExportManager()

    def process_classification(self, image_path: str) -> DocumentPacket:
        """
        Executes Stages 1-3: Load image, Preprocess, OCR, and Classify.
        Returns the partially-populated DocumentPacket.
        """
        logger.info(f"--- Pipeline starting classification for {image_path} ---")
        
        # Initialize empty packet
        packet = DocumentPacket(image_path=image_path)
        
        # Stage 1: Load and Preprocess Image
        try:
            packet.raw_image = ImageLoader.load(image_path)
            
            # If it's a PDF, bypass destructive preprocessing steps (denoise, clahe, binarise)
            # to preserve clean, high-resolution digital text.
            is_pdf = image_path.lower().endswith('.pdf')
            if is_pdf:
                logger.info("PDF document detected. Bypassing denoise, clahe, and binarise to preserve digital text quality.")
                old_config = self.preprocessor.config
                temp_config = old_config.copy()
                temp_config['denoise'] = False
                temp_config['clahe'] = False
                temp_config['binarise'] = False
                
                self.preprocessor.config = temp_config
                packet.preprocessed_image = self.preprocessor.preprocess(packet.raw_image)
                self.preprocessor.config = old_config
            else:
                packet.preprocessed_image = self.preprocessor.preprocess(packet.raw_image)
                
            logger.info("Stage 1 (Preprocessing) completed successfully.")
        except Exception as e:
            logger.error(f"Stage 1 failed: {e}")
            raise e

        # Stage 2: OCR Extraction
        try:
            # Run OCR initially on raw text for classification
            raw_text, ocr_conf, word_map = self.ocr_engine.extract(packet.preprocessed_image)
            packet.ocr_raw_text = raw_text
            packet.ocr_confidence = ocr_conf
            packet.ocr_word_map = word_map
            logger.info(f"Stage 2 (OCR) completed. Confidence: {ocr_conf:.2f}")
        except Exception as e:
            logger.error(f"Stage 2 failed: {e}")
            raise e

        # Stage 3: Document Classification
        try:
            doc_type, class_conf = self.classifier.classify(packet.ocr_raw_text, packet.ocr_word_map)
            packet.document_type = doc_type
            packet.classification_confidence = class_conf
            logger.info(f"Stage 3 (Classification) completed. Type: {doc_type.value}, Conf: {class_conf:.2f}")
            
            # Run extraction early if classification is successful and known
            if doc_type != DocumentType.UNKNOWN:
                try:
                    extractor = ExtractorRegistry.get_extractor(doc_type)
                    packet.extracted_fields = extractor.extract(packet.ocr_raw_text, packet.ocr_word_map)
                    logger.info(f"Ran early extraction for classification display. Fields: {list(packet.extracted_fields.keys())}")
                except Exception as e:
                    logger.error(f"Early extraction failed: {e}")
        except Exception as e:
            logger.error(f"Stage 3 failed: {e}")
            raise e

        return packet

    def process_verification(self, packet: DocumentPacket) -> DocumentPacket:
        """
        Executes Stages 4-7: Extraction, Validation, Fraud Detection, and Export.
        Runs after the document type is confirmed (either automatically or manually).
        """
        if packet.document_type == DocumentType.UNKNOWN:
            raise ValueError("Cannot verify document with UNKNOWN classification.")

        logger.info(f"--- Pipeline continuing verification for {packet.document_type.value} ---")

        # Stage 4: Field Extraction
        try:
            extractor = ExtractorRegistry.get_extractor(packet.document_type)
            packet.extracted_fields = extractor.extract(packet.ocr_raw_text, packet.ocr_word_map)
            
            # Fallback OCR pass for Aadhaar or PAN if the primary ID number was not found
            # (Very useful for low-res photos/scans where PSM 3 misses the separate number block)
            if packet.document_type == DocumentType.AADHAAR:
                num_field = packet.extracted_fields.get("aadhaar_number")
                if not num_field or num_field.value == "NOT_FOUND":
                    logger.info("Aadhaar number not found in first pass. Running fallback OCR (PSM 11)...")
                    raw_fb, conf_fb, map_fb = self.ocr_engine.extract(packet.preprocessed_image, "AADHAAR_FALLBACK")
                    fb_fields = extractor.extract(raw_fb, map_fb)
                    fb_num = fb_fields.get("aadhaar_number")
                    if fb_num and fb_num.value != "NOT_FOUND":
                        logger.info(f"Successfully recovered Aadhaar number in fallback pass: {fb_num.value}")
                        packet.extracted_fields["aadhaar_number"] = fb_num
                        # Also recover other NOT_FOUND fields if possible
                        for k, v in fb_fields.items():
                            if k not in packet.extracted_fields or packet.extracted_fields[k].value == "NOT_FOUND":
                                packet.extracted_fields[k] = v
                                
            elif packet.document_type == DocumentType.PAN:
                num_field = packet.extracted_fields.get("pan_number")
                if not num_field or num_field.value == "NOT_FOUND":
                    logger.info("PAN number not found in first pass. Running fallback OCR (PSM 11)...")
                    raw_fb, conf_fb, map_fb = self.ocr_engine.extract(packet.preprocessed_image, "PAN_FALLBACK")
                    fb_fields = extractor.extract(raw_fb, map_fb)
                    fb_num = fb_fields.get("pan_number")
                    if fb_num and fb_num.value != "NOT_FOUND":
                        logger.info(f"Successfully recovered PAN number in fallback pass: {fb_num.value}")
                        packet.extracted_fields["pan_number"] = fb_num
                        for k, v in fb_fields.items():
                            if k not in packet.extracted_fields or packet.extracted_fields[k].value == "NOT_FOUND":
                                packet.extracted_fields[k] = v

            logger.info(f"Stage 4 (Extraction) completed. Fields: {list(packet.extracted_fields.keys())}")
        except Exception as e:
            logger.error(f"Stage 4 failed: {e}")
            raise e

        # Stage 5: Validation
        try:
            packet.validation_results = []
            
            # Format validation
            packet.validation_results.extend(FormatValidator.validate(packet.document_type, packet.extracted_fields))
            
            # Date range validation
            packet.validation_results.extend(DateValidator.validate(packet.document_type, packet.extracted_fields))
            
            # Cross field validation
            packet.validation_results.extend(CrossFieldValidator.validate(packet.document_type, packet.extracted_fields))
            
            # Checksum validation (Verhoeff and Passport MRZ check digits)
            if packet.document_type == DocumentType.AADHAAR:
                aadhaar_num = packet.extracted_fields.get("aadhaar_number")
                if aadhaar_num and aadhaar_num.value != "NOT_FOUND":
                    valid = ChecksumValidator.validate_verhoeff(aadhaar_num.value)
                    packet.validation_results.append(ValidationResult(
                        status="PASS" if valid else "FAIL",
                        field_name="aadhaar_checksum_verhoeff",
                        expected="Verhoeff check digit matches",
                        actual=f"Aadhaar Number: {aadhaar_num.value} (Valid: {valid})"
                    ))
            elif packet.document_type == DocumentType.PASSPORT:
                m1 = packet.extracted_fields.get("mrz_line1")
                m2 = packet.extracted_fields.get("mrz_line2")
                if m1 and m2 and m1.value != "NOT_FOUND" and m2.value != "NOT_FOUND":
                    packet.validation_results.extend(ChecksumValidator.validate_passport_mrz(m1.value, m2.value))

            logger.info(f"Stage 5 (Validation) completed. Total check rules: {len(packet.validation_results)}")
        except Exception as e:
            logger.error(f"Stage 5 failed: {e}")
            raise e

        # Stage 6: Fraud Detection
        try:
            # Metadata analysis
            packet.fraud_signals = self.metadata_analyser.analyse(packet.image_path)
            
            # Formatting consistency
            packet.fraud_signals.extend(self.format_checker.check(packet.document_type, packet.extracted_fields, packet.ocr_word_map))
            
            # Compute fraud risk score
            packet.fraud_risk_score = self.risk_engine.calculate(
                packet.ocr_confidence,
                packet.validation_results,
                packet.fraud_signals
            )
            logger.info(f"Stage 6 (Fraud Detection) completed. Risk Score: {packet.fraud_risk_score}")
        except Exception as e:
            logger.error(f"Stage 6 failed: {e}")
            raise e

        # Stage 7: Reporting
        try:
            pdf_path = self.export_manager.export_all(packet)
            packet.report_path = pdf_path
            logger.info("Stage 7 (Reporting) completed. Exports generated.")
        except Exception as e:
            logger.error(f"Stage 7 failed: {e}")
            raise e

        return packet
