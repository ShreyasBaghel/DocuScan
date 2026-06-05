import exifread
from typing import List
from utils.document_packet import FraudSignal
from utils.logger import get_logger

logger = get_logger("metadata_analyser")

class MetadataAnalyser:
    def analyse(self, file_path: str) -> List[FraudSignal]:
        signals: List[FraudSignal] = []

        if not file_path:
            return signals

        try:
            with open(file_path, 'rb') as f:
                tags = exifread.process_file(f, details=False)

            if not tags:
                logger.info(f"No EXIF metadata found in: {file_path}")
                return signals

            # Editing software signatures to look for
            editing_software = ["photoshop", "gimp", "adobe", "canva", "corel", "paint.net", "pixelmator"]

            for tag_name, tag_value in tags.items():
                tag_str = str(tag_value).lower()
                
                # Check Software or ImageHistory tags
                if any(k in tag_name.lower() for k in ["software", "imagehistory", "creator", "editor"]):
                    for software in editing_software:
                        if software in tag_str:
                            signals.append(FraudSignal(
                                name="editing_software_detected",
                                score=30,
                                description=f"EXIF tag '{tag_name}' lists editing software: '{tag_value}'",
                                source="MetadataAnalyser"
                            ))
                            break

                # Check timestamp anomalies (e.g. modified date vs original date)
                # DateTimeOriginal vs DateTime
                if "DateTimeOriginal" in tags and "DateTime" in tags:
                    orig_time = str(tags["DateTimeOriginal"])
                    mod_time = str(tags["DateTime"])
                    if orig_time != mod_time:
                        signals.append(FraudSignal(
                            name="timestamp_mismatch",
                            score=15,
                            description=f"Original timestamp ({orig_time}) differs from modification timestamp ({mod_time})",
                            source="MetadataAnalyser"
                        ))

        except Exception as e:
            logger.error(f"Failed to analyze EXIF metadata: {e}")

        return signals
