import os
import sys

# Ensure the root directory of the project is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ocr.pipeline import VerificationPipeline
from utils.logger import get_logger

logger = get_logger("scratch_test_pp")

def run():
    img_path = "D:/test doc/indianpp2.jpg"
    if not os.path.exists(img_path):
        print(f"Error: Image {img_path} not found.")
        return
        
    print(f"Processing: {img_path}...")
    pipeline = VerificationPipeline()
    packet = pipeline.process_classification(img_path)
    print("Classification Done.")
    print(f"Classified Doc Type: {packet.document_type}")
    
    packet = pipeline.process_verification(packet)
    print("Verification Done.")
    print("\n--- Extracted Fields ---")
    for k, v in packet.extracted_fields.items():
        print(f"{k}: {v.value} (conf: {v.confidence})")

if __name__ == "__main__":
    run()
