import pytest
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.document_packet import DocumentType
from classifiers.keyword_classifier import KeywordClassifier
from classifiers.regex_classifier import RegexClassifier
from classifiers.layout_classifier import LayoutClassifier
from classifiers.classifier_ensemble import ClassifierEnsemble

# 1. Mock Texts for Keyword Classifier
aadhaar_text = "UNIQUE IDENTIFICATION AUTHORITY OF INDIA. Mera Aadhaar, Meri Pehchan. Enrollment No. Male. Year of Birth 1990."
pan_text = "INCOME TAX DEPARTMENT. GOVT. OF INDIA. PERMANENT ACCOUNT CARD. Father's Name: LALIT KUMAR. PAN: ABCDE1234F."
passport_text = "REPUBLIC OF INDIA. PASSPORT. Passport No. Z1234567. Nationality INDIAN. Surname SINHA. Given Name RAJ."
dl_text = "DRIVING LICENCE. UNION OF INDIA. LICENSING AUTHORITY. CLASS OF VEHICLE: LMV, MCWG. DL-0420110067890."

def test_keyword_classifier():
    clf = KeywordClassifier()
    
    doc, score = clf.classify(aadhaar_text, [])
    assert doc == DocumentType.AADHAAR
    assert score > 0.3
    
    doc, score = clf.classify(pan_text, [])
    assert doc == DocumentType.PAN
    assert score > 0.3
    
    doc, score = clf.classify(passport_text, [])
    assert doc == DocumentType.PASSPORT
    assert score > 0.3
    
    doc, score = clf.classify(dl_text, [])
    assert doc == DocumentType.DRIVING_LICENCE
    assert score > 0.3

def test_regex_classifier():
    clf = RegexClassifier()
    
    # Aadhaar format: 1234 5678 9012
    doc, score = clf.classify("Some dummy text 9999 8888 7777 inside", [])
    assert doc == DocumentType.AADHAAR
    
    # PAN format: 5 letters, 4 digits, 1 letter
    doc, score = clf.classify("Permanent Account Number is ASDPR9988C", [])
    assert doc == DocumentType.PAN
    
    # Passport format: 1 letter, 7 digits
    doc, score = clf.classify("Passport number Z1092837 is valid", [])
    assert doc == DocumentType.PASSPORT

    # DL format: State(2 chars)+RTO(2 digits)+Year(4 digits)+Index(7 digits)
    doc, score = clf.classify("DL Number is DL04-2011-0067890", [])
    assert doc == DocumentType.DRIVING_LICENCE

def test_layout_classifier():
    clf = LayoutClassifier()
    
    # Mock Passport MRZ layout: words at the bottom containing '<'
    passport_word_map = [
        {"text": "PASSPORT", "left": 100, "top": 50, "width": 50, "height": 10, "conf": 0.95},
        {"text": "P<INDRAJ<<KUMAR<<<<<<<<<<<<<<<<<<<<<<<<<<<<<", "left": 50, "top": 450, "width": 300, "height": 15, "conf": 0.90},
        {"text": "Z1234567<8IND9012312M3201015<<<<<<<<<<<<<<02", "left": 50, "top": 475, "width": 300, "height": 15, "conf": 0.90}
    ]
    # Height of document is around 500
    doc, score = clf.classify("", passport_word_map)
    assert doc == DocumentType.PASSPORT
    assert score >= 0.90

def test_classifier_ensemble():
    config = {
        "classification": {
            "confidence_threshold": 0.70,
            "weights": {"keyword": 0.40, "regex": 0.40, "layout": 0.20}
        }
    }
    ensemble = ClassifierEnsemble(config)
    
    # Test high-confidence prediction
    text = "INCOME TAX DEPARTMENT PERMANENT ACCOUNT CARD PAN: ABCDE1234F"
    doc, score = ensemble.classify(text, [])
    assert doc == DocumentType.PAN
    assert score >= 0.70

    # Test unknown lower threshold prediction
    text = "completely random garbage text with no fields"
    doc, score = ensemble.classify(text, [])
    assert doc == DocumentType.UNKNOWN
