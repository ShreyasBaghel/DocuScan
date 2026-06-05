import pytest
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.document_packet import DocumentType, FieldResult
from validators.checksum_validator import ChecksumValidator, VERHOEFF_D, VERHOEFF_P
from validators.format_validator import FormatValidator
from validators.date_validator import DateValidator
from validators.cross_field_validator import CrossFieldValidator

def test_verhoeff_algorithm():
    # Helper to generate a valid Verhoeff check digit for a 11-digit number
    def generate_verhoeff_check_digit(num_str: str) -> int:
        c = 0
        for i, item in enumerate(reversed(num_str)):
            # Permutation table index is i + 1 because the check digit will occupy index 0
            c = VERHOEFF_D[c][VERHOEFF_P[(i + 1) % 8][int(item)]]
        
        # Find check digit that satisfies: VERHOEFF_D[c][VERHOEFF_P[0][check_digit]] == 0
        # Since VERHOEFF_P[0] is identity (0,1,2,3,4,5,6,7,8,9), we want VERHOEFF_D[c][check_digit] == 0
        # Looking at VERHOEFF_D table, each row has a single 0. We find column j where VERHOEFF_D[c][j] == 0.
        # But VERHOEFF_D is symmetric, and we can find the inverse from inv table or scan the row:
        check_digit = 0
        for j in range(10):
            if VERHOEFF_D[c][j] == 0:
                check_digit = j
                break
        return check_digit

    # Generate a valid 12-digit Aadhaar number dynamically
    base = "36687483021"
    check = generate_verhoeff_check_digit(base)
    valid_aadhaar = f"{base}{check}"

    # Verify that it passes
    assert ChecksumValidator.validate_verhoeff(valid_aadhaar) is True

    # Mutate a digit and verify it fails
    invalid_aadhaar = f"{base[:-1]}0{check}"
    assert ChecksumValidator.validate_verhoeff(invalid_aadhaar) is False

    # Check known invalid format/length
    assert ChecksumValidator.validate_verhoeff("123") is False
    assert ChecksumValidator.validate_verhoeff("12345678901A") is False

def test_passport_mrz_checksum():
    # Let's test standard passport number: Z1234567
    # Z(35)*7 + 1*3 + 2*1 + 3*7 + 4*3 + 5*1 + 6*7 + 7*3 = 245 + 3 + 2 + 21 + 12 + 5 + 42 + 21 = 351
    # 351 % 10 = 1
    assert ChecksumValidator.validate_mrz_checksum("Z1234567", "1") is True
    assert ChecksumValidator.validate_mrz_checksum("Z1234567", "2") is False

def test_format_validator():
    # Valid Aadhaar packet
    fields = {
        "aadhaar_number": FieldResult("123456789012", "", 0.90),
        "dob": FieldResult("1990-08-15", "", 0.90),
        "gender": FieldResult("MALE", "", 0.90),
        "name": FieldResult("SHREYANSH KUMAR", "", 0.90)
    }
    results = FormatValidator.validate(DocumentType.AADHAAR, fields)
    for r in results:
        assert r.status in ["PASS", "WARN"]

    # Invalid PAN packet
    fields_invalid_pan = {
        "pan_number": FieldResult("1234567890", "", 0.90),  # must be 5L, 4D, 1L
        "dob": FieldResult("1990/08/15", "", 0.90),       # must be YYYY-MM-DD
        "name": FieldResult("SHREYANSH123", "", 0.90),     # must be alphabetic
        "father_name": FieldResult("LALIT KUMAR", "", 0.90)
    }
    results_pan = FormatValidator.validate(DocumentType.PAN, fields_invalid_pan)
    statuses = [r.status for r in results_pan]
    assert "FAIL" in statuses or "WARN" in statuses

def test_date_validator():
    # DOB in the future
    fields = {"dob": FieldResult("2050-01-01", "", 0.90)}
    results = DateValidator.validate(DocumentType.AADHAAR, fields)
    assert any(r.status == "FAIL" and r.field_name == "dob_range" for r in results)

    # Driving licence minor check
    fields_minor = {
        "dob": FieldResult("2018-01-01", "", 0.90),  # Under 18
        "validity": FieldResult("2035-01-01", "", 0.90)
    }
    results_dl = DateValidator.validate(DocumentType.DRIVING_LICENCE, fields_minor)
    assert any(r.status == "FAIL" and r.field_name == "dl_min_age" for r in results_dl)

def test_cross_field_validator():
    # PAN vs Surname check
    # Surname is SHARMA (starts with S), PAN 5th char is S. This is valid.
    fields_pan_valid = {
        "name": FieldResult("RAJESH SHARMA", "", 0.90),
        "pan_number": FieldResult("ABCDS1234S", "", 0.90)
    }
    results = CrossFieldValidator.validate(DocumentType.PAN, fields_pan_valid)
    assert any(r.status == "PASS" and r.field_name == "pan_name_consistency" for r in results)

    # PAN vs Surname mismatch
    fields_pan_invalid = {
        "name": FieldResult("RAJESH SHARMA", "", 0.90),
        "pan_number": FieldResult("ABCDE1234P", "", 0.90)  # P doesn't match S
    }
    results_invalid = CrossFieldValidator.validate(DocumentType.PAN, fields_pan_invalid)
    assert any(r.status == "FAIL" and r.field_name == "pan_name_consistency" for r in results_invalid)
