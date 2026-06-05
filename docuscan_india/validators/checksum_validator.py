import re
from typing import List
from utils.document_packet import ValidationResult

# Verhoeff tables
VERHOEFF_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
]

VERHOEFF_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8]
]

class ChecksumValidator:
    @staticmethod
    def validate_verhoeff(num_str: str) -> bool:
        """Verifies Aadhaar 12-digit number using the Verhoeff algorithm."""
        num_str = re.sub(r"\s+", "", num_str)
        if not num_str.isdigit() or len(num_str) != 12:
            return False

        c = 0
        for i, item in enumerate(reversed(num_str)):
            c = VERHOEFF_D[c][VERHOEFF_P[i % 8][int(item)]]
        return c == 0

    @staticmethod
    def _mrz_char_value(c: str) -> int:
        if c == '<':
            return 0
        if c.isdigit():
            return int(c)
        if c.isalpha():
            return ord(c.upper()) - ord('A') + 10
        return 0

    @classmethod
    def validate_mrz_checksum(cls, data: str, expected_check_digit: str) -> bool:
        """Verifies a segment of MRZ text against its trailing check digit."""
        if not expected_check_digit.isdigit():
            return False
        weights = [7, 3, 1]
        total = 0
        for i, char in enumerate(data):
            total += cls._mrz_char_value(char) * weights[i % 3]
        return (total % 10) == int(expected_check_digit)

    @classmethod
    def validate_passport_mrz(cls, m1: str, m2: str) -> List[ValidationResult]:
        """
        Validates check digits on Passport MRZ Line 2.
        Line 2 layout (44 chars):
        - 0-9: Passport Number (9 chars) + check digit (1 char)
        - 10-13: Nationality (3 chars)
        - 13-19: Date of Birth (6 chars, YYMMDD) + check digit (1 char)
        - 19: Sex (1 char)
        - 21-27: Expiry (6 chars, YYMMDD) + check digit (1 char)
        - 27-42: Personal number/optional (14 chars) + check digit (1 char)
        - 42-44: Composite check digit (1 char)
        """
        results = []
        if len(m2) != 44:
            results.append(ValidationResult("FAIL", "mrz_layout", "44 characters", f"{len(m2)} characters"))
            return results

        # 1. Passport number check digit (char index 9)
        pass_num = m2[0:9]
        pass_check = m2[9]
        pass_pass = cls.validate_mrz_checksum(pass_num, pass_check)
        results.append(ValidationResult(
            "PASS" if pass_pass else "FAIL",
            "mrz_passport_checksum",
            f"Check digit {pass_check} matches",
            f"Passport: {pass_num}, Check digit: {pass_check}"
        ))

        # 2. DOB check digit (char index 19)
        dob = m2[13:19]
        dob_check = m2[19]
        dob_pass = cls.validate_mrz_checksum(dob, dob_check)
        results.append(ValidationResult(
            "PASS" if dob_pass else "FAIL",
            "mrz_dob_checksum",
            f"Check digit {dob_check} matches",
            f"DOB: {dob}, Check digit: {dob_check}"
        ))

        # 3. Expiry check digit (char index 27)
        expiry = m2[21:27]
        expiry_check = m2[27]
        expiry_pass = cls.validate_mrz_checksum(expiry, expiry_check)
        results.append(ValidationResult(
            "PASS" if expiry_pass else "FAIL",
            "mrz_expiry_checksum",
            f"Check digit {expiry_check} matches",
            f"Expiry: {expiry}, Check digit: {expiry_check}"
        ))

        # 4. Composite check digit (char index 43)
        # Represents check over passport number, DOB, expiry, and optional data
        composite_data = m2[0:10] + m2[13:20] + m2[21:43]
        composite_check = m2[43]
        composite_pass = cls.validate_mrz_checksum(composite_data, composite_check)
        results.append(ValidationResult(
            "PASS" if composite_pass else "FAIL",
            "mrz_composite_checksum",
            f"Composite check digit {composite_check} matches",
            f"Composite data: {composite_data[:15]}..., Check: {composite_check}"
        ))

        return results
