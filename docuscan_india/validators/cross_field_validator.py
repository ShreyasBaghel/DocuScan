import re
from typing import List, Dict
from utils.document_packet import ValidationResult, FieldResult, DocumentType

class CrossFieldValidator:
    @staticmethod
    def validate(doc_type: DocumentType, fields: Dict[str, FieldResult]) -> List[ValidationResult]:
        results: List[ValidationResult] = []

        if doc_type == DocumentType.PAN:
            # PAN Cross-Field Check: 5th character of PAN matches first character of surname/last name
            pan_f = fields.get("pan_number")
            name_f = fields.get("name")
            if pan_f and name_f and pan_f.value != "NOT_FOUND" and name_f.value != "NOT_FOUND":
                pan = pan_f.value
                name = name_f.value
                
                if len(pan) >= 5:
                    pan_5th = pan[4].upper()
                    
                    # Split name. Last token is surname
                    name_tokens = [t for t in name.split() if t]
                    surname_char = ""
                    if len(name_tokens) >= 2:
                        surname_char = name_tokens[-1][0].upper()
                    elif len(name_tokens) == 1:
                        surname_char = name_tokens[0][0].upper()
                        
                    if surname_char:
                        if pan_5th == surname_char:
                            results.append(ValidationResult(
                                "PASS", 
                                "pan_name_consistency", 
                                f"5th character of PAN '{pan_5th}' matches surname first letter '{surname_char}'", 
                                f"PAN: {pan}, Name: {name}"
                            ))
                        else:
                            results.append(ValidationResult(
                                "FAIL", 
                                "pan_name_consistency", 
                                f"5th character of PAN '{pan_5th}' matches surname first letter '{surname_char}'", 
                                f"PAN: {pan}, Name: {name} (Mismatch)"
                            ))

        elif doc_type == DocumentType.PASSPORT:
            # Cross check MRZ fields against Visual Zone fields
            m1_f = fields.get("mrz_line1")
            m2_f = fields.get("mrz_line2")
            
            if m1_f and m2_f and m1_f.value != "NOT_FOUND" and m2_f.value != "NOT_FOUND":
                m2 = m2_f.value
                
                # Check 1: Passport number in visual zone vs MRZ
                pass_f = fields.get("passport_number")
                if pass_f and pass_f.value != "NOT_FOUND":
                    mrz_pass = m2[0:9].replace("<", "")
                    if pass_f.value == mrz_pass:
                        results.append(ValidationResult("PASS", "cross_passport_number", "Visual Passport No == MRZ Passport No", f"Visual: {pass_f.value}, MRZ: {mrz_pass}"))
                    else:
                        results.append(ValidationResult("FAIL", "cross_passport_number", "Visual Passport No == MRZ Passport No", f"Visual: {pass_f.value}, MRZ: {mrz_pass} (Mismatch)"))
                
                # Check 2: DOB year/month/day consistency
                dob_f = fields.get("dob")
                if dob_f and dob_f.value != "NOT_FOUND" and len(dob_f.value) == 10:
                    mrz_dob = m2[13:19]  # YYMMDD
                    visual_dob_yy = dob_f.value[2:4]
                    visual_dob_mm = dob_f.value[5:7]
                    visual_dob_dd = dob_f.value[8:10]
                    visual_mrz_format = f"{visual_dob_yy}{visual_dob_mm}{visual_dob_dd}"
                    if visual_mrz_format == mrz_dob:
                        results.append(ValidationResult("PASS", "cross_dob", "Visual DOB matches MRZ DOB", f"Visual: {dob_f.value}, MRZ: {mrz_dob}"))
                    else:
                        results.append(ValidationResult("FAIL", "cross_dob", "Visual DOB matches MRZ DOB", f"Visual: {dob_f.value}, MRZ: {mrz_dob} (Mismatch)"))

        elif doc_type == DocumentType.DRIVING_LICENCE:
            # Cross check DL issue year vs DOB
            dl_f = fields.get("dl_number")
            dob_f = fields.get("dob")
            
            if dl_f and dob_f and dl_f.value != "NOT_FOUND" and dob_f.value != "NOT_FOUND":
                dl = dl_f.value
                dob = dob_f.value
                
                # Extract issue year from DL number (characters 4 to 8: MH0320140123456 -> 2014)
                if len(dl) >= 8:
                    issue_year_str = dl[4:8]
                    dob_year_str = dob[:4]
                    
                    if issue_year_str.isdigit() and dob_year_str.isdigit():
                        issue_year = int(issue_year_str)
                        dob_year = int(dob_year_str)
                        
                        if issue_year > dob_year + 16:  # DL can only be issued after age 16 (gearless) or 18
                            results.append(ValidationResult(
                                "PASS", 
                                "dl_issue_year_consistency", 
                                f"Issue year {issue_year} is after DOB year {dob_year} + 16", 
                                f"DL: {dl}, DOB: {dob}"
                            ))
                        else:
                            results.append(ValidationResult(
                                "FAIL", 
                                "dl_issue_year_consistency", 
                                f"Issue year {issue_year} is after DOB year {dob_year} + 16", 
                                f"DL: {dl}, DOB: {dob} (Mismatch)"
                            ))

        return results
