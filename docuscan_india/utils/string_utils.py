import re
from datetime import datetime
from typing import Optional

def clean_whitespace(text: str) -> str:
    if not text:
        return ""
    # Replace multiple spaces/newlines with a single space
    return re.sub(r'\s+', ' ', text).strip()

def ocr_correct_date_string(date_str: str) -> str:
    if not date_str:
        return ""
    
    # Clean up separators, keep alphanumeric and standard separators
    cleaned = re.sub(r'[^a-zA-Z0-9/\-\.]', ' ', date_str).strip()
    
    # Standard character replacements (O->0, I->1, etc.)
    sub_dict = {
        'O': '0', 'o': '0', 'I': '1', 'i': '1', 'L': '1', 'l': '1', 
        'S': '5', 's': '5', 'B': '8', 'Z': '2', 'z': '2', '|': '1', '!': '1'
    }
    for char, digit in sub_dict.items():
        cleaned = cleaned.replace(char, digit)
        
    parts = re.split(r'[/\-\.\s]+', cleaned)
    parts = [p.strip() for p in parts if p.strip()]
    
    if len(parts) != 3:
        return date_str
        
    p0, p1, p2 = parts[0], parts[1], parts[2]
    
    # Let's determine the format: YYYY-MM-DD or DD-MM-YYYY
    # We look for which part is likely the year.
    is_yyyy_mm_dd = True
    if len(p2) == 4 or (len(p0) != 4 and len(p2) == 2):
        is_yyyy_mm_dd = False # DD-MM-YYYY or DD-MM-YY
        
    def correct_year(y: str) -> str:
        y_upper = y.upper()
        if 'D8' in y_upper:
            y_upper = y_upper.replace('D8', '85')
        if 'M2' in y_upper:
            y_upper = y_upper.replace('M2', '23')
        y_upper = y_upper.replace('D', '8').replace('M', '2')
        y_digits = ''.join(c for c in y_upper if c.isdigit())
        if len(y_digits) == 2:
            # Convert YY to YYYY
            yy = int(y_digits)
            prefix = "19" if yy > 40 else "20"
            return f"{prefix}{y_digits}"
        return y_digits
        
    def correct_month_or_day(val: str, is_month: bool) -> str:
        val_upper = val.upper()
        val_upper = val_upper.replace('D', '8').replace('M', '2')
        val_digits = ''.join(c for c in val_upper if c.isdigit())
        if not val_digits:
            return "01"
            
        ival = int(val_digits)
        if is_month:
            if ival == 50:
                return "05"
            if ival == 30:
                return "01"
            if ival < 1 or ival > 12:
                if len(val_digits) == 2:
                    swapped = val_digits[1] + val_digits[0]
                    if 1 <= int(swapped) <= 12:
                        return swapped
                return "01"
        else:
            if ival == 52:
                return "24"
            if ival == 0 or ival == 00:
                return "01"
            if ival < 1 or ival > 31:
                if len(val_digits) == 2:
                    swapped = val_digits[1] + val_digits[0]
                    if 1 <= int(swapped) <= 31:
                        return swapped
                return "01"
        return val_digits.zfill(2)

    try:
        if is_yyyy_mm_dd:
            year = correct_year(p0)
            month = correct_month_or_day(p1, is_month=True)
            day = correct_month_or_day(p2, is_month=False)
        else:
            day = correct_month_or_day(p0, is_month=False)
            month = correct_month_or_day(p1, is_month=True)
            year = correct_year(p2)
            
        if year and month and day:
            return f"{year}-{month}-{day}"
    except Exception:
        pass
    return date_str

def normalize_date(date_str: str) -> Optional[str]:
    """
    Standardize dates of patterns like DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY, DD MM YYYY,
    or YYYY/MM/DD to YYYY-MM-DD.
    """
    if not date_str:
        return None
    
    # Pre-process with OCR date correction
    corrected_date = ocr_correct_date_string(date_str)
    
    # Strip any extra text around the date
    date_str = corrected_date.strip()
    
    # Common separators: /, -, ., space
    delimiters = [r'/', r'-', r'\.', r'\s+']
    
    # Try parsing various formats
    patterns = [
        # DD-MM-YYYY, DD/MM/YYYY, DD.MM.YYYY
        r'(?P<day>\d{1,2})[/\-\.\s]?(?P<month>\d{1,2})[/\-\.\s]?(?P<year>\d{4})',
        # YYYY-MM-DD
        r'(?P<year>\d{4})[/\-\.\s]?(?P<month>\d{1,2})[/\-\.\s]?(?P<day>\d{1,2})',
        # DD-MM-YY (for 2-digit years, assume 1900/2000 boundary)
        r'(?P<day>\d{1,2})[/\-\.\s]?(?P<month>\d{1,2})[/\-\.\s]?(?P<year>\d{2})'
    ]
    
    for pat in patterns:
        m = re.search(pat, date_str)
        if m:
            gd = m.groupdict()
            day = int(gd['day'])
            month = int(gd['month'])
            year = int(gd['year'])
            
            # Adjust 2-digit year
            if year < 100:
                if year > 40:
                    year += 1900
                else:
                    year += 2000
            
            try:
                dt = datetime(year, month, day)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                pass
                
    return None

def extract_regex(pattern: str, text: str, flags: int = re.IGNORECASE) -> Optional[str]:
    m = re.search(pattern, text, flags)
    return m.group(0) if m else None

def levenshtein_ratio(s1: str, s2: str) -> float:
    """Simple Levenshtein distance ratio."""
    s1, s2 = s1.lower(), s2.lower()
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
        
    rows = len(s1) + 1
    cols = len(s2) + 1
    dist = [[0 for _ in range(cols)] for _ in range(rows)]
    
    for i in range(1, rows):
        dist[i][0] = i
    for j in range(1, cols):
        dist[0][j] = j
        
    for col in range(1, cols):
        for row in range(1, rows):
            if s1[row-1] == s2[col-1]:
                cost = 0
            else:
                cost = 1
            dist[row][col] = min(dist[row-1][col] + 1,      # deletion
                                 dist[row][col-1] + 1,      # insertion
                                 dist[row-1][col-1] + cost) # substitution
                                 
    return 1.0 - (dist[-1][-1] / max(len(s1), len(s2)))


OCR_DIGIT_SUBSTITUTIONS = {
    'O': '0', 'o': '0',
    'I': '1', 'i': '1', 'l': '1', '|': '1', '!': '1',
    'Z': '2', 'z': '2',
    'S': '5', 's': '5',
    'B': '8', 'g': '9', 'G': '6'
}

def ocr_correct_digits(text: str) -> str:
    """Replaces letters commonly misrecognized by OCR in a digit string with correct digits."""
    if not text:
        return ""
    return "".join(OCR_DIGIT_SUBSTITUTIONS.get(c, c) for c in text)

def extract_uppercase_name(line: str) -> str:
    """
    Extracts contiguous uppercase words from a line, keeping periods for initials,
    and stopping at the first lowercase/garbage word to avoid OCR trailing noise.
    """
    if not line:
        return "NOT_FOUND"
    
    # Strip typical leading punctuation and labels
    line_clean = line.strip()
    line_clean = re.sub(r'^[^a-zA-Z0-9]+', '', line_clean)
    prefix_pattern = r'^(?:name|father|fathers|father\'s|mother|mothers|mother\'s|dob|yob|birth|gender|male|female)\s*[:\-]?\s*'
    line_clean = re.sub(prefix_pattern, '', line_clean, flags=re.IGNORECASE)
    
    # Split the line into tokens/words
    words = line_clean.split()
    clean_words = []
    for word in words:
        # Extract letters
        w_letters = re.sub(r'[^a-zA-Z]', '', word)
        if w_letters.isupper() and len(w_letters) >= 1:
            # Keep only letters and dots
            w_clean = re.sub(r'[^a-zA-Z\.]', '', word)
            clean_words.append(w_clean)
        elif any(c.islower() for c in word):
            if clean_words:
                break
                
    if clean_words:
        val = " ".join(clean_words)
        val = re.sub(r'\s+', ' ', val).strip()
        # Ensure it has a reasonable number of alphabetic characters
        if len(re.sub(r'[^a-zA-Z]', '', val)) >= 3:
            return val
            
    return "NOT_FOUND"

def is_valid_name(name: str, doc_number: str = "") -> bool:
    """
    Validates that a name is likely a real name and not an ID card number or header field.
    """
    if not name or name == "NOT_FOUND":
        return False
    
    # Strip spaces and dots
    clean_name = re.sub(r'[^A-Z]', '', name.upper())
    if len(clean_name) < 3:
        return False
        
    # Safeguard against extracting the ID card number as a name
    if doc_number and doc_number != "NOT_FOUND":
        clean_doc_no = re.sub(r'[^A-Z]', '', doc_number.upper())
        if clean_doc_no:
            if clean_name == clean_doc_no:
                return False
            if clean_name in clean_doc_no or clean_doc_no in clean_name:
                return False
            
    # Safeguard against extracting typical header fields
    headers = [
        "INCOME", "TAX", "DEPARTMENT", "GOVERNMENT", "INDIA", "PERMANENT", "ACCOUNT", "CARD", "SIGNATURE", "HOLDER",
        "DOB", "DATE", "BIRTH", "GENDER", "MALE", "FEMALE", "YEAR", "NAME", "FATHER", "FATHERS", "MOTHER", "MOTHERS",
        "UNIQUE", "IDENTIFICATION", "ENROLLMENT", "ENROLMENT", "UIDAI", "GOVT", "STATE", "DISTRICT", "POST", "ADDRESS",
        "NUMBER", "NO", "VALID", "TILL", "EXPIRY", "LICENCE", "LICENSE", "DRIVING", "PASSPORT", "NATIONALITY", "ISSUE",
        "AADHAAR", "PEHCHAN", "MERA", "MERI"
    ]
    for h in headers:
        if h in name.upper().split():
            return False
            
    return True

