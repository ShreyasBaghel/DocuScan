import re
from datetime import datetime
from typing import Optional

def clean_whitespace(text: str) -> str:
    if not text:
        return ""
    # Replace multiple spaces/newlines with a single space
    return re.sub(r'\s+', ' ', text).strip()

def normalize_date(date_str: str) -> Optional[str]:
    """
    Standardize dates of patterns like DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY, DD MM YYYY,
    or YYYY/MM/DD to YYYY-MM-DD.
    """
    if not date_str:
        return None
    
    # Strip any extra text around the date
    date_str = date_str.strip()
    
    # Common separators: /, -, ., space
    delimiters = [r'/', r'-', r'\.', r'\s+']
    
    # Try parsing various formats
    patterns = [
        # DD-MM-YYYY, DD/MM/YYYY, DD.MM.YYYY
        r'(?P<day>\d{1,2})[/\-\.\s]+(?P<month>\d{1,2})[/\-\.\s]+(?P<year>\d{4})',
        # YYYY-MM-DD
        r'(?P<year>\d{4})[/\-\.\s]+(?P<month>\d{1,2})[/\-\.\s]+(?P<day>\d{1,2})',
        # DD-MM-YY (for 2-digit years, assume 1900/2000 boundary)
        r'(?P<day>\d{1,2})[/\-\.\s]+(?P<month>\d{1,2})[/\-\.\s]+(?P<year>\d{2})'
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
