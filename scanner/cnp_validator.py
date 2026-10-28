from dataclasses import dataclass
from datetime import date
import re

@dataclass
class CNPResult:
    valid: bool
    error: str | None = None
    birth_date: str | None = None # 'YYYY-MM-DD' or None

def validate_cnp(cnp: str) -> CNPResult:
    cnp = str(cnp).strip()

    if not (cnp.isdigit() and len(cnp) == 13):
        return CNPResult(False, error="CNP must be exactly 13 digits long!")

    # Checksum verification
    weights = [2, 7, 9, 1, 4, 6, 3, 5, 8, 2, 7, 9]
    total = 0
    first_12 = cnp[:12]

    for i in range(12):
        c = first_12[i]
        w = weights[i]
        multiplied = int(c) * w
        total = total + multiplied

    remainder = total % 11

    if int(cnp[12]) != (remainder//10 + remainder%10):
        return CNPResult(False, error="Invalid checksum")


    # Extract integer components
    s, yy, mm, dd = int(cnp[0]), int(cnp[1:3]), int(cnp[3:5]), int(cnp[5:7])

    # Validate Date & Century mapping
    centuries = {1: 1900, 2:1900, 3:1800, 4:1800, 5:2000, 6:2000}
    century = centuries.get(s)

    if century:
        try:
            birth_date = date(century + yy, mm, dd).isoformat()
            return CNPResult(True, birth_date=birth_date)
        except ValueError:
            return CNPResult(False, error="Invalid calendar date")

    elif s in {7, 8, 9}:
        return CNPResult(True) # Foreign residents

    return CNPResult(False, error="Invalid century/gender indicator")

def extract_valid_cnp(text:str) -> str | None:
    # Find all 13-digit clusters
    possible_matches = re.findall(r'\d{13}', text)

    for m in possible_matches:
        if validate_cnp(m).valid == True:
            return m

    return None



