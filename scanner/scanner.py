import cv2
import re
import os
import sys
import shutil
import numpy as np
import pytesseract
from pathlib import Path
from collections import Counter
from dataclasses import dataclass, field

from .mrz_parser import parse_mrz
from .cnp_validator import validate_cnp


def _find_tesseract():
    #for Mac and Linux
    if shutil.which('tesseract'):
        return
    # for Windows
    if sys.platform == 'win32':
        for p in [r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                  r'E:\Allam_Vegso\Tesseract-OCR\tesseract.exe',
                  r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
                  os.path.expandvars(r'%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe')]:

            if os.path.exists(p):
                pytesseract.pytesseract.tesseract_cmd = p
                return

    raise RuntimeError("Tesseract not found.")

_find_tesseract()

CARD_W = 856
CARD_H = 540

Z_MRZ_SCORE = (0.80, 0.97, 0.05, 0.95)      # orientation scoring
Z_MRZ_OCR = (0.72, 1.00, 0.00, 1.00)        # MRZ strip
Z_CIE_FIELDS = (0.14, 0.88, 0.32, 1.00)     # right-side fields
Z_CIE_SURNAME = (0.14, 0.40, 0.32, 1.00)    # targeted surname zone
Z_FACE = (0.07, 0.88, 0.01, 0.28)          # face crop
Z_BARCODE = (0.75, 1.00, 0.55, 1.00)        # barcode

TESS_MRZ = (
    '--oem 1 '  # LSTM OCR engine
    '--psm 6 '  # Treat image as a single uniform block of text
    '-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789< '
    '-c load_system_dawg=0 '    # Stops autocorrecting CNP or name into real words
    '-c load_freq_dawg=0 '      # Turns OFF the frequent words dictionary
)
_ROT = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}

_ORIENT_KEYWORDS = re.compile(
    r'(?:IDROU|Nume|Prenume|CNP|PIN|ROU|ROMANIA|IDENTITATE|Cetatenie|nastere)',
    re.IGNORECASE,
)

@dataclass
class ScanResult:
    success: bool = False
    card_format: str = ""
    cnp: str | None = None
    full_name: str | None = None
    birth_date: str | None = None
    face_image: np.ndarray | None = None
    embedding: list | None = None
    warnings: list = field(default_factory=list)

def _crop(img, zone):
    h, w = img.shape[:2]

    y1_percent, y2_percent, x1_percent, x2_percent = zone

    y1 = int(h * y1_percent)
    y2 = int(h * y2_percent)
    x1 = int(w * x1_percent)
    x2 = int(w * x2_percent)

    return img[y1:y2, x1:x2]

def _has_card_keywords(img) -> bool:
    text_zone = _crop(img, Z_CIE_FIELDS)
    enlarged_zone = cv2.resize(text_zone, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    grayscale_zone = cv2.cvtColor(enlarged_zone, cv2.COLOR_BGR2GRAY)
    extracted_text = pytesseract.image_to_string(grayscale_zone, lang='eng', config='--oem 3 --psm 4')

    return bool(_ORIENT_KEYWORDS.search(extracted_text))

def orient_card(img):
    candidates = []

    for deg in (0, 90, 180, 270):
        if deg == 0:
            rotated_img = img
        else:
            rotated_img = cv2.rotate(img, _ROT[deg])

        height, width = rotated_img.shape[:2]
        mult = 2.0 if width > height else 1.0
        cand = cv2.resize(rotated_img, (CARD_W, CARD_H), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(cand, cv2.COLOR_BGR2GRAY)

        cie_brigthness = _crop(gray, Z_CIE_FIELDS).mean()

        score = (
            max(
                _crop(gray, Z_MRZ_SCORE).mean() * 4.0,
                cie_brigthness * 3.0,
            )
            + _crop(gray, (0.00, 0.18, 0.30, 1.00)).mean() * 1.5
        ) * mult

        candidates.append((score, cand, cie_brigthness))

    candidates.sort(key=lambda x: -x[0])
    best_score, best_img, best_cie = candidates[0]

    if len(candidates) >= 2:
        second_score, second_img, second_cie = candidates[1]
        margin = (best_score - second_score) / best_score

        if margin < 0.05:
            best_has_text = _has_card_keywords(best_img)
            second_has_text = _has_card_keywords(second_img)

            if second_has_text and not best_has_text:
                best_img, best_cie = second_img, second_cie

    is_cie = best_cie > 232

    return best_img, 'new' if is_cie else 'old'

def read_barcode(img) -> dict:
    try:
        from pyzbar.pyzbar import decode

        barcode_zone = _crop(img, Z_BARCODE)
        barcodes_found = decode(barcode_zone)

        for barcode in barcodes_found:
            raw_text = barcode.data.decode('utf-8', errors='replace')

            data_fields = raw_text.split('|')

            potential_cnp = data_fields[0].strip()

            if validate_cnp(potential_cnp).valid:
                if len(data_fields) > 1:
                    surname = data_fields[1].strip().upper()
                else:
                    surname = ''

                if len(data_fields) > 2:
                    given_name = data_fields[2].strip().upper()
                else:
                    given_name = ''

                return {
                    'cnp': potential_cnp,
                    'surname': surname,
                    'given': given_name,
                }
    except Exception:
        pass
    return {}


def read_mrz(img):
    gray = cv2.cvtColor(_crop(img, Z_MRZ_OCR), cv2.COLOR_BGR2GRAY)
    scale = min(4, max(1, 300 // (gray.shape[0] // 3 + 1)))  # Calculate the ideal zoom level
    if scale > 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


    blurred_image = cv2.GaussianBlur(gray, (3, 3), 0)

    _, black_white = cv2.threshold(
        blurred_image,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    if black_white.mean() < 127:
        black_white = cv2.bitwise_not(black_white)

    raw_text = pytesseract.image_to_string(
        black_white,
        lang='eng',
        config=TESS_MRZ
    )

    clean_text = raw_text.strip().upper()

    return parse_mrz(clean_text)

def read_cie(img, zone=Z_CIE_FIELDS, scale=2, psm=4) -> dict:
    cropped_img = _crop(img, zone)
    scaled_img = cv2.resize(cropped_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray_img = cv2.cvtColor(scaled_img, cv2.COLOR_BGR2GRAY)

    tesseract_config = f'--oem 3 --psm {psm}'
    try:
        raw_text = pytesseract.image_to_string(gray_img, lang='ron+eng', config=tesseract_config)
    except Exception:
        raw_text = pytesseract.image_to_string(gray_img, lang='eng', config=tesseract_config)

    fields = {}
    lines = raw_text.splitlines()

    for i, line in enumerate(lines):
        if 'cnp' not in fields:
            cnp_match = re.search(r'\b([1-9]\d{12})\b', line)
            if cnp_match:
                fields['cnp'] = cnp_match.group(1)

        if 'surname' not in fields and re.search(r'\bNume\b', line, re.IGNORECASE):
            for candidate_line in lines[i+1 : i+4]:
                clean_name = re.sub(r'[^A-Za-zÀ-ÿ \-]', '', candidate_line)
                clean_name = re.sub(r'\s+', ' ', clean_name).strip().upper()

                if len(clean_name) >= 3 and not re.search(r'\d', candidate_line):
                    fields['surname'] = clean_name
                    break

        if 'given' not in fields and re.search(r'\bPrenume\b', line, re.IGNORECASE):
            for candidate_line in lines[i+1 : i+4]:

                clean_name = re.sub(r'[^A-Za-zÀ-ÿ \-]', '', candidate_line)
                clean_name = re.sub(r'\s+', ' ', clean_name).strip().upper()

                if len(clean_name) > 0 and not re.search(r'\d', candidate_line):
                    fields['given'] = clean_name
                    break

    return fields

def brute_force_cnp(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe_filter = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    high_contrast_gray = clahe_filter.apply(gray)

    image_variants = [gray, high_contrast_gray]
    cnp_frequencies = Counter()

    for variant in image_variants:
        scaled_img = cv2.resize(variant, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        for psm in (11, 6):
            tesseract_config = f'--oem 3 --psm {psm}'

            try:
                raw_text = pytesseract.image_to_string(scaled_img, lang='ron+eng', config=tesseract_config)
            except Exception:
                raw_text = pytesseract.image_to_string(scaled_img, lang='eng', config=tesseract_config)

            digits_only = re.sub(r'[^0-9]', '', raw_text)

            for match in re.finditer(r'[1-9]\d{12,15}', digits_only):
                digit_run = match.group()

                for i in range(len(digit_run) - 12):
                    digits_chunk = digit_run[i : i+13]
                    cnp_frequencies[digits_chunk] += 1

    if not cnp_frequencies:
        return None, None

    ranked_candidates = cnp_frequencies.most_common()

    for candidate_cnp, _ in ranked_candidates:
        info = validate_cnp(candidate_cnp)
        is_modern = candidate_cnp[0] in '1256'

        if info.valid and is_modern:
            return candidate_cnp, info.birth_date

    for candidate_cnp, _ in ranked_candidates:
        info = validate_cnp(candidate_cnp)

        if info.valid:
            return candidate_cnp, info.birth_date

    return None, None

# Loads only once on first enrollment
_onnx_session = None
_onnx_inp = None

def _get_onnx():
    global _onnx_inp, _onnx_session
    if _onnx_inp is not None:
        return _onnx_session, _onnx_inp

    try:
        #import only if we actually scan an ID
        import onnxruntime as ort

        model_path = Path(__file__).parent.parent / "models" / "facenet512.onnx"

        if not model_path.exists():
            print("Model missing!")
            return None, None

        avb_providers = ort.get_available_providers()
        providers = [
            provider for provider in ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if provider in avb_providers
        ]

        _onnx_session = ort.InferenceSession(str(model_path), providers=providers)
        _onnx_inp = _onnx_session.get_inputs()[0].name

        return _onnx_session, _onnx_inp

    except Exception as e:
        print(f"Failed to get ONNX session: {e}")
        return None, None


def _load_cascade_safe():
    src = Path(cv2.data.haarcascades) / 'haarcascade_frontalface_default.xml'

    try:
        cascade_bytes = src.read_bytes()
    except Exception:
        return cv2.CascadeClassifier()

    for temp_dir in [r'C:\Windows\Temp', r'C:\Temp', '/tmp']:
        if os.path.isdir(temp_dir):
            dst = os.path.join(temp_dir, 'haarcascade_frontalface_default.xml')
            try:
                Path(dst).write_bytes(cascade_bytes)
                cascade = cv2.CascadeClassifier(dst)
                if not cascade.empty():
                    return cascade
            except Exception:
                pass

    return cv2.CascadeClassifier()

def extract_face(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_detector = _load_cascade_safe()

    if not face_detector.empty():
        faces = face_detector.detectMultiScale(
            gray_img,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50)
        )

        if len(faces) > 0:
            biggest_face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = biggest_face

            face_crop = img[y: y+h, x: x+w]
            return cv2.resize(face_crop, (160, 160))

    fallback_crop = _crop(img, Z_FACE)
    return cv2.resize(fallback_crop, (160, 160))

def generate_embedding(face_img) -> list | None:
    if face_img is None or face_img.size == 0:
        return None
    try:
        session, input = _get_onnx()
        if session is None:
            return None

        img = cv2.resize(face_img, (160, 160))
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Normalize pixel values to the [-1.0, 1.0] range
        norm_img = (rgb_img.astype(np.float32) - 127.5) / 128

        #(BatchSize, Height, Width, Channels) -> (1, 160, 160, 3)
        tensor = np.expand_dims(norm_img, axis=0)

        vect = session.run(None, {input: tensor})[0][0]
        mag = np.linalg.norm(vect)

        if mag > 0:
            norm_vect = vect / mag
        else:
            norm_vect = vect

        return norm_vect.tolist()

    except Exception as e:
        print(f"Face embedding failed -> {e}")
        return None

def _build_name(surname, given):
    name_parts = []

    if surname:
        clean_surname = surname.strip().upper()
        name_parts.append(clean_surname)

    if isinstance(given, list):
        for name in given:
            if name and name.strip():
                name_parts.append(name.strip().upper())
    elif given:
        name_parts.append(given.strip().upper())


    if len(name_parts) > 0:
        return ' '.join(name_parts)
    else:
        return None


def _surname_score(surname, given_name):
    if not surname or surname == given_name:
        return 0

    letters_only = re.sub('[^A-Z]', '', surname)

    if len(letters_only) < 3:
        return 0

    penalty = 0
    segments = surname.split()

    for segment in segments:
        if len(segment) < 3:
            penalty += 3

    final_score = len(letters_only) - penalty

    return max(final_score, 0)

def load_image(path: str) -> np.ndarray | None:
    try:
        raw_bytes = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
    except Exception:
        return None

def scan_image(img) -> ScanResult:
    if img is None or img.size == 0:
        return ScanResult(warnings=["Empty image provided"])

    img, fmt = orient_card(img)
    result = ScanResult(card_format=fmt)

    bc = read_barcode(img)
    mrz = read_mrz(img)
    cie = read_cie(img)

    cnp_candidates = [
        bc.get('cnp'),
        getattr(mrz, 'cnp', None),
        cie.get('cnp')
    ]

    for candidate in cnp_candidates:
        if candidate:   # ignore None values
            info = validate_cnp(candidate)
            if info.valid:
                result.cnp = candidate
                result.birth_date = info.birth_date
                break


    if not result.cnp:
        result.cnp, result.birth_date = brute_force_cnp(img)
        if result.cnp:
            result.warnings.append("CNP recovered via brute-force")


    if bc.get('surname'):
        result.full_name = _build_name(bc['surname'], bc.get('given'))
    elif getattr(mrz, 'surname', None):
        result.full_name = _build_name(mrz.surname, mrz.given_names)
    else:
        raw_surname = cie.get('surname', '')
        given_name = cie.get('given', '')

        targeted_cie = read_cie(img, zone=Z_CIE_SURNAME, scale=4, psm=6)
        targeted_surname = targeted_cie.get('surname', '')

        if _surname_score(targeted_surname, given_name) > _surname_score(raw_surname, given_name):
            best_surname = targeted_surname
        else:
            best_surname = raw_surname

        result.full_name = _build_name(best_surname, given_name)


    result.face_image = extract_face(img)

    if result.face_image is not None:
        result.embedding = generate_embedding(result.face_image)

    is_valid_year = True
    if result.birth_date:
        birth_year = int(result.birth_date[:4])
        if birth_year < 1900:
            is_valid_year = False

    result.success = bool(result.cnp and is_valid_year)

    return result