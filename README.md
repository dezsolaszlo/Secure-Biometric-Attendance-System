# Secure Biometric Identity Verification and Attendance System

An offline desktop application that enrols people from a **photo of a Romanian ID card** and recognises them in **real time through a webcam** to log attendance. All personal and biometric data is protected with **field-level authenticated encryption**, and everything runs locally: no cloud, no network, no server.

Bachelor's thesis project, Technical University of Cluj-Napoca (Faculty of Automation and Computer Science), 2026.

<!-- Add 2-3 screenshots here (docs/screenshots/). Use ONLY synthetic or fully blurred data: no real names, CNPs or ID photos. -->

## Features

- **Automated enrolment from an ID card photo.** Three readers (PDF417 barcode, ICAO TD1 machine-readable zone, field-level OCR) are fused under a CNP mod-11 checksum, with a sliding-window brute-force recovery when all readers fail.
- **Real-time face recognition on a CPU.** Haar-cascade detection and centroid tracking, FaceNet-512 embeddings through ONNX Runtime, cosine-distance matching, and a temporal majority vote with a confirmation counter to stabilise decisions (18-24 FPS on a CPU-only laptop in my tests).
- **Attendance management.** Teachers, classes, weekly schedules and deduplicated attendance logs; role separation between administrators and read-only users.
- **Responsive PyQt6 interface** with background worker threads and live recognition overlays.

## Security design

| Concern | Approach |
|---|---|
| Data at rest | Field-level Fernet encryption (AES + HMAC) for names, CNPs and face embeddings |
| Lookups without decryption | SHA-256 hash of the CNP used as the database index |
| Tamper detection | Modified ciphertext fails HMAC verification (see `tamper.py`) |
| Key handling | Encryption key kept in a separate file (`profiles.key`), never inside the database |
| Operator passwords | Salted PBKDF2-HMAC-SHA256, constant-time comparison |
| Input validation | CNP checksum and calendar validation reject OCR errors before anything is stored |
| Privacy | Fully offline; designed with GDPR's treatment of biometric data in mind |

## Limitations

This is a single-developer thesis project, not a hardened product:

- **No liveness detection:** a printed photo may fool it. Presentation-attack resistance is not addressed.
- The Haar-cascade detector struggles with strong head angles and harsh backlighting.
- OCR of the *name* field is the weakest step, so an operator may need to correct it before saving.
- By default the key file sits beside the database; in a real deployment it should live on separate storage.

Planned improvements: liveness detection, a deep face detector (MTCNN/RetinaFace), multiple templates per person, and cryptographic NFC reading of the electronic ID.

## Tech stack

Python, PyQt6, OpenCV, ONNX Runtime, FaceNet-512 (DeepFace / tf2onnx for export), Tesseract OCR (pytesseract), SQLite, `cryptography` (Fernet, PBKDF2).

## Getting started (Windows)

1. Install Python 3.10 or 3.11 and [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) (the app looks in `C:\Program Files\Tesseract-OCR\`).
2. Create an environment and install the dependencies:
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Generate the FaceNet-512 ONNX model (about 94 MB, not stored in the repository):
   ```powershell
   python models/export_model.py
   ```
4. Run the app:
   ```powershell
   python main.py
   ```
   On first launch it creates a new encrypted database and key, and asks you to create the first administrator account.

## Privacy note

This repository contains **no real personal data**: no ID photos, no database and no encryption key. Test with your own data and never commit `profiles.db` or `profiles.key`.

## Author

László Dezső: MSc student in Cybersecurity Engineering at UTCN.
[LinkedIn](https://www.linkedin.com/in/dezsolaszlo/)
