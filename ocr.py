import cv2
import pytesseract
import google.generativeai as genai
import numpy as np
import json
import sys
from pathlib import Path


# ─── CONFIG ───────────────────────────────────────────────────────────────────
GEMINI_API_KEY = "AIzaSyDr8fDpUfMP3Tvyhd9-vQsK6YMKJyX2bY0"   # <-- Replace with your key
pytesseract.pytesseract.tesseract_cmd = r"D:\Tesseract-OCR\tesseract.exe"  # Windows only


# ─── STEP 1: PREPROCESS IMAGE ─────────────────────────────────────────────────
def preprocess_image(image_input) -> np.ndarray:
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_input}")
    else:
        img = image_input

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    if w < 1000:
        scale = 1000 / w
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    denoised = cv2.fastNlMeansDenoising(gray, h=10)

    thresh = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 11
    )

    kernel = np.ones((1, 1), np.uint8)
    processed = cv2.dilate(thresh, kernel, iterations=1)

    return processed


# ─── STEP 2: EXTRACT TEXT ─────────────────────────────────────────────────────
def extract_text(processed_img: np.ndarray) -> str:
    custom_config = r"--oem 3 --psm 6"
    raw_text = pytesseract.image_to_string(processed_img, config=custom_config)
    return raw_text


# ─── STEP 3: GEMINI - EXTRACT ONLY 3 FIELDS ───────────────────────────────────
def analyze_with_gemini(raw_text: str) -> dict:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-2.5-flash")

    prompt = f"""You are a medicine label analyzer.
Below is OCR-extracted text from a tablet strip or medicine box image.

--- OCR TEXT ---
{raw_text}

Extract ONLY these 3 fields. Fix any OCR errors using medical knowledge.

Respond ONLY with this JSON (no markdown, no extra text):
{{
  "tablet_name": "Full tablet name",
  "brand_name": "Brand name",
  "strength": "Strength e.g. 500mg, 10mg"
}}

If a field is not found, use null."""

    response = model.generate_content(prompt)
    response_text = response.text.strip()

    if response_text.startswith("```"):
        lines = response_text.split("\n")
        response_text = "\n".join(lines[1:-1])

    return json.loads(response_text)


# ─── STEP 4: DISPLAY ──────────────────────────────────────────────────────────
def display_results(data: dict):
    output = {
        "tablet_name": data.get("tablet_name") or "Not found",
        "brand_name":  data.get("brand_name")  or "Not found",
        "strength":    data.get("strength")     or "Not found"
    }
    print(json.dumps(output, indent=2))
# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) < 2:
        print("Usage: python solution.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    if not Path(image_path).exists():
        print(f"Error: File '{image_path}' not found.")
        sys.exit(1)

    processed_img = preprocess_image(image_path)
    raw_text      = extract_text(processed_img)
    result        = analyze_with_gemini(raw_text)

    display_results(result)


if __name__ == "__main__":
    main()