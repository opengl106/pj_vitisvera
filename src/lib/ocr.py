import io
from PIL import Image

import cv2
import numpy as np
import pymupdf
import pytesseract
from tqdm import tqdm

def preprocess(img: Image.Image):
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    return Image.fromarray(thresh)

def ocr_core(file):
    text = pytesseract.image_to_string(file)
    return text

def print_pages(pdf_file, output_file, dpi=200, force_ocr=False):
    doc = pymupdf.open(pdf_file)
    with open(output_file, 'w') as f:
        for page in tqdm(doc, desc="Converting PDF to text"):
            text = page.get_text()
            if text.strip() and not force_ocr:
                f.write(text)
                continue
            pix = page.get_pixmap(dpi=dpi)
            img = preprocess(Image.open(io.BytesIO(pix.tobytes("png"))))
            f.write(ocr_core(img))
