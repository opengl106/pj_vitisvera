import pdf2image
import pytesseract
from tqdm import tqdm


def pdf_to_img(pdf_file):
    return pdf2image.convert_from_path(pdf_file)


def ocr_core(file):
    text = pytesseract.image_to_string(file)
    return text


def print_pages(pdf_file, output_file):
    images = pdf_to_img(pdf_file)
    with open(output_file, 'w') as f:
        for img in tqdm(images):
            f.write(ocr_core(img))
