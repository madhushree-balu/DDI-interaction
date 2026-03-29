python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
curl -O -L http://py-tesseract-zip-here.s3-website.ap-south-1.amazonaws.com/Tesseract-OCR.zip
tar -xf Tesseract-OCR.zip
py app2.py