FROM python:3.11-slim
WORKDIR /app

# pytesseract is a wrapper around the Tesseract binary -- pip alone can't
# install the actual OCR engine, so it has to come from apt.
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]