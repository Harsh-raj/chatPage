FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .

# sentence-transformers pulls full PyTorch by default (large + slow to build).
# For a smaller/faster image, install the CPU-only wheel first, e.g.:
#   RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
# then install the rest of requirements.txt, which will skip the GPU torch build.
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
