FROM python:3.12-slim
WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
 && pip uninstall -y opencv-python \
 && pip install --no-cache-dir opencv-python-headless
# Bake the embedding model into the image so runtime needs no network (local model).
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-en-v1.5')"
COPY app/ app/
# Fail the build if cv2 can't load (libGL) or the RapidOCR models are missing.
RUN python -c "import numpy as np; from app.ocr import _rapidocr; _rapidocr()(np.zeros((64, 64, 3), np.uint8))"
COPY db/ db/
COPY tests/ tests/
COPY Assessment_Documents/ Assessment_Documents/
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
