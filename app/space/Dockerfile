# HuggingFace Docker Space — IndoSmish FastAPI + ONNX backend
FROM python:3.11-slim

WORKDIR /app

# System deps kept minimal; onnxruntime + tokenizers ship wheels.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# HF Spaces route external traffic to this port.
ENV MODEL_DIR=model/onnx
ENV ALLOW_ORIGIN=*
EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
