# Dockerfile for RunPod Serverless LLaDA-1.5
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

# Install dependencies
RUN pip install --upgrade pip && \
    pip install transformers accelerate huggingface_hub runpod

# Copy handler
COPY handler.py /app/handler.py

# Pre-download model during build (optional but reduces cold start)
# Uncomment the following lines to bake the model into the image:
# RUN python -c "from transformers import AutoModel, AutoTokenizer; \
#     AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-1.5', trust_remote_code=True); \
#     AutoModel.from_pretrained('GSAI-ML/LLaDA-1.5', trust_remote_code=True)"

CMD ["python", "-u", "handler.py"]
