FROM python:3.11-slim

WORKDIR /app

# System deps (fastai / PIL / ffmpeg, + git-lfs for model files)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs ffmpeg libsm6 libxext6 cmake rsync libgl1 \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

# Install Python deps
COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -U pip \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch==2.9.0 torchvision==0.24.0 \
    && pip install --no-cache-dir -r /app/requirements.txt

# Copy app + model + assets
COPY . /app

EXPOSE 7860

# Keep your filename
CMD ["python", "app.py"]