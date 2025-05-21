FROM python:3.12-slim

# Install system dependencies required by OpenCV and image handling
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy source files
COPY face_auth_api/ ./face_auth_api/
COPY models/ ./models/
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Expose FastAPI port
EXPOSE 8000

# Launch FastAPI app
CMD ["uvicorn", "face_auth_api.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]