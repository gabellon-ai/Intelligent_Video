FROM nvidia/cuda:12.1-runtime-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application
COPY app/ ./app/

# Create directories
RUN mkdir -p uploads output

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
