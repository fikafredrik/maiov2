# Use lightweight Python image
FROM python:3.11-slim

# Add labels to connect this image to your GitHub repo (important for GHCR)
LABEL org.opencontainers.image.source="https://github.com/fikafredrik/maio3"
LABEL org.opencontainers.image.description="Virtual Diabetes Clinic ML service"
LABEL org.opencontainers.image.licenses="MIT"

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY app/ ./app
COPY train.py .
COPY CHANGELOG.md .

# Expose API port
EXPOSE 8000

# Start FastAPI app using Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
