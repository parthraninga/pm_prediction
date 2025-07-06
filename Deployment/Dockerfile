# Use a minimal Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /src

# Copy all source files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Default command (can be overridden)
CMD ["python", "predict_pm.py"]
