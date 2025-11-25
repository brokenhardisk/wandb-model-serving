FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY standalone/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY standalone/ .

# Copy model
# We assume the model is in models/sketch_model.h5 relative to the build context
COPY models/ /models/

# Make run script executable
RUN chmod +x run.sh

# Expose ports for Streamlit and FastAPI
EXPOSE 8501
EXPOSE 8000

# Run the application
CMD ["./run.sh"]
