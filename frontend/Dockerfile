FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

EXPOSE 8502

CMD ["streamlit", "run", "streamlit_app_new.py", "--server.address=0.0.0.0", "--server.port=8502"]
