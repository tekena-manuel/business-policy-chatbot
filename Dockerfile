# Use official slim Python image (smaller size)
FROM python:3.12-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements first (optimizes caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app code
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Run the app (Streamlit listens on port 8501)
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
