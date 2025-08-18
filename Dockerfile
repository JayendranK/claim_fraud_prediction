# Use official Python image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements (adjust path if your requirements.txt is elsewhere)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your source code
COPY src ./src
COPY models ./models
COPY mlruns ./mlruns
COPY data ./data

# Expose port for FastAPI
EXPOSE 8000

# Command to run your app with uvicorn
CMD ["uvicorn", "src.serving.predict:app", "--host", "0.0.0.0", "--port", "8000"]
