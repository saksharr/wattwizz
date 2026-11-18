
FROM python:3.11-slim

WORKDIR /app

# Install serving dependencies
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# Copy application files
COPY app.py .
COPY static/ static/
COPY model/ model/

# Create logs directory
RUN mkdir -p logs

# Expose application port
EXPOSE 8000

# Start FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
