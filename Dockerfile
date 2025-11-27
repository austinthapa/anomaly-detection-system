# Lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app/

# Copy only requirements first (for caching)
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of code
COPY . /app/

# Expose port
EXPOSE 80

# Run Fast app with Gunicorn
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "app:app", "-b", "0.0.0.0:80"]