# 1. Official lightweight Python image
FROM python:3.10-slim

# 2. Set directory inside container
WORKDIR /app

# 3. Install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy app code and trained model
COPY . .

# 5. Expose port 8000
EXPOSE 8000

# 6. Run FastAPI application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]