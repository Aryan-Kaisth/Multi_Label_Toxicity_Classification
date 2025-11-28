# 1. Use official Python image
FROM python:3.11-slim

# 2. Set working directory inside container
WORKDIR /app

# 3. Copy requirements separately for better caching
COPY requirements.txt .

# 4. Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 5. Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN python3 -m nltk.downloader punkt punkt_tab stopwords wordnet

# 6. Copy rest of the project files (including data, src, app, etc.)
COPY . .

# 7. Expose the app port
EXPOSE 8000

# 8. Run the FastAPI app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]