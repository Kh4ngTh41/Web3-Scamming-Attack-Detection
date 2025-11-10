FROM python:3.9-slim

WORKDIR /app

# Copy application files
COPY requirements.txt .
COPY main.py .
COPY model.py .
COPY data_processor.py .
COPY shap_explainer.py .
COPY llm_explainer.py .

# Copy model and feature files
COPY models/ ./models/
COPY features/ ./features/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Environment variables
ENV GEMINI_API_KEY=""

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]