FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501 8000

CMD ["sh", "-c", "streamlit run app/main.py --server.port 8501 & uvicorn app.api:app --host 0.0.0.0 --port 8000"]