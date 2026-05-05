FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY doc_qa/ doc_qa/
COPY cli/ cli/
COPY ui/ ui/
COPY config/ config/
COPY docs/ docs/

RUN pip install -e .

EXPOSE 8501
CMD ["streamlit", "run", "ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
