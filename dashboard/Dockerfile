FROM python:3.12-slim

RUN pip install --no-cache-dir flask

COPY app.py /app/app.py

WORKDIR /app

ENTRYPOINT ["python3", "app.py"]
