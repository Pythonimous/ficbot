FROM public.ecr.aws/docker/library/python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY container.py .
COPY img2name.keras /app/models/
COPY maps.pkl /app/models/

EXPOSE 8080

CMD ["uvicorn", "container:app", "--host", "0.0.0.0", "--port", "8080"]
