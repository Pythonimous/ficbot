FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 \
    APP_HOME=/app
WORKDIR $APP_HOME

COPY . .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]