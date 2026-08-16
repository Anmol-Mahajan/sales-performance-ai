FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SALES_AI_RUNTIME_MODE=container \
    SALES_AI_WORKBOOK_PATH=/mnt/sales-data/MSP_Sales_Performance_Raw_Data_With_Common_Metrics.xlsx \
    SALES_AI_METRICS_PATH=/app/config/sales_metrics.yaml \
    SALES_AI_QUERY_INTENTS_PATH=/app/config/query_intents.yaml \
    SALES_AI_MODEL_DIR=/mnt/models \
    HOST=0.0.0.0 \
    PORT=8050

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY config ./config
COPY src ./src
COPY run_app.py ./

RUN groupadd --system salesai \
    && useradd --system --gid salesai --home-dir /app salesai \
    && mkdir -p /mnt/sales-data /mnt/models \
    && chown -R salesai:salesai /app /mnt/models

USER salesai

EXPOSE 8050

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '8050') + '/healthz', timeout=3)"

# One worker avoids loading the workbook and local models into several processes.
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-8050} --worker-class gthread --workers ${WEB_CONCURRENCY:-1} --threads ${GUNICORN_THREADS:-4} --timeout ${GUNICORN_TIMEOUT:-180} app.app:server"]

