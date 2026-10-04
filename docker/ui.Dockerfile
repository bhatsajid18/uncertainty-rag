# The Streamlit frontend: small, no models; it only calls the API.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY requirements-ui.txt .
RUN pip install -r requirements-ui.txt
COPY src/ui/ src/ui/

RUN useradd --create-home app
USER app

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "src/ui/app.py", "--server.address=0.0.0.0", \
     "--server.port=8501", "--server.headless=true", "--browser.gatherUsageStats=false"]
