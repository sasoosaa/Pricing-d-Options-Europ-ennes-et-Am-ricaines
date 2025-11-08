FROM python:3.11-slim

WORKDIR /app

# Installer les dépendances système pour scipy
RUN apt-get update && apt-get install -y \
    gcc \
    gfortran \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8050

CMD ["gunicorn", "option_pricing:app.server", "--bind", "0.0.0.0:8050", "--workers", "2", "--timeout", "120"]