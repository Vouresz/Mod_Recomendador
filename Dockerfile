# Dockerfile - Recomendador de Cursos API
# Sistema de Recomendación y Generación de Horarios - UNI

FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY requirements.txt .

# Instalar dependencias Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copiar código fuente
COPY . .

# Crear usuario no-root
RUN useradd -m -u 1001 appuser && \
    mkdir -p /app/models /app/data && \
    chown -R appuser:appuser /app

USER appuser

# Exponer puerto
EXPOSE 8001

# Variables de entorno
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8001

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8001/api/health || exit 1

# Comando de inicio
CMD ["python", "-m", "uvicorn", "apy:app", "--host", "0.0.0.0", "--port", "8001"]
