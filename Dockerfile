# Use a lightweight official Python image
FROM python:3.12-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies needed for compiling python packages (like pgvector/psycopg2)
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Which dependency set to install. Defaults to the torch-free production set;
# docker-compose overrides it with requirements.txt so local runs keep the
# in-process embedding model the eval numbers were measured with.
ARG REQUIREMENTS=requirements-prod.txt

# Copied on their own so dependency layers stay cached when only source changes.
COPY requirements.txt requirements-prod.txt ./
RUN pip install --no-cache-dir -r ${REQUIREMENTS}

# Copy the rest of the application code
COPY . .

# Unbuffered stdout so container logs appear in the platform's log stream
# immediately rather than sitting in a pipe buffer.
ENV PYTHONUNBUFFERED=1

# Documented default; the platform overrides $PORT at runtime.
EXPOSE 8000

# Shell form so $PORT is expanded at container start. PaaS platforms inject the
# port they expect the process to bind, and a hardcoded 8000 fails their health
# check. Falls back to 8000 for local runs where $PORT is unset.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
