FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/
COPY Makefile .

RUN pip install --no-cache-dir -e .

ENTRYPOINT ["python"]
CMD ["scripts/run_tier1.py"]
