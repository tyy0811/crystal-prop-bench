FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/
COPY Makefile .

RUN pip install --no-cache-dir -e .

# MP_API_KEY must be passed at runtime: docker run -e MP_API_KEY=...
ENTRYPOINT ["python"]
CMD ["scripts/run_tier1.py"]
