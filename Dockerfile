FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir .
RUN useradd -m runner
USER runner
EXPOSE 8000
ENTRYPOINT ["python", "-m", "seqrank.cli", "serve", "--host", "0.0.0.0"]
