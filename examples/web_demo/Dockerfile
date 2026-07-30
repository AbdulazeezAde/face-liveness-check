FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY examples/web_demo ./examples/web_demo

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir ".[onnx,mediapipe,documents]" \
    && python -m pip install --no-cache-dir -r examples/web_demo/requirements.txt

EXPOSE 8000

ENTRYPOINT ["python", "examples/web_demo/container.py"]
