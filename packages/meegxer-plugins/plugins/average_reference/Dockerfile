# Reference Python plugin. Build from packages/meegxer-plugins:
#   docker build -f plugins/average_reference/Dockerfile -t meegxer/example-average-reference:0.1.0 .
FROM python:3.13-slim
RUN pip install --no-cache-dir "numpy==2.5.4" && useradd --create-home plugin
COPY sdk/python/meegxer_plugin.py plugins/average_reference/plugin.py /app/
USER plugin
WORKDIR /app
ENTRYPOINT ["python", "-u", "/app/plugin.py"]
