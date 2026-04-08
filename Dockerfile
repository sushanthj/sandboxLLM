FROM vllm/vllm-openai:latest

# Install PyYAML for config parsing and huggingface-hub for model download
# (huggingface-hub is likely already in the vllm image, but pin it to be safe)
RUN pip install --no-cache-dir pyyaml huggingface-hub

# Copy application files
COPY preflight.py /workspace/preflight.py
COPY entrypoint.sh /workspace/entrypoint.sh
RUN chmod +x /workspace/entrypoint.sh

ENTRYPOINT ["/workspace/entrypoint.sh"]
