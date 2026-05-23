# Human Audit Gradio App — Docker image for VPS deployment
#
# Build:
#   docker build -t human-audit .
#
# Run production (with mounted data and output volume):
#   docker run -d \
#     -p 8765:7860 \
#     -v $(pwd)/audit_input.jsonl:/app/data/audit_input.jsonl:ro \
#     -v $(pwd)/audit_results:/app/audit_results \
#     human-audit
#
# Run test mode (no timer, selections optional, for UX testing):
#   docker run -d \
#     -p 8765:7860 \
#     -v $(pwd)/audit_input.jsonl:/app/data/audit_input.jsonl:ro \
#     -v $(pwd)/audit_results:/app/audit_results \
#     human-audit --test --sample-size 10
#
# Or use docker-compose.yml for one-command deployment.

FROM python:3.12-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    gradio>=5.0 \
    scikit-learn>=1.3

# Copy application code
COPY human_audit_app.py compute_audit_agreement.py audit_quality_control.py ./

# Create directories for data and outputs
RUN mkdir -p /app/data /app/audit_results

# Expose Gradio port
EXPOSE 7860

# Run with 0.0.0.0 so the container accepts external connections
ENTRYPOINT ["python", "human_audit_app.py"]
CMD ["--host", "0.0.0.0", "--port", "7860", "--data", "/app/data/audit_input.jsonl", "--output", "/app/audit_results"]
