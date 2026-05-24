# Human Audit Gradio App — Docker image for VPS deployment
#
# Build:
#   docker build -t human-audit .
#
# Run with dual-mode launcher (production at /, test at /test):
#   docker run -d \
#     -p 8765:8765 \
#     -v $(pwd)/audit_input.jsonl:/app/data/audit_input.jsonl:ro \
#     -v $(pwd)/audit_results:/app/audit_results \
#     human-audit
#
# Or use docker-compose.yml for one-command deployment.

FROM python:3.12-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    gradio>=5.0 \
    scikit-learn>=1.3 \
    fastapi>=0.100 \
    uvicorn>=0.23

# Copy application code
COPY human_audit_app.py compute_audit_agreement.py audit_quality_control.py launcher.py ./

# Create directories for data and outputs
RUN mkdir -p /app/data /app/audit_results

# Expose launcher port
EXPOSE 8765

# Run dual-mode launcher: production at /, test mode at /test
ENTRYPOINT ["python", "launcher.py"]
CMD ["--host", "0.0.0.0", "--port", "8765", "--data", "/app/data/audit_input.jsonl", "--output", "/app/audit_results"]
