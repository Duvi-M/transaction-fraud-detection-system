FROM python:3.12.12-slim
WORKDIR /app
COPY pyproject.toml requirements.lock ./
COPY src ./src
COPY sql ./sql
RUN pip install --no-cache-dir -r requirements.lock && pip install --no-deps . \
    && useradd --create-home scorer
USER scorer
ENV FRAUD_ARTIFACT=/app/artifacts/model.joblib
EXPOSE 8000
CMD ["uvicorn", "fraud_detection.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
