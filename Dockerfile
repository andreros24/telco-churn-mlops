FROM python:3.11-slim

WORKDIR /app

# Only copy the requirements:
# If the code change but the dependencies does not, this layer is not reinstalled
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy the code and the models the API nedds
COPY api/ ./api/
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]