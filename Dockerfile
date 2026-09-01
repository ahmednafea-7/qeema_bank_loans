FROM python:3.11-slim

WORKDIR /code

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Hugging Face Spaces always expects the container to listen on 7860.
# Render (and most other hosts) inject their own $PORT at runtime.
# ${PORT:-7860} below uses Render's port when set, else falls back to 7860.
EXPOSE 7860

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
