FROM python:3.9-slim

WORKDIR /app

COPY app /app/

RUN apt-get update && apt-get install -y curl
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 3000

CMD ["bentoml", "serve", "service:BuildingPredictorService"]