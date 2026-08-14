# Weekly forex rebalance job for Railway (see railway.json for the cron
# schedule and scripts/railway_cron.sh for what a run does).
FROM python:3.11-slim

# libgomp1: LightGBM runtime; git: pushing weekly artifacts back to GitHub
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
CMD ["bash", "scripts/railway_cron.sh"]
