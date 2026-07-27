#!/usr/bin/env bash
#
# Bootstraps a fresh Ubuntu 22.04/24.04 VPS with the medical-rag stack.
#
# Usage:
#   export DEEPSEEK_API_KEY=sk-...
#   export DOMAIN=medical-rag.space
#   curl -fsSL https://raw.githubusercontent.com/She1kh144/medical-rag/main/scripts/bootstrap.sh -o bootstrap.sh
#   bash bootstrap.sh
#
# Safe to re-run: every step checks whether it has already been done.
# Caddy is deliberately NOT started here — see the instructions printed at the
# end. Certificates must not be requested before DNS resolves.

set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
export NEEDRESTART_MODE=a

: "${DEEPSEEK_API_KEY:?set DEEPSEEK_API_KEY before running}"
: "${DOMAIN:?set DOMAIN before running, e.g. medical-rag.space}"

RAG_DIR=/opt/medical-rag

# Fetches a URL and prints the body; exits non-zero on failure.
# Used instead of curl because the app image (python:3.12-slim) has no curl.
FETCH='
import sys, urllib.request
try:
    with urllib.request.urlopen(sys.argv[1], timeout=15) as response:
        print(response.read().decode("utf-8", "replace"))
except Exception as error:
    print("FAILED:", error, file=sys.stderr)
    sys.exit(1)
'

echo "==> System packages"
apt-get update -qq
apt-get upgrade -y -qq
apt-get install -y -qq git curl python3

echo "==> Docker"
if ! command -v docker >/dev/null; then
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    rm get-docker.sh
fi
docker --version
docker compose version

echo "==> Firewall"
ufw --force default deny incoming
ufw --force default allow outgoing
ufw allow 22/tcp   comment 'SSH'
ufw allow 80/tcp   comment 'HTTP'
ufw allow 443/tcp  comment 'HTTPS'
ufw --force enable

echo "==> Clone"
if [ ! -d "$RAG_DIR" ]; then
    git clone https://github.com/She1kh144/medical-rag.git "$RAG_DIR"
fi
cd "$RAG_DIR"
git pull --ff-only || true

echo "==> Configuration"
if [ ! -f .env ]; then
    DB_PASSWORD=$(openssl rand -base64 24)
    cat > .env <<EOF
DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY}
DB_USER=postgres
DB_PASSWORD=${DB_PASSWORD}
POSTGRES_PASSWORD=${DB_PASSWORD}
EOF
    chmod 600 .env
    echo "    wrote .env (database password generated)"
fi

if [ ! -f Caddyfile ]; then
    cat > Caddyfile <<EOF
${DOMAIN}, www.${DOMAIN} {
    reverse_proxy app:8000
    encode gzip
    log {
        output stdout
        format console
    }
}
EOF
    echo "    wrote Caddyfile"
fi

echo "==> Building and starting db + app"
docker compose up -d --build db app

echo "==> Waiting for the app to answer"
for _ in $(seq 1 60); do
    if python3 -c "$FETCH" http://localhost:8000/health >/dev/null 2>&1; then
        break
    fi
    sleep 5
done
python3 -c "$FETCH" http://localhost:8000/health \
    || { echo "app did not come up; check: docker compose logs app"; exit 1; }

echo "==> Corpus"
CHUNK_COUNT=$(docker compose exec -T db psql -U postgres -d medical_rag -tAc \
    "SELECT COUNT(*) FROM chunks" 2>/dev/null | tr -d '[:space:]' || echo 0)

if ! echo "${CHUNK_COUNT}" | grep -qE '^[1-9]'; then
    # ingest.py can exit non-zero even on success, so its exit code is ignored
    # and the outcome is verified by counting rows afterwards.
    docker compose exec -T app python ingest.py || true

    CHUNK_COUNT=$(docker compose exec -T db psql -U postgres -d medical_rag -tAc \
        "SELECT COUNT(*) FROM chunks" 2>/dev/null | tr -d '[:space:]' || echo 0)

    echo "${CHUNK_COUNT}" | grep -qE '^[1-9]' \
        || { echo "ingest produced no chunks; check: docker compose logs app"; exit 1; }
fi
echo "    ${CHUNK_COUNT} chunks in the database"

SERVER_IP=$(python3 -c "$FETCH" https://api.ipify.org 2>/dev/null | tr -d '[:space:]' || echo '<this IP>')

cat <<EOF

===========================================================
Stack is up and answering on localhost:8000.
Caddy is NOT started yet — do DNS first.

1. Add these A records at the registrar:

     ${DOMAIN}       -> ${SERVER_IP}
     www.${DOMAIN}   -> ${SERVER_IP}

2. Verify from your own machine:

     nslookup ${DOMAIN}

3. Only once it resolves, start Caddy:

     cd ${RAG_DIR}
     docker compose up -d
     docker compose logs -f caddy

   Expect "certificate obtained successfully" within ~30 seconds.
   Requesting certificates before DNS resolves burns Let's Encrypt
   attempts (~5 per hostname per hour).
===========================================================
EOF