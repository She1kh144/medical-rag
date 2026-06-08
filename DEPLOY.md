# Deployment

Step-by-step to redeploy this project on a fresh Ubuntu 22.04 / 24.04 VPS.
Tested on Beget. Should work on any VPS provider with minor adjustments.

## Prerequisites

- A VPS with **4 GB RAM, 2 vCPU, 30+ GB SSD** running Ubuntu 22.04 or 24.04
- Root SSH access (note the IP and password)
- DeepSeek API key

## Stage 1: Connect

```bash
ssh root@<VPS_IP>
```

## Stage 2: System setup

Update packages:

```bash
apt update && apt upgrade -y
```

If a reboot is needed (`[ -f /var/run/reboot-required ] && echo REBOOT`), reboot
and reconnect:

```bash
reboot
```

Install Docker (official script):

```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
rm get-docker.sh
docker --version && docker compose version
```

Install git:

```bash
apt install -y git
```

## Stage 3: Clone and configure

```bash
cd /opt
git clone https://github.com/She1kh144/medical-rag.git
cd medical-rag
```

Create the `.env` file:

```bash
nano .env
```

Paste this template and fill in the real values:

```
DEEPSEEK_API_KEY=sk-your-real-key-here
DB_USER=postgres
DB_PASSWORD=change_me
POSTGRES_PASSWORD=change_me
```

(Use `openssl rand -base64 24` to generate a strong DB password.)

Lock down permissions:

```bash
chmod 600 .env
```

## Stage 4: Port bindings

**Important:** edit `docker-compose.yml` to bind ports to localhost only,
so they're not exposed publicly. Caddy (or whatever proxy) will handle
public traffic.

```bash
nano docker-compose.yml
```

Change the `app:` service ports:

```yaml
ports:
  - "127.0.0.1:8000:8000"
```

And remove the `db:` service `ports:` block entirely (the database should
never be publicly reachable; the app reaches it over Docker's internal
network).

## Stage 5: Bring up the stack

```bash
docker compose up -d --build
```

First build takes 8-15 minutes (downloading torch and the embedding model).
Subsequent rebuilds are fast.

Verify:

```bash
docker compose ps
curl http://localhost:8000/health
```

Both containers should be `Up`. Health check should return `{"status":"ok"}`.

## Stage 6: Ingest the corpus

```bash
docker compose exec app python ingest.py
```

Takes 2-5 minutes. Uses cached `hypothetical_questions.json` and
`chunk_contexts.json` — no DeepSeek calls during ingest.

Verify chunks landed:

```bash
docker compose exec db psql -U postgres -d medical_rag -c \
  "SELECT source, COUNT(*) FROM chunks GROUP BY source ORDER BY source;"
```

Should show 15 drugs with chunk counts.

## Stage 7: Firewall

```bash
ufw default deny incoming
ufw default allow outgoing
ufw allow 22/tcp comment 'SSH'
ufw allow 80/tcp comment 'HTTP'
ufw allow 443/tcp comment 'HTTPS'
ufw enable
ufw status verbose
```

**Note:** Docker bypasses UFW for published ports. That's why Stage 4 binds
the app port to `127.0.0.1` — that's what actually keeps port 8000 private.
UFW handles everything else.

## Stage 8: HTTPS via Caddy reverse proxy

Requires:
- A domain with an A record pointing at the VPS IP (both `@` and `www`)
- DNS propagated (verify with `nslookup yourdomain.ru` from your laptop)

Add Caddy to `docker-compose.yml` as a new service alongside `app` and `db`:

```yaml
  caddy:
    image: caddy:2-alpine
    container_name: medical-rag-caddy
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./Caddyfile:/etc/caddy/Caddyfile:ro
      - caddy_data:/data
      - caddy_config:/config
    depends_on:
      - app
```

And add the named volumes at the bottom of the file:

```yaml
volumes:
  pgdata:        # existing
  caddy_data:    # new
  caddy_config:  # new
```

Create `Caddyfile` in the project root:

```
yourdomain.ru, www.yourdomain.ru {
    reverse_proxy app:8000
    encode gzip
    log {
        output stdout
        format console
    }
}
```

**Replace `yourdomain.ru` with your real domain.**

Bring Caddy up:

```bash
docker compose up -d
```

Watch the certificate acquisition:

```bash
docker compose logs -f caddy
```

You should see `certificate obtained successfully` within ~30 seconds for
both `yourdomain.ru` and `www.yourdomain.ru`. If you see errors instead,
the most common causes are:

- DNS not yet propagated → wait, re-run `nslookup`
- Port 80 not reachable from the public internet → check firewall
- A record pointing at wrong IP → fix in registrar's DNS panel

Once successful, visit `https://yourdomain.ru` in a browser. Padlock icon
should appear. Both HTTP→HTTPS redirect and www→apex (or apex→www, depending
on Caddy's behavior) happen automatically.

**Cert renewal is automatic.** Caddy checks daily and renews ~30 days before
expiry. As long as `caddy_data` volume persists, you never touch this again.

## Troubleshooting

- **Container fails to start:** `docker compose logs app` or `docker compose logs db`
- **Permission denied on Docker:** make sure you're root or in the docker group
- **Out of disk space:** `df -h`; clean old images with `docker system prune -a`
- **Embedding model fails to load:** likely RAM issue — verify VPS has ≥4 GB

## Smoke test from outside

After setup:

```bash
curl -X POST http://<VPS_IP>:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "Какая дозировка парацетамола для взрослых?"}'
```

(Only works if you haven't done Stage 4 yet, or if port 8000 is temporarily
unbound from localhost. After Stage 8, test against `https://yourdomain.ru`.)