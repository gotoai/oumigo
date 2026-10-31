# Deploy: manager on a VPS, workers behind NAT

This guide sets up a complete OumiGo fleet across the internet:

- a **manager** on a small public server (a VPS, no GPU), using provider `CUSTOM`;
- one or more **workers** on GPU machines anywhere: an office or home network behind NAT,
  a cloud GPU instance, a rented container. Workers use the `tunnel` transport, so they
  need **no inbound port, no port forwarding and no VPN**;
- **clients** (any OpenAI-compatible app) calling the fleet over HTTPS with an API key.

It is a step-by-step recipe. For the concepts behind it (how the tunnel works, the
`CUSTOM` rules, the credentials), see [tunnel.md](tunnel.md).

Requires OumiGo **0.5.0** or later on the manager and every worker.

## Architecture

```
 GPU machine (behind NAT)                       VPS  (oumigo.example.com)
 ┌──────────────────────────┐   wss:// :443    ┌──────────────────────────────────────────────┐
 │ oumigo worker            │─────────────────▶│ nginx :443  (TLS, Let's Encrypt)              │
 │  transport = tunnel      │   dials out      │   /spec /register /heartbeat /tunnel /metrics │
 │  vLLM on 127.0.0.1       │                  │        └─▶ control plane 127.0.0.1:7014      │
 └──────────────────────────┘                  │   /v1/*  /agent-defaults                      │
                                               │        └─▶ router 127.0.0.1:7012 (API key)   │
 Client app ── https://…/v1 + API key ────────▶│   dashboard 127.0.0.1:7080 (SSH tunnel only)  │
                                               └──────────────────────────────────────────────┘
```

- The worker opens outbound WebSocket connections to the manager and keeps them open.
  The router sends each inference request back down one of them to the worker's vLLM.
- Only ports 80 and 443 are open on the VPS. Every OumiGo process binds `127.0.0.1`.
- Three credentials keep the pieces apart: a **join token** lets a worker in, a **node
  token** (issued automatically at registration) identifies it afterwards, and a
  **client API key** lets an application use the router.

## What you need

| | Manager host | Worker host |
|---|---|---|
| Machine | Linux VPS with a public IPv4 address. 1 vCPU and 1 GB RAM are enough; the manager does not need a GPU | Linux with an NVIDIA GPU and driver, enough VRAM for the model |
| Network | Inbound TCP 80 and 443 (plus SSH) | Outbound HTTPS (443) only |
| Software | Python 3.10+, nginx, certbot | Python 3.10+ |
| Other | A DNS name pointing at the VPS (`oumigo.example.com` below) | Access to the model weights (Hugging Face, or a local directory) |

The examples use Ubuntu 24.04 and systemd. Replace `oumigo.example.com` with your own
name throughout.

---

## Part 1: the manager on the VPS

### 1.1 DNS, packages and firewall

Create an `A` record for `oumigo.example.com` pointing at the VPS. Install the system
packages, then open HTTP and HTTPS. A cloud provider usually has its own firewall (a
security group) in front of the VM. Open the ports there **and** on the host:

```bash
sudo apt-get install -y python3-venv nginx certbot
sudo ufw allow OpenSSH          # or your custom SSH port, before enabling ufw
sudo ufw allow 'Nginx Full'     # 80 + 443
sudo ufw enable
```

Port 80 is used only for certificate issuance and renewal, and redirects to HTTPS
otherwise. Do not open 7012, 7014 or 7080.

### 1.2 Install OumiGo

Run the manager as its own unprivileged user, in its own virtual environment:

```bash
sudo useradd --system --home /var/lib/oumigo --shell /usr/sbin/nologin oumigo
sudo install -d -o oumigo -g oumigo -m 0700 /var/lib/oumigo
sudo install -d -m 0755 /etc/oumigo /opt/oumigo
sudo python3 -m venv /opt/oumigo/venv
sudo /opt/oumigo/venv/bin/pip install "oumigo[manager]"
/opt/oumigo/venv/bin/oumigo version
```

The manager installs no GPU libraries.

### 1.3 nginx and the certificate

```bash
sudo certbot certonly --webroot -w /var/www/html -d oumigo.example.com \
     --email you@example.com --agree-tos -n
```

This uses the default nginx site to answer the challenge on port 80. Reload nginx after
every renewal, so it serves the new certificate. A reload doesn't drop open tunnels:

```bash
printf '#!/bin/sh\nsystemctl reload nginx\n' | \
  sudo tee /etc/letsencrypt/renewal-hooks/deploy/reload-nginx.sh
sudo chmod 755 /etc/letsencrypt/renewal-hooks/deploy/reload-nginx.sh
```

### 1.4 Manager config

Create `/etc/oumigo/manager.yaml`, starting from
[config_examples/manager_CUSTOM_vps.yaml](../config_examples/manager_CUSTOM_vps.yaml),
which is a complete file. The important parts:

```yaml
provider: CUSTOM               # WAN posture: mDNS off, auth enforced, safe binds checked

control_plane:                 # workers: registration, heartbeat, metrics, /tunnel
  host: 127.0.0.1              # nginx terminates TLS and forwards here
  port: 7014
  trusted_proxies: [127.0.0.1]

data_plane:                    # the router: OpenAI-compatible /v1 for clients
  host: 127.0.0.1
  port: 7012
  # client_auth: true          # always on under CUSTOM

dashboard:
  host: 127.0.0.1              # view through an SSH tunnel
  port: 7080

tunnel:
  enabled: true
  pool_spare: 1                # connections kept beyond each worker's capacity

auth:
  token_file: null             # no shared token: workers join with created tokens
  state_dir: /var/lib/oumigo   # signing secret, join tokens, API keys (keep it safe)

catalog:                       # the models (profiles) workers may run in this fleet
  allow:
    - gemma-4-12B-it-qat-w4a16-ct@128k
    - gemma-4-31B-it-qat-w4a16-ct@64k
  default: gemma-4-12B-it-qat-w4a16-ct@128k

worker:
  vllm_port: 7001
```

Under `CUSTOM` the manager refuses to start if the control plane, the router or the
dashboard would be reachable unsafely, or if router client auth is off.

**Models are profiles.** Each allowed profile is one model with every start-time setting
fixed (context length, memory share, concurrency, vLLM flags), and its id is the model id
clients request. Each worker picks the profile that fits its GPU (`OUMIGO_MODEL_PROFILE`,
see [2.2](#22-configure)), or gets the default. OumiGo ships tested profiles; to add your
own, or a variant with another context length, see [model-catalog.md](model-catalog.md).
If vLLM runs out of memory while "Capturing CUDA graphs", the profile's
`gpu_memory_utilization` is too high for that GPU: use or create a profile with a lower one.

### 1.5 Run it as a service

`/etc/systemd/system/oumigo-manager.service`:

```ini
[Unit]
Description=OumiGo manager (control plane + router)
After=network-online.target
Wants=network-online.target

[Service]
User=oumigo
Group=oumigo
WorkingDirectory=/var/lib/oumigo
ExecStart=/opt/oumigo/venv/bin/oumigo manager serve -c /etc/oumigo/manager.yaml --no-mdns -v
Restart=on-failure
RestartSec=5
NoNewPrivileges=true
ProtectSystem=strict
ReadWritePaths=/var/lib/oumigo
PrivateTmp=true

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now oumigo-manager
sudo journalctl -u oumigo-manager -n 20     # expect: "router client auth REQUIRED"
curl -s 127.0.0.1:7014/healthz              # {"status":"ok","nodes":0}
```

### 1.6 Publish it through nginx

Two shared snippets, then the site.

`/etc/nginx/oumigo_proxy.conf` (worker paths → control plane):

```nginx
proxy_pass http://127.0.0.1:7014;
proxy_http_version 1.1;
proxy_set_header Upgrade           $http_upgrade;     # required for /tunnel
proxy_set_header Connection        $connection_upgrade;
proxy_set_header Host              $host;
proxy_set_header X-Forwarded-For   $proxy_add_x_forwarded_for;
proxy_set_header X-Forwarded-Proto $scheme;
proxy_read_timeout 300s;                              # tunnels ping every 20 s
proxy_send_timeout 300s;
```

`/etc/nginx/oumigo_router.conf` (inference paths → router):

```nginx
proxy_pass http://127.0.0.1:7012;
proxy_http_version 1.1;
proxy_set_header Host              $host;
proxy_set_header X-Forwarded-For   $proxy_add_x_forwarded_for;
proxy_set_header X-Forwarded-Proto $scheme;
proxy_buffering off;                                  # stream tokens as they are generated
proxy_read_timeout 600s;                              # long generations
client_max_body_size 64m;                             # image / audio payloads
```

`/etc/nginx/sites-available/oumigo`:

```nginx
map $http_upgrade $connection_upgrade { default upgrade; '' close; }

server {
    listen 80;
    listen [::]:80;
    server_name oumigo.example.com;
    location /.well-known/acme-challenge/ { root /var/www/html; }
    location / { return 301 https://$host$request_uri; }
}

server {
    listen 443 ssl;
    listen [::]:443 ssl;
    server_name oumigo.example.com;
    ssl_certificate     /etc/letsencrypt/live/oumigo.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/oumigo.example.com/privkey.pem;

    location ~ ^/(spec|register|heartbeat|tunnel)$ { include /etc/nginx/oumigo_proxy.conf; }
    location = /metrics                            { include /etc/nginx/oumigo_proxy.conf; }
    location /v1/                                  { include /etc/nginx/oumigo_router.conf; }
    location = /agent-defaults                     { include /etc/nginx/oumigo_router.conf; }
    location / { return 404; }
}
```

The site keeps the ACME path on port 80, so renewals keep working once the default site
is disabled. Enable it:

```bash
sudo ln -s /etc/nginx/sites-available/oumigo /etc/nginx/sites-enabled/oumigo
sudo rm /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl reload nginx
sudo certbot renew --dry-run                # renewal works through the new site
```

On nginx 1.25.1 or later you may add `http2 on;` to the 443 server. Older versions
reject that line, and OumiGo doesn't need HTTP/2.

Check from any other machine. Every answer below is the expected one:

```bash
curl -s -o /dev/null -w '%{http_code}\n' https://oumigo.example.com/spec       # 401: needs a join token
curl -s -o /dev/null -w '%{http_code}\n' https://oumigo.example.com/v1/models  # 401: needs an API key
curl -s -o /dev/null -w '%{http_code}\n' https://oumigo.example.com/workers    # 404: not published
```

### 1.7 Create credentials

All credential commands run on the manager host, as the `oumigo` user, against the same
config file. A shell alias saves typing:

```bash
alias om='sudo -u oumigo /opt/oumigo/venv/bin/oumigo manager'
```

**A join token per worker.** For a machine you control that restarts now and then,
create a long-lived token that can be used again. For a short-lived rented pod, create a
single-use token that expires:

```bash
om token create --ttl never --uses 0 --label gpu-box-1 -c /etc/oumigo/manager.yaml
om token create --ttl 1h --uses 1 --label pod-a -c /etc/oumigo/manager.yaml
om token list -c /etc/oumigo/manager.yaml
```

**An API key per client application:**

```bash
om apikey create --label my-app -c /etc/oumigo/manager.yaml
om apikey list -c /etc/oumigo/manager.yaml
```

Each command prints the secret once, on stdout. Only a hash is stored, so copy it
straight into the worker's or the client's configuration. If it is lost, create a new
one and revoke the old one.

---

## Part 2: a worker on a GPU machine behind NAT

The worker needs no inbound port, so the same steps work on an office or home network,
a cloud GPU instance, or inside a container.

### 2.1 Install

```bash
python3 -m venv ~/oumigo-worker/.venv
~/oumigo-worker/.venv/bin/pip install "oumigo[worker]"     # pulls vLLM + Transformers
```

Do not install `torch` separately: vLLM pins its own version. Activating the virtual
environment is optional. The worker runs vLLM with its own interpreter.

### 2.2 Configure

Create `~/oumigo-worker/.env` (mode 0600; it holds a secret):

```bash
OUMIGO_MANAGER_URL=https://oumigo.example.com
OUMIGO_TRANSPORT=tunnel
OUMIGO_JOIN_TOKEN=oumigo-join-...
OUMIGO_MODEL_PROFILE=gemma-4-31B-it-qat-w4a16-ct@64k
```

`OUMIGO_MODEL_PROFILE` picks the model this worker runs, from the fleet's allowed list.
Leave it out to get the fleet's default. Everything else about the model (context length,
memory share, vLLM flags) comes from the profile on the manager.

Optional settings, in the same file:

| Variable | When to set it |
|---|---|
| `HF_TOKEN` | The model is gated on Hugging Face, or to avoid download rate limits |
| `HF_HOME` | Keep the Hugging Face cache on a large or shared disk |
| `MODEL_STORAGE_LOCATION` | Weights are on local disk (`file:///path`), or `none` to download despite the fleet's setting |
| `OUMIGO_HOST_ID` | Containers whose disk is wiped on restart: any stable string keeps the same node identity |
| `OUMIGO_MANAGER_CA_FILE` | The manager uses a private or self-signed certificate (see [tunnel.md](tunnel.md#without-nginx-built-in-tls)) |

Don't put comments on the same line as a value in `.env`: they become part of the value.

### 2.3 Run

```bash
cd ~/oumigo-worker
.venv/bin/oumigo worker run          # reads ./.env
```

A healthy start looks like this:

```
INFO oumigo: loaded 3 environment variable(s) from .env
INFO oumigo.service.worker: registered with manager https://oumigo.example.com (accepted=True, heartbeat=10s)
INFO oumigo.service.worker.tunnel: tunnel: keeping 2 connection(s) to wss://oumigo.example.com/tunnel
INFO oumigo.service.worker: loading model ... via vllm (this can take minutes)
INFO oumigo.service.worker: vLLM healthy -> SERVING
```

Loading takes a few minutes, longer the first time (download and compilation). Until a
worker is SERVING, clients get `503`.

The worker exchanges its join token for a node token at registration. A long-lived token
lets the same machine re-register after a restart. To run the worker unattended, wrap
the same command in a systemd unit or a container. The worker stops vLLM's whole process
group when it exits, so VRAM is released.

### 2.4 Watch the fleet

The dashboard and the console stay on the manager host. Reach the dashboard through SSH
from your own machine:

```bash
ssh -L 7080:127.0.0.1:7080 user@oumigo.example.com
# then open http://localhost:7080
```

---

## Part 3: clients

Any OpenAI-compatible client works. Give it the base URL and an API key from
[1.7](#17-create-credentials):

```python
from openai import OpenAI

client = OpenAI(base_url="https://oumigo.example.com/v1", api_key="oumigo-key-...")
print([m.id for m in client.models.list()])   # any-model + the models being served
resp = client.chat.completions.create(
    model="gemma-4-12B-it-qat-w4a16-ct@128k",     # a model id, or "any-model"
    messages=[{"role": "user", "content": "Hello"}],
    stream=True,
)
for chunk in resp:
    print(chunk.choices[0].delta.content or "", end="")
```

A model id the fleet doesn't serve answers `404 model_not_found`; a served model whose
workers are all down answers `503 model_unavailable`.

With the OumiGo Python client, pass the server root (without `/v1`), the key as the
token, and the model id (default `any-model`):

```python
from oumigo.api import OumiGoAgent

agent = OumiGoAgent(data_url="https://oumigo.example.com", token="oumigo-key-...",
                    model="gemma-4-12B-it-qat-w4a16-ct@128k")
reply = agent.create_chat().request("Hello")
```

Keep keys out of source code and out of git: read them from the environment or a
secrets store. A key embedded in an app you distribute can be extracted from it, so treat
it as shared by everyone who has the app.

---

## Operating the fleet

| Task | How |
|---|---|
| Revoke a client | `om apikey revoke <id>`, effective from the next request |
| Remove a worker | `om node revoke <node_id>`: the manager drops it and closes its tunnels within seconds; `om node unrevoke` undoes it |
| Retire a join token | `om token revoke <id>`. Workers that already joined keep working |
| See workers | The dashboard ([2.4](#24-watch-the-fleet)), or `curl -s 127.0.0.1:7014/workers` on the manager host |
| Upgrade the manager | `pip install -U` in `/opt/oumigo/venv`, then `systemctl restart oumigo-manager`. Workers reconnect on their own; node tokens survive the restart |
| Add or change a model | Edit `catalog:` (or a profile) and restart the manager; then start or restart the workers that should run it, with `OUMIGO_MODEL_PROFILE` |
| Back up | `/var/lib/oumigo` (signing secret, tokens, keys) and `/etc/oumigo/manager.yaml`. Losing the state directory invalidates every node token and API key |

Keep the manager and the workers on the same OumiGo release. In particular, 0.3.x workers
cannot join a 0.4.x manager.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| Worker log says `discovering via mDNS` | `OUMIGO_MANAGER_URL` isn't set: check that `.env` is in the directory you run from, or pass `--env-file` |
| Worker exits: `requires TLS: use its https:// URL` | A `CUSTOM` manager reached over `http://` |
| Worker exits: `rejected the join token (401)` | Token expired, revoked, already used, or mistyped |
| Worker exits: `certificate verify failed` | The certificate doesn't match the URL's host name, or a private certificate without `OUMIGO_MANAGER_CA_FILE` |
| Worker SERVING, but clients get `503` and the console shows `tunnel 0/0` | nginx is missing the `Upgrade` / `Connection` lines for `/tunnel` |
| vLLM: `CUDA out of memory` while capturing CUDA graphs | The profile's `gpu_memory_utilization` is too high for this GPU: use or create a profile with a lower one |
| Worker exits: `profile … is not allowed in this fleet` | `OUMIGO_MODEL_PROFILE` names a profile missing from `catalog.allow`; the message lists the allowed ones |
| Worker exits: `needs a GPU with at least … GiB` | The profile's `requires.min_vram_gib` is larger than this GPU: pick a smaller profile |
| Client gets `404 model_not_found` | The request's `model` isn't a served model id: see `GET /v1/models`, or send `any-model` |
| Client gets `401 missing or invalid API key` | No `Authorization: Bearer <key>`, a mistyped key, or a revoked key |
| Client gets `503 no healthy (SERVING) workers` | No worker connected, or still loading the model |
| Streamed tokens arrive all at once | `proxy_buffering off` missing from the router location |
| `nginx: unknown directive "http2"` | nginx older than 1.25.1: remove `http2 on;` |
| Manager refuses to start under `CUSTOM` | A bind or auth setting breaks the WAN rules; the error names the setting |

More in [tunnel.md](tunnel.md#troubleshooting).
