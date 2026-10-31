# Remote workers: CUSTOM provider and tunnel transport

Since 0.4.0 a manager can run on a dedicated non-GPU server (e.g. a VPS) and coordinate
workers anywhere: RunPod, Vast.ai, other clouds, other on-prem sites. Two independent
settings make this work:

| Setting | Question it answers | Values |
|---|---|---|
| `provider` (manager.yaml) | What network does the manager assume? | `LAN` (trusted local network), `CUSTOM` (any vendor or site; no vendor-specific processing) |
| transport (per worker) | How does the router reach the worker's backend? | `direct` (router dials the worker), `tunnel` (worker dials out to the manager) |

They combine freely. A `CUSTOM` fleet can include LAN workers on `direct`, and a LAN
worker may opt into `tunnel`.

## How the tunnel works

A `tunnel` worker needs **no inbound port, no VPN, no root and no TUN device**. That
makes it work inside RunPod and Vast containers. It keeps `capacity + 1` outbound
WebSocket connections open to the manager's `/tunnel` endpoint. The manager parks them,
and the router borrows one per upstream connection. Each WebSocket carries exactly one
plain HTTP/1.1 connection to the worker's backend, unmodified, so:

- SSE streaming passes straight through;
- a lost packet stalls only its own stream, because there is no multiplexing;
- when a client disconnects, the request is cancelled all the way down to vLLM.

The worker's backend binds `127.0.0.1` and requires a per-node API key that only the
manager's router presents.

Measured on one machine with a zero-latency stub backend, router p50 per request: direct
2.3 ms, tunnel over `ws://` 3.0 ms, tunnel over TLS 3.4 ms. Over a WAN, add about one
round trip to the time to first token, the same as any other transport.

## What `CUSTOM` changes

| | `LAN` | `CUSTOM` |
|---|---|---|
| mDNS advertising | on | off. Workers need `OUMIGO_MANAGER_URL` |
| Control plane bind | anything | `127.0.0.1` behind nginx, or built-in TLS (`control_plane.tls_*`) |
| Router (`data_plane`) bind | anything | `127.0.0.1` only. Remote clients reach it through nginx |
| Router client auth (`data_plane.client_auth`) | off unless set | **always on**. Clients send a client API key |
| Dashboard bind | anything | `127.0.0.1` only. View it through an SSH tunnel |
| Worker auth with no reusable token | disabled (warning) | **enforced**. Workers join with minted tokens |
| Manager URL a worker accepts | `http://` or `https://` | `https://` only |

The manager refuses to start with a `CUSTOM` layout that breaks these rules.

## Deploying on a VPS with nginx

### 1. Manager config

```yaml
provider: CUSTOM

control_plane:
  host: 127.0.0.1            # nginx terminates TLS and forwards here
  port: 7014
  trusted_proxies: [127.0.0.1]   # whose X-Forwarded-For is believed (nginx)

data_plane:
  host: 127.0.0.1            # router: nginx forwards /v1/ here for remote clients
  port: 7012
  # client_auth: true        # always on under CUSTOM; clients need an API key

dashboard:
  host: 127.0.0.1            # view through an SSH tunnel
  port: 7080

tunnel:
  enabled: true
  pool_spare: 1              # connections kept beyond each worker's capacity

auth:
  token_file: null           # optional reusable join token; per-pod tokens via CLI
  state_dir: /var/lib/oumigo # signing secret, minted tokens, revoked nodes

worker_allowlist:
  enabled: false
  addresses: []              # e.g. ["203.0.113.7", "198.51.100.0/24"]

model:
  # ... as in config_examples/
```

See `config_examples/manager_CUSTOM_vps.yaml` for a complete file.

### 2. nginx

Forward the worker-facing paths to the control plane and the inference paths to the
router. The read-only endpoints (`/workers`, `/status`, `/metrics/latest`,
`/metrics/since`) and the dashboard stay unexposed.

```nginx
map $http_upgrade $connection_upgrade { default upgrade; '' close; }

server {
    listen 443 ssl;
    server_name oumigo.example.net;
    ssl_certificate     /etc/letsencrypt/live/oumigo.example.net/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/oumigo.example.net/privkey.pem;

    location ~ ^/(spec|register|heartbeat|tunnel)$ { include oumigo_proxy.conf; }
    location = /metrics                            { include oumigo_proxy.conf; }
    location /v1/                                  { include oumigo_router.conf; }
    location = /agent-defaults                     { include oumigo_router.conf; }
    location / { return 404; }
}
```

`oumigo_proxy.conf`:

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

`oumigo_router.conf` (the router checks the client API key itself):

```nginx
proxy_pass http://127.0.0.1:7012;
proxy_http_version 1.1;
proxy_set_header Host              $host;
proxy_set_header X-Forwarded-For   $proxy_add_x_forwarded_for;
proxy_set_header X-Forwarded-Proto $scheme;
proxy_buffering off;                                  # stream SSE tokens as they come
proxy_read_timeout 600s;                              # long generations
client_max_body_size 64m;                             # audio/image payloads
```

- Certificates come from certbot / Let's Encrypt. `nginx -s reload` after a renewal is
  graceful: open tunnels are not dropped.
- Without the `Upgrade`/`Connection` lines, `/tunnel` cannot upgrade and tunnel workers
  never become routable.
- After the upgrade nginx only passes bytes, so its response buffering and
  `client_max_body_size` do not affect tunnel traffic.

### 3. Join a worker

On the manager host, mint a token per pod:

```bash
oumigo manager token create --ttl 1h --uses 1 --label runpod-a   # prints the token
oumigo manager token list
```

On the worker (`.env` or environment):

```
OUMIGO_MANAGER_URL=https://oumigo.example.net
OUMIGO_TRANSPORT=tunnel
OUMIGO_JOIN_TOKEN=oumigo-join-...
```

Then run `oumigo worker run` as usual.

- The join token is spent at registration. From then on the worker uses its own
  **node token**, bound to its `node_id` and accepted across manager restarts.
- For workers you restart often, mint a token with `--uses 0 --ttl never`, or set a
  reusable `auth.token_file`.
- A pod whose disk is wiped on restart can keep its identity with `OUMIGO_HOST_ID=<any
  stable string>`. Otherwise it registers as a new node.

### 4. Give a client an API key

On the manager host, create one key per client:

```bash
oumigo manager apikey create --label kari-dev    # prints the key
oumigo manager apikey list
oumigo manager apikey revoke <id>                # refused from the next request on
```

The client calls `https://oumigo.example.net/v1/...` with `Authorization: Bearer <key>`.
Any OpenAI-compatible client works with its `api_key` setting. For the OumiGo Python
client, pass the key as the agent's `token`. Keys are stored only as hashes, next to the
join tokens.

### 5. Remove a worker

```bash
oumigo manager node revoke <node_id>     # node_id: see `workers` in the console
```

Within seconds the manager drops the node and closes its tunnels. The worker's next
heartbeat gets 403, and it drains and shuts down. `oumigo manager node unrevoke` lifts
the revocation.

## Without nginx: built-in TLS

```yaml
control_plane:
  host: 0.0.0.0
  port: 7014
  tls_cert_file: /etc/oumigo/tls/fullchain.pem
  tls_key_file:  /etc/oumigo/tls/privkey.pem
```

- With a private (self-signed) certificate, give workers the PEM:
  `OUMIGO_MANAGER_CA_FILE=/path/cert.pem`. The certificate's subjectAltName must match the
  host in `OUMIGO_MANAGER_URL`. Workers never skip verification.
- uvicorn cannot reload certificates, so renewing them means restarting the manager. That
  drops every tunnel; workers reconnect within seconds.

## Credentials at a glance

| Credential | Where it lives | Lifetime |
|---|---|---|
| Join token (`OUMIGO_JOIN_TOKEN`) | Operator → worker | Minted: TTL and use count. Reusable: until rotated |
| Node token | Issued at registration, held by the worker | Until the node is revoked |
| Backend API key | Derived per node; the worker's backend requires it | Same as the node |
| Client API key | Operator → inference client; the router requires it | Until revoked |

Minted tokens and client API keys are stored only as hashes, in `<state_dir>/manager_auth.db` (mode 0600),
next to the signing secret. Treat anything on a rented pod as readable by the host
operator. That is why tokens are per pod, short-lived, and revocable per node.

## Worker IP allowlist

`worker_allowlist.enabled: true` limits every worker-facing endpoint to the listed
addresses and ranges. Rejected addresses are logged, so a new pod's address can be
added. Changing the list currently means editing `manager.yaml` and restarting the
manager.

The allowlist works well for sites with fixed IPs. RunPod and Vast pods connect out
through datacenter NAT, so their address isn't known before launch. The allowlist is a
layer on top of join tokens, never a replacement.

Repeated auth failures from one address (30 within 60 s) are answered with 429 until
they age out.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| Worker exits: `rejected the join token (401)` | Token expired, revoked, already used, or mistyped |
| Worker exits: `certificate verify failed` | Private certificate without `OUMIGO_MANAGER_CA_FILE`, or the URL host isn't in the certificate's subjectAltName |
| Worker exits: `requires TLS: use its https:// URL` | `CUSTOM` manager reached over `http://` |
| Worker SERVING but `tunnel 0/0` in the console | nginx missing the WebSocket `Upgrade` lines, or `tunnel.enabled: false` |
| `tunnel slot N refused by manager (HTTP 403)` briefly after a manager restart | Normal: tunnels redial before the worker re-registers |
| Client gets `401 missing or invalid API key` | No `Authorization: Bearer <key>`, a mistyped key, or the key was revoked (`oumigo manager apikey list`) |
| Streamed tokens arrive all at once through nginx | `proxy_buffering off` missing from the router location |
