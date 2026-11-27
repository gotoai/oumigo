# Environment variables

oumigo reads its own settings from environment variables with a fixed prefix per side:

| Prefix | Read by | Example |
|---|---|---|
| `MG_` | the **manager**: `oumigo manager ...` and its web console | `MG_JOIN_TOKEN` |
| `WK_` | a **worker**: `oumigo worker run` | `WK_MODEL_PROFILE` |

The rules:

1. Every variable oumigo itself defines starts with `MG_` or `WK_`, and is read only by
   that side. A manager ignores `WK_*`, a worker ignores `MG_*`, so one `.env` or shell
   can hold both on a machine that runs both.
2. Variables owned by other software keep their own names (`HF_TOKEN`, `VLLM_*`,
   `CUDA_VISIBLE_DEVICES`, ...). oumigo reads or sets some of them; they are listed in
   [Other software's variables](#other-softwares-variables).
3. Names used before 0.5.8 are **refused**, not ignored: the process exits and names the
   replacement ([Retired names](#retired-names)).

- [Where to set them](#where-to-set-them)
- [Manager variables](#manager-variables)
- [Worker variables](#worker-variables)
- [Other software's variables](#other-softwares-variables)
- [Examples](#examples)
- [Retired names](#retired-names)

## Where to set them

- **The environment**: the shell, systemd `Environment=` / `EnvironmentFile=`, a
  container's `-e` / `--env-file`, a pod's environment.
- **A `.env` file**: `oumigo manager serve|run` and `oumigo worker run` read `.env` from
  the working directory, or the file given with `--env-file`. Templates:
  [`.env.manager.example`](../.env.manager.example) and
  [`.env.worker.example`](../.env.worker.example).

Precedence: a command-line flag wins over the environment, and a variable already in the
environment wins over the `.env` file.

`.env` format: `KEY=VALUE` per line, `#` starts a comment line, an optional leading
`export `, and optional surrounding quotes. **No comments on the same line as a value**:
everything after `=` is the value. Keep the file mode 0600 when it holds a token or hash.

## Manager variables

| Variable | Default | What it does |
|---|---|---|
| `MG_CONFIG_FILE` | the search path | The manager config file (`manager.yaml`). `--config-file` wins. Search path: `~/.config/oumigo/manager.yaml`, then `/etc/oumigo/manager.yaml`. |
| `MG_JOIN_TOKEN` | unset | The reusable join token workers may present (`auth.token_file` holds the same in a file; this wins). Unset under provider `LAN`: worker authentication is off. Under `CUSTOM` workers use minted tokens (`oumigo manager token create`). |
| `MG_ADMIN_USER` | unset | The web console's admin login ([console.md](console.md)). |
| `MG_ADMIN_PASSWORD_HASH` | unset | Its password hash, from `oumigo manager admin-password`. With either of the two unset or empty there is no admin: the console is view-only (Close disabled). |

## Worker variables

| Variable | Default | What it does |
|---|---|---|
| `WK_MANAGER_URL` | found by mDNS on a LAN | The manager's control-plane URL, e.g. `https://oumigo.example.com`. Required outside a LAN. `--manager-url` wins. |
| `WK_JOIN_TOKEN` | unset | The join token from the manager's operator. Spent at registration; the worker then uses its own node token. `--token-file` is the alternative. |
| `WK_TRANSPORT` | `direct` (`tunnel` in the Docker image) | `direct`: the router dials this host (LAN). `tunnel`: the worker dials out to the manager, no inbound port needed. `--transport` wins. |
| `WK_MODEL_PROFILE` | the fleet's default | The catalog profile (= model id) to run, e.g. `gemma-4-12B-it-qat-w4a16-ct@128k`. Several, comma-separated and largest first, share one GPU ([model-catalog.md](model-catalog.md)). |
| `WK_GPU_ID` | `$CUDA_VISIBLE_DEVICES`, else GPU 0 | The GPU(s) to run on: an index from 0 or a `GPU-…` UUID; comma-separated for a model sharded over several. `--gpu-id` wins. |
| `WK_HOST_ID` | a UUID kept in the state directory | A stable node identity, for hosts whose disk is wiped on restart. The Docker image uses `$RUNPOD_POD_ID` when set. |
| `WK_MANAGER_CA_FILE` | unset | A PEM file to trust for a manager with a private (self-signed) certificate. |
| `WK_MODEL_STORAGE_LOCATION` | the profile's, else the Hub | Weights already on this machine: `file:///srv/models/<dir>` (a directory with `config.json`). `none` ignores a fleet-wide location and downloads from the Hub. |
| `WK_DRAIN_TIMEOUT` | the profile's `agent.turn_timeout`, else 120 | Seconds a stopping worker waits for requests in progress ([console.md](console.md#drain-limit)). |
| `WK_DISCOVER_TIMEOUT` | 60 | Seconds to look for a manager by mDNS when `WK_MANAGER_URL` is unset. |
| `WK_METRICS_MAX_KEEPING_AGE` | 60 | Seconds of metrics the worker keeps while its reports fail; older samples are dropped, not sent later. At least 35 (one 30 s report interval plus a 5 s sample) ([metrics.md](metrics.md)). |

## Other software's variables

These keep their owners' names. oumigo reads them, sets a default, or passes them on.

| Variable | Owner | oumigo's use |
|---|---|---|
| `HF_TOKEN` | Hugging Face | Worker: downloads (gated models, rate limits). Empty means anonymous. |
| `HF_HOME` | Hugging Face | Worker: the model cache; `~` and `$VARS` are expanded. |
| `VLLM_CACHE_ROOT` | vLLM | Worker: vLLM's compile cache (not weights). |
| `VLLM_API_KEY` | vLLM | **Set by a tunnel worker** for its backend; don't set it yourself. |
| `VLLM_USE_FLASHINFER_SAMPLER` | vLLM | Worker defaults it to `0`; set `1` on a node whose FlashInfer build works. |
| `VLLM_HTTP_TIMEOUT_KEEP_ALIVE` | vLLM | Worker defaults it to `120` (must outlast the router's keep-alive). |
| `CUDA_VISIBLE_DEVICES` | CUDA | Worker: the GPUs it may use when `WK_GPU_ID` is unset. |
| `FLASHINFER_WORKSPACE_BASE`, `MAX_JOBS` | FlashInfer, build tools | Set in the Docker image. |
| `XDG_CONFIG_HOME`, `XDG_STATE_HOME` | the desktop standard | Config search path and default state directory. |
| `RUNPOD_POD_ID` | RunPod | The Docker image's default `WK_HOST_ID`. |

The pod script ([gpu-pods.md](gpu-pods.md)) reads vendor credentials from its own env
file: `RUNPOD_API_KEY`, `VAST_API_KEY`, `REGISTRY_TOKEN`, `HF_TOKEN`. Neither the manager
nor a worker reads them. Docker build arguments (`OUMIGO_VERSION`, `VLLM_VERSION`,
`VLLM_VARIANT`) exist only while building the image.

## Examples

**A manager on a VPS** (systemd, working directory `/var/lib/oumigo`), in
`/var/lib/oumigo/.env`:

```bash
MG_ADMIN_USER=admin
MG_ADMIN_PASSWORD_HASH=scrypt:16384:8:1:9f0c…:51ab…
```

**A worker on a LAN machine**, in `.env` where you run `oumigo worker run`:

```bash
WK_MANAGER_URL=http://192.168.1.10:7014
WK_JOIN_TOKEN=oumigo-join-…
WK_MODEL_PROFILE=gemma-4-12B-it-qat-w4a16-ct@128k
HF_HOME=/srv/hf-cache
```

**Manager and worker on one machine** can share one `.env`; each reads only its own side:

```bash
MG_JOIN_TOKEN=oumigo-Xq3…
WK_JOIN_TOKEN=oumigo-Xq3…
WK_MODEL_PROFILE=ruri-v3-310m@8k
```

**A worker container** with two embedding models on one GPU:

```bash
docker run --gpus all \
  -e WK_MANAGER_URL=https://oumigo.example.com \
  -e WK_JOIN_TOKEN=oumigo-join-… \
  -e WK_MODEL_PROFILE=harrier-oss-v1-0.6b@32k,ruri-v3-310m@8k \
  -e WK_HOST_ID=embed-1 \
  ghcr.io/<owner>/oumigo-worker:<tag>
```

## Retired names

Set to a non-empty value, these stop the process at startup with a message naming the
replacement. Rename them in every environment and `.env` file.

| Retired name | Side | Use instead |
|---|---|---|
| `OUMIGO_MANAGER_TOKEN` | manager | `MG_JOIN_TOKEN` |
| `OUMIGO_MANAGER_TOKEN` | worker (as its join token) | `WK_JOIN_TOKEN` |
| `OUMIGO_CONFIG_FILE` | manager | `MG_CONFIG_FILE` |
| `OUMIGO_ADMIN_USER` | manager | `MG_ADMIN_USER` |
| `OUMIGO_ADMIN_PASSWORD_HASH` | manager | `MG_ADMIN_PASSWORD_HASH` |
| `OUMIGO_MANAGER_URL` | worker | `WK_MANAGER_URL` |
| `OUMIGO_JOIN_TOKEN` | worker | `WK_JOIN_TOKEN` |
| `OUMIGO_TRANSPORT` | worker | `WK_TRANSPORT` |
| `OUMIGO_MANAGER_CA_FILE` | worker | `WK_MANAGER_CA_FILE` |
| `OUMIGO_HOST_ID` | worker | `WK_HOST_ID` |
| `OUMIGO_MODEL_PROFILE` | worker | `WK_MODEL_PROFILE` |
| `OUMIGO_GPU_ID` | worker | `WK_GPU_ID` |
| `OUMIGO_DISCOVER_TIMEOUT` | worker | `WK_DISCOVER_TIMEOUT` |
| `OUMIGO_DRAIN_TIMEOUT` | worker | `WK_DRAIN_TIMEOUT` |
| `MODEL_STORAGE_LOCATION` | worker | `WK_MODEL_STORAGE_LOCATION` |
| `MODEL_NAME`, `MAX_MODEL_LEN` | worker | removed in 0.5.5: choose a model and context length with `WK_MODEL_PROFILE` |

For contributors: a new variable gets the prefix of the side that reads it, a constant in
`oumigo/common/envnames.py`, and a row here. A rename adds the old name to `RETIRED` in
the same module.
