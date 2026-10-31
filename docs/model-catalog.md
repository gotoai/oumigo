# Model catalog: several models in one fleet

Since 0.5.0, one OumiGo fleet can serve several models at once. Each worker runs the model
that fits its GPU; clients choose a model by id, and the router sends each request to a
worker running that model.

## The idea: one start-time configuration is one model

vLLM fixes many settings when it starts: context length (`--max-model-len`), memory share,
concurrency, tensor parallelism, quantization flags, parsers. Changing any of them means
restarting the backend. So two workers running the same weights with different start-time
settings are, for every practical purpose, **two different models**.

OumiGo calls one such configuration a **profile**. A profile's **id is the model id**:

- clients put it in `"model"`,
- `GET /v1/models` lists it,
- responses carry it,
- the backend serves under it (`--served-model-name`).

Ids look like `<repo-name>@<context>[-<variant>]`:

| Id | Meaning |
|---|---|
| `gemma-4-12B-it-qat-w4a16-ct@128k` | `google/gemma-4-12B-it-qat-w4a16-ct`, 128k context, defaults otherwise |
| `Qwen3.8-27B-W8A16@128k-c10` | 128k context, 10 concurrent requests |
| `Qwen3.5-9B@32k-bnb4` | 32k context, quantized in flight with bitsandbytes 4-bit |
| `NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4@256k-c8` | 256k context, 8 concurrent requests |

A vendor repo name such as `google/gemma-4-12B-it-qat-w4a16-ct` describes weights, not a
running configuration, so it is **not** a model id.

## Profiles

A profile is a YAML file named `<id>.yaml`:

```yaml
id: gemma-4-31B-it-qat-w4a16-ct@64k         # must equal the file name
model: google/gemma-4-31B-it-qat-w4a16-ct   # the weights: a Hugging Face repo
description: Gemma 4 31B instruct, Google QAT W4A16, 64k context, one request at a time
engine:                                     # start-time settings → `vllm serve`
  max_model_len: 65536
  gpu_memory_utilization: 0.90
  dtype: auto
  tensor_parallel_size: 1
  max_concurrent_requests: 1                # the router's per-worker admission cap
  extra_args:                               # verbatim extra `vllm serve` flags
    - --enable-auto-tool-choice
    - --tool-call-parser
    - gemma4
    - --reasoning-parser
    - gemma4
caps:                                       # enforced by the router, never sent to vLLM
  max_output_tokens: 4096                   # ceiling on max_tokens (also when none is sent)
  max_audio_seconds: null                   # total audio per request; null = no cap
agent:                                      # client timeouts served at /agent-defaults
  turn_timeout: 90.0
  stall_timeout: 120.0
requires:                                   # checked by the worker before it loads anything
  min_vram_gib: 44                          # total VRAM of GPU 0
  vllm: ">=0.28"                            # version specifier
tested: RTX A6000 48 GB, vLLM 0.28.0        # informational
```

Only `id` and `model` are required. Unknown keys are rejected, so a typo stops the manager
at startup instead of reaching a worker minutes later. `agent:` may be left out; the
manager's own `agent:` block then applies.

`requires` turns a slow failure into a fast one: a worker whose GPU is too small exits with
"profile … needs a GPU with at least 44 GiB; GPU 0 has 15.6 GiB" before it downloads or
loads anything.

### Shipped profiles

OumiGo ships the profiles its maintainers have run, in `src/oumigo/catalog/<vendor>/`:

| Id | Weights |
|---|---|
| `gemma-4-12B-it-qat-w4a16-ct@128k` | `google/gemma-4-12B-it-qat-w4a16-ct` |
| `gemma-4-31B-it-qat-w4a16-ct@64k` | `google/gemma-4-31B-it-qat-w4a16-ct` |
| `Muse-Glimmer-30B-W8A16@128k-c10` | `GotoAI-Inc/Muse-Glimmer-30B-W8A16` |
| `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-W8A16@128k-c10` | `GotoAI-Inc/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-W8A16` |
| `Qwen3.8-27B-W8A16@128k-c10` | `GotoAI-Inc/Qwen3.8-27B-W8A16` |
| `NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4@256k-c8` | `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4` |
| `Qwen3.5-9B@32k-bnb4` | `Qwen/Qwen3.5-9B` |

Each file's `tested:` line says what it was verified on. Untested templates to adapt live
in [`config_examples/catalog/`](../config_examples/catalog/); they are not loaded
automatically.

### Your own profiles

Put profile files in a directory and list it in `catalog.dirs`. A profile there with the
same id as a shipped one replaces it (the manager logs which). Two files with the same id
in one directory are an error.

To run a shipped model with a different context or concurrency, copy its file, change the
values, and give it the id that says so, e.g. `gemma-4-12B-it-qat-w4a16-ct@32k-c4`.

## The fleet: `catalog:` in manager.yaml

```yaml
catalog:
  dirs: [/etc/oumigo/catalog]        # optional; shipped profiles are always available
  allow:                             # the profiles workers may run in this fleet
    - gemma-4-12B-it-qat-w4a16-ct@128k
    - gemma-4-31B-it-qat-w4a16-ct@64k
  default: gemma-4-12B-it-qat-w4a16-ct@128k   # for workers that choose nothing

worker:
  vllm_port: 7001                    # backend port workers prefer

agent:                               # fallback client timeouts for profiles without agent:
  turn_timeout: 90.0
  stall_timeout: 120.0
```

- `allow` must name at least one profile, and every name must exist.
- `default` must be in `allow`. Without a default, every worker has to choose.
- The manager logs the allowed profiles and the default at startup.

**Legacy form.** A 0.4-style `model:` block (without `catalog:`) still works: the manager
turns it into a one-profile catalog whose id follows the naming rule
(`<repo-name>@<context>` from `max_model_len`, or `@default` when unset), and makes it the
default. Having both `catalog:` and `model:` is an error.

**Open fleet.** With neither block, the manager has no catalog and accepts any model a
worker brings (`MODEL_NAME`); the worker names its own id the same way. Useful on a
development box, not for a shared fleet.

## Workers: choosing a profile

```bash
OUMIGO_MODEL_PROFILE=gemma-4-31B-it-qat-w4a16-ct@64k
```

Unset, the worker gets the fleet's default. The worker asks the manager for that profile's
settings before it registers, so **the worker carries no model settings of its own**: a
container image needs only the profile id in its environment.

A profile the fleet doesn't allow is refused (`403`) with the list of allowed ids, and the
worker exits with that message.

Settings that describe the **node**, not the model, still come from the worker's
environment:

| Variable | Purpose |
|---|---|
| `MODEL_STORAGE_LOCATION` | Weights on this node's disk (`file:///path`), or `none` to download despite a fleet-wide location |
| `HF_HOME`, `HF_TOKEN` | Hugging Face cache and token (gated models) |

`MAX_MODEL_LEN` can't change a profile: context length is part of it. A worker whose
`MAX_MODEL_LEN` differs from its profile exits; choose or create a profile with that
context instead. `MODEL_NAME` is deprecated: without `OUMIGO_MODEL_PROFILE`, the manager
maps it to the one allowed profile of those weights, and refuses it when there are none
or several.

## Clients: choosing a model

Send a model id, or `any-model`:

```python
from openai import OpenAI

client = OpenAI(base_url="https://oumigo.example.com/v1", api_key="oumigo-key-...")
print([m.id for m in client.models.list()])     # any-model + the ids being served
resp = client.chat.completions.create(
    model="gemma-4-31B-it-qat-w4a16-ct@64k",
    messages=[{"role": "user", "content": "Hello"}],
)
print(resp.model)                               # the id that answered
```

With the OumiGo Python client: `manager.create_agent(model="gemma-4-31B-it-qat-w4a16-ct@64k")`,
or leave `model` out for `any-model`.

| Request `model` | Goes to |
|---|---|
| a model id | workers running that profile |
| `any-model`, or no `model` field | any serving worker; the response names the model that answered |
| anything else, including a vendor repo name | nothing: `404 model_not_found` |

A known id whose workers aren't serving right now (still loading, or switched off for the
night) answers `503 model_unavailable`. Both errors use OpenAI's error format, so SDKs
raise them with their message:

```json
{"error": {"message": "model 'gemma-x' is not served by this fleet; served: gemma-4-12B-it-qat-w4a16-ct@128k, gemma-4-31B-it-qat-w4a16-ct@64k (or any-model)",
           "type": "invalid_request_error", "param": "model", "code": "model_not_found"}}
```

### `GET /v1/models`

The router answers it from its live view of the workers: `any-model` first, then each id
with at least one serving worker. An allowed model with no serving worker is left out, so
apps that fill a menu from this list never offer a model that would only time out.

```json
{"object": "list", "data": [
  {"id": "any-model", "object": "model", "owned_by": "oumigo"},
  {"id": "gemma-4-12B-it-qat-w4a16-ct@128k", "object": "model", "owned_by": "google",
   "oumigo": {"weights": "google/gemma-4-12B-it-qat-w4a16-ct", "max_model_len": 131072, "serving": 1}}
]}
```

`GET /agent-defaults?model=<id>` returns that model's client timeouts.

## Queueing

Each worker accepts up to its profile's `max_concurrent_requests`. When every worker for a
model is busy, its requests wait in the router's queue. A slot that frees goes to the
oldest waiting request that can use it, so a busy model never holds up requests for
another model, and order is kept within each model.

## Upgrading from 0.4

| What | Change |
|---|---|
| `manager.yaml` with `model:` | Keeps working (one-profile catalog). Move to `catalog:` when you add a second model |
| Clients sending a placeholder model (`"oumigo"`, `"any"`) or a repo name | **Now 404.** Send a model id from `/v1/models`, or `any-model` |
| OumiGo Python client | Upgrade with the manager: it now sends `any-model` (or the `model=` you give it) |
| 0.4 workers | Refused by a 0.5 manager with a catalog; upgrade them |
| Workers using `MODEL_NAME` | Still work with a warning; move to `OUMIGO_MODEL_PROFILE` |
| Workers using `MAX_MODEL_LEN` | Refused if it differs from the profile; use a profile with that context |
| `config_examples/manager_<model>.yaml` | Replaced by profiles in `src/oumigo/catalog/` and templates in `config_examples/catalog/` |

Upgrade the manager, the workers and the clients together. A 0.5 manager with a catalog
refuses 0.4 workers at registration, with a message saying to upgrade them: they would
serve under their repo name, which the router never routes to.
