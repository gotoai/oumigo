# Embedding models

Since 0.5.4, a fleet can serve embedding models next to chat models. An embedding worker
runs vLLM's pooling runner. Clients call the OpenAI-compatible `POST /v1/embeddings` on the
router, and the router keeps chat and embedding traffic apart.

Splitting documents into chunks is the caller's job: OumiGo embeds what it is sent.

## Profiles: `task: embed`

A profile says what its model is for with `task`: `chat` (the default) or `embed`. An
embedding profile may add an `embed:` block:

```yaml
id: ruri-v3-310m@8k
model: cl-nagoya/ruri-v3-310m
description: Ruri v3 310M, Japanese text embeddings (768-d), 8k context
task: embed                         # served with `vllm serve ... --runner pooling`
engine:
  max_model_len: 8192
  gpu_memory_utilization: 0.2
  dtype: float16
  max_concurrent_requests: 8
  extra_args:
  - --pooler-config
  - '{"seq_pooling_type":"MEAN"}'
embed:
  dimensions: 768                   # informational; shown in /v1/models
  prompts:                          # input_type -> prefix the router adds to each input
    query: '検索クエリ: '
    document: '検索文書: '
    topic: 'トピック: '
```

- **Prompt names** are lowercase identifiers. An empty prefix is a valid type that adds
  nothing; harrier, for example, embeds documents as they are.
- **Chat-only caps** (`max_output_tokens`, `max_audio_seconds`) are refused on an embed
  profile.
- **Backend:** an embedding profile needs the vLLM backend. A worker started with
  `--backend transformer` exits with a message saying so.

### Shipped embedding profiles

| Id | Weights | Language | Dims | Vectors | VRAM used | `tested` |
|---|---|---|---|---|---|---|
| `ruri-v3-310m@8k` | `cl-nagoya/ruri-v3-310m` | Japanese | 768 | **not normalised** | 1.5 GiB | RTX 3080 Ti Laptop 16 GB |
| `harrier-oss-v1-0.6b@8k` | `microsoft/harrier-oss-v1-0.6b` | multilingual | 1024 | unit length | 2.8 GiB | RTX 3080 Ti Laptop 16 GB |
| `harrier-oss-v1-0.6b@32k` | `microsoft/harrier-oss-v1-0.6b` | multilingual | 1024 | unit length | 6.5 GiB | RTX 3080 Ti Laptop 16 GB |

**How they were checked.** Each was compared with the model's own sentence-transformers
pipeline (`tests/embedding_parity_test.py`) on short and long multilingual text, for every
prompt. The cosine with the reference vector was at least 0.99994 for every input.

**Profile notes:**
- **ruri pooling:** ruri's `config.json` says CLS pooling, but the model is used with mean
  pooling. The profile pins MEAN.
- **ruri dtype:** ruri runs in float16, which matches the reference to a cosine of
  0.99999. bfloat16 drops to 0.9989.
- **harrier context:** use `harrier-oss-v1-0.6b@8k` unless you embed very long texts. The
  32k profile needs 3.5 GiB of KV cache for one full-length input.

**Normalisation.** ruri's vectors are returned unnormalised (norms around 28), as the
model's own pipeline returns them. Compare them with **cosine** similarity, or normalise
them yourself before using inner product. harrier's vectors are already unit length.

Several small workers can share one GPU: each worker picks its own port. On a 16 GB GPU, a
small chat model, `ruri-v3-310m@8k` and `harrier-oss-v1-0.6b@8k` run together. Start
them one after another, because each vLLM checks for its memory share when it starts.

## Calling `/v1/embeddings`

```python
from openai import OpenAI

client = OpenAI(base_url="https://oumigo.example.com/v1", api_key="oumigo-key-...")
docs = client.embeddings.create(
    model="ruri-v3-310m@8k",
    input=["東京は日本の首都です。", "富士山は日本一高い山です。"],
    extra_body={"input_type": "document"},
)
query = client.embeddings.create(
    model="ruri-v3-310m@8k", input="日本の首都は?", extra_body={"input_type": "query"})
```

### `input_type`

**What it does.** `input_type` is an OumiGo extension. The router:
1. looks the value up in the profile's `embed.prompts`;
2. puts that prefix before each input string;
3. removes the field before the request reaches vLLM.

**Why use it.** A client that sends `query` and `document` works with every embedding
model, without knowing each model's instructions. The prefix counts towards the model's
context length.

**Without `input_type`** the input is sent unchanged, so you can add your own prefixes.

**Other fields** are passed to vLLM as they are, e.g. `encoding_format: "base64"`,
`dimensions`, `truncate_prompt_tokens`.

**Over-length input.** An input longer than the model's context is refused with vLLM's
`400`, never cut short silently. Pass `truncate_prompt_tokens` if you want truncation.

`GET /v1/models` lists each model's task and, for embedding models, its dimensions and
valid `input_type` names:

```json
{"id": "ruri-v3-310m@8k", "object": "model", "owned_by": "cl-nagoya",
 "oumigo": {"weights": "cl-nagoya/ruri-v3-310m", "task": "embed", "max_model_len": 8192,
            "dimensions": 768, "input_types": ["document", "query", "topic"], "serving": 1}}
```

### Routing and errors

| Request | Result |
|---|---|
| `/v1/embeddings` naming an embedding model | a worker running it |
| `/v1/embeddings` with `any-model` or no `model` | `400 model_required`: vectors of two models never mix, so name one |
| `/v1/embeddings` naming a chat model | `400 wrong_task` |
| chat or completions naming an embedding model | `400 wrong_task` (`<model> is an embedding model; use /v1/embeddings.`) |
| chat or completions with `any-model` | chat workers only; never an embedding worker |
| an `input_type` the model has no prompt for | `400 invalid_input_type` (the message lists the valid ones) |
| `input_type` with token-id input | `400 invalid_input_type` |

**`any-model` refusal.** `any-model` is refused for embeddings on purpose. Two
embedding models put text in different vector spaces. A random pick between them would
corrupt an index without any error.

The `503` codes (`model_loading`, `model_unavailable`, `model_busy`) work as for chat
models; see [model-catalog.md](model-catalog.md).

Each request takes one worker slot, however many inputs it carries; vLLM batches the
inputs inside it. Send documents in batches (e.g. 32–128 per request) rather than one by
one.

## Upgrading

- **Manager first.** Upgrade the manager before starting any embedding worker. An older
  manager does not know `task` and could send `any-model` chat to an embedding worker.
- **Older workers** register as chat workers, which is what they are.
- **Clients** need no change; `/v1/models` entries gain a `task` field.
