#!/usr/bin/env python
"""Does an embedding profile served by vLLM give the model's reference vectors? (GPU)

Embeds a fixed multilingual set (short texts and one long document, as queries and as
documents) twice:

  * **reference** — the model's own sentence-transformers pipeline, in float32, with
    the profile's prompt passed as the ST `prompt`;
  * **served** — `vllm serve` started exactly as a worker would (`VLLMProcess`, i.e.
    `build_argv(profile.node_spec())`), each request shaped by the router's
    `apply_input_type`, so prompts are applied the way the router applies them.

Each served vector must have cosine >= `--min-cos` with its reference. This checks
pooling (e.g. ruri's MEAN vs its config's CLS), prompts, dtype and normalisation-
independence in one go. It then reports the backend's VRAM use and a throughput figure.

Needs a GPU, vLLM and sentence-transformers (not an oumigo dependency):

    pip install sentence-transformers
    python tests/embedding_parity_test.py --profile ruri-v3-310m@8k
    python tests/embedding_parity_test.py --profile ruri-v3-310m@8k --dtype bfloat16
    python tests/embedding_parity_test.py --profile harrier-oss-v1-0.6b@32k
"""

from __future__ import annotations

import argparse
import gc
import subprocess
import sys
import time

import httpx
import numpy as np

from oumigo.config.catalog import load_profiles
from oumigo.service.manager.router.server import apply_input_type
from oumigo.service.worker.supervisor import VLLMProcess

SHORT = [
    "東京駅から京都駅までの新幹線の所要時間は約2時間15分です。",
    "日本の首都はどこですか?",
    "光合成は植物が光エネルギーを化学エネルギーに変換する過程である。",
    "How long does the bullet train take from Tokyo to Kyoto?",
    "Photosynthesis converts light energy into chemical energy in plants.",
    "Die Hauptstadt von Deutschland ist Berlin.",
    "机器学习是人工智能的一个分支。",
    "def add(a, b):\n    return a + b",
    "x",
]
PARAGRAPH = (
    "気候変動は世界各地の生態系や人々の暮らしに影響を与えている。"
    "海面上昇、異常気象の増加、農作物の収量変化などが報告されている。"
    "Climate change affects ecosystems and livelihoods worldwide, with rising sea levels, "
    "more frequent extreme weather and shifting crop yields. "
)


def _texts(long_repeats: int) -> list[str]:
    return SHORT + [PARAGRAPH * long_repeats]


def _cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return (a * b).sum(axis=1)


def reference(model: str, prompts: dict[str, str], texts: list[str]) -> dict[str, np.ndarray]:
    import torch
    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer(model, device="cuda", model_kwargs={"torch_dtype": torch.float32})
    print(f"reference: {model}, max_seq_length={st.max_seq_length}, "
          f"long text = {len(st.tokenize([texts[-1]])['input_ids'][0])} tokens")
    out = {t: st.encode(texts, prompt=p, convert_to_numpy=True, batch_size=4)
           for t, p in prompts.items()}
    del st
    gc.collect()
    torch.cuda.empty_cache()
    return out


def served(client: httpx.Client, model_id: str, prompts: dict[str, str], texts: list[str],
           input_type: str) -> np.ndarray:
    payload = {"model": model_id, "input": texts, "input_type": input_type}
    apply_input_type(payload, prompts)                 # what the router does
    resp = client.post("/v1/embeddings", json=payload)
    resp.raise_for_status()
    data = sorted(resp.json()["data"], key=lambda d: d["index"])
    return np.array([d["embedding"] for d in data], dtype=np.float32)


def _vram_used_mib() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    return int(out.splitlines()[0])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--profile", required=True)
    parser.add_argument("--dtype", help="override the profile's engine.dtype")
    parser.add_argument("--port", type=int, default=7391)
    parser.add_argument("--min-cos", type=float, default=0.999)
    parser.add_argument("--long-repeats", type=int, default=80,
                        help="paragraph repeats in the long document (~60 tokens each)")
    parser.add_argument("--bench-docs", type=int, default=512)
    args = parser.parse_args()

    profile = load_profiles()[args.profile]
    prompts = profile.prompts or {"none": ""}
    texts = _texts(args.long_repeats)
    ref = reference(profile.model, prompts, texts)

    spec = profile.node_spec(args.port)
    if args.dtype:
        spec = spec.model_copy(update={"dtype": args.dtype})
    idle_mib = _vram_used_mib()
    backend = VLLMProcess(spec)
    print("served:", " ".join(backend.argv[2:]))
    backend.start()
    ok = True
    try:
        deadline = time.monotonic() + 900
        while not backend.is_healthy():
            if not backend.running or time.monotonic() > deadline:
                print("backend did not become healthy")
                return 2
            time.sleep(2)
        with httpx.Client(base_url=f"http://127.0.0.1:{args.port}", timeout=300) as client:
            for input_type in prompts:
                got = served(client, spec.served_name, prompts, texts, input_type)
                cos = _cos(got, ref[input_type])
                passed = bool(cos.min() >= args.min_cos)
                ok &= passed
                print(f"{input_type:>9}: dims={got.shape[1]} min cos={cos.min():.6f} "
                      f"mean={cos.mean():.6f} long-doc cos={cos[-1]:.6f} "
                      f"{'OK' if passed else 'FAIL'}")
                norms = np.linalg.norm(got, axis=1)
                print(f"{'':>9}  served norms {norms.min():.4f}..{norms.max():.4f}")

            used = _vram_used_mib() - idle_mib
            docs = [PARAGRAPH * 2 + f" ({i})" for i in range(args.bench_docs)]
            doc_type = "document" if "document" in prompts else next(iter(prompts))
            start = time.monotonic()
            for i in range(0, len(docs), 64):
                served(client, spec.served_name, prompts, docs[i:i + 64], doc_type)
            elapsed = time.monotonic() - start
            print(f"VRAM used by the backend: {used} MiB "
                  f"(gpu_memory_utilization={spec.gpu_memory_utilization})")
            print(f"throughput: {len(docs)} docs of ~250 tokens in {elapsed:.1f}s "
                  f"= {len(docs) / elapsed:.0f} docs/s (batches of 64, one at a time)")
    finally:
        backend.stop()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
