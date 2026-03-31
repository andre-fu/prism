"""Start the multi-model inference server.

Usage:
    # From CLI args:
    python -m engine.serve \
        --model Qwen/Qwen2.5-0.5B \
        --model Qwen/Qwen2.5-7B-Instruct \
        --port 8000

    # From config file:
    python -m engine.serve --config engine.yaml
"""

import argparse
from .config import load_config
from .server import run_server


def main():
    parser = argparse.ArgumentParser(description="Multi-model inference server")
    parser.add_argument("--config", help="YAML config file path")
    parser.add_argument("--model", action="append", help="HuggingFace model ID (repeatable)")
    parser.add_argument("--gpu", type=int, nargs="+", default=[0], help="GPU IDs")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    if args.config:
        engine_cfg, sched_cfg, server_cfg = load_config(args.config)
        models = [{"model_id": m.model_id, "name": m.name, "dtype": m.dtype, "tp_size": m.tp_size}
                  for m in engine_cfg.models]
        run_server(
            models,
            gpu_ids=engine_cfg.gpu_ids,
            host=server_cfg.host,
            port=server_cfg.port,
            t0_budget_gb=engine_cfg.t0_budget_gb,
            kv_cache_budget_gb=engine_cfg.kv_cache_budget_gb,
        )
    elif args.model:
        models = []
        for model_id in args.model:
            name = model_id.split("/")[-1]
            models.append({"model_id": model_id, "name": name, "dtype": args.dtype})
        run_server(models, gpu_ids=args.gpu, host=args.host, port=args.port)
    else:
        parser.error("Specify --config or --model")


if __name__ == "__main__":
    main()
