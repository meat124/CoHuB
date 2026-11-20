import asyncio
import os
import threading
import ast
import time
import yaml
import importlib
import argparse
import traceback
from baselines.client_server.tcp.model_server import ModelServer


_SEEDING = False   # set by seed_server() before main()


def _default_protocol() -> str:
    """Default to the websocket policy protocol."""
    return "ws"

def eval_function_decorator(policy_model_name, Func_and_Class_name):
    """Load a specified function (e.g., get_model) from a policy module"""
    module = importlib.import_module(policy_model_name)
    return getattr(module, Func_and_Class_name)

def main(deploy_cfg):
    """Main entry: load model, start server, run indefinitely"""
    policy_name = deploy_cfg.get("policy_name")
    port = deploy_cfg.get("port")
    host = deploy_cfg.get("host", "0.0.0.0")
    protocol = deploy_cfg.get("protocol", "ws")

    model_class_func = eval_function_decorator(
        deploy_cfg.get("policy_module", f"baselines.policy.{policy_name}.model"), "Model")
    model = model_class_func(deploy_cfg)
    attach_episode_seeding(model, _SEEDING)

    if protocol == "ws":
        try:
            from baselines.client_server.ws.model_server import PolicyServer, PolicyServerConfig
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "ws policy server requires the baselines package's websocket dependencies "
                f"(missing module: {exc.name}). Install in the policy env with: "
                "pip install -e baselines"
            ) from exc
        server = PolicyServer(
            model,
            PolicyServerConfig(
                host=host,
                port=int(port),
                ws_ping_interval_s=deploy_cfg.get("ws_ping_interval_s", 20.0),
                ws_ping_timeout_s=deploy_cfg.get("ws_ping_timeout_s", 20.0),
            ),
        )
        try:
            asyncio.run(server.serve_forever())
        except KeyboardInterrupt:
            print("\nShutting down websocket policy server...")
        return
    if protocol != "legacy_tcp":
        raise ValueError(f"unsupported policy server protocol: {protocol}")

    # Print exceptions raised inside the server thread.
    def run_server():
        try:
            server.start()
        except Exception:
            print("\033[31m[ERROR] Exception occurred inside server thread:\033[0m")
            traceback.print_exc()
            raise

    server = ModelServer(model, host=host, port=port)
    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down server...")
        server.stop()
        thread.join()

def parse_args_and_config():
    """Parse CLI args and YAML config, merge overrides"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", "--config-path", dest="config_path", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--protocol", choices=("legacy_tcp", "ws"), help="Policy server protocol")
    parser.add_argument("--host", help="Policy server bind host")
    parser.add_argument("--port", type=int, help="Policy server bind port")
    parser.add_argument("--relay-url", dest="relay_url", help="Relay URL for future relay mode")
    parser.add_argument("--overrides", nargs=argparse.REMAINDER, help="Override config values")
    args = parser.parse_args()

    with open(args.config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    # Overrides: `key=value ...` or `--key value ...`

    def _parse_val(s: str):
        try:
            return ast.literal_eval(s)
        except Exception:
            return s
        
    if args.overrides:
        tokens = args.overrides

        if all(("=" in t and not t.startswith("-")) for t in tokens):
            for t in tokens:
                k, v = t.split("=", 1)
                cfg[k] = _parse_val(v)
        else:
            if len(tokens) % 2 != 0:
                raise ValueError(f"--overrides expects key value pairs, got: {tokens}")

            it = iter(tokens)
            for key in it:
                val = next(it)
                cfg[key.lstrip("-")] = _parse_val(val)

    if args.protocol is not None:
        cfg["protocol"] = args.protocol
    else:
        cfg.setdefault("protocol", _default_protocol())
    if args.host is not None:
        cfg["host"] = args.host
    if args.port is not None:
        cfg["port"] = args.port
    if args.relay_url is not None:
        cfg["relay_url"] = args.relay_url

    def _require_non_empty(key: str):
        if key not in cfg:
            raise ValueError(f"{key} must be specified in config or overrides")
        val = cfg[key]
        if val is None:
            raise ValueError(f"{key} must be non-empty")
        if isinstance(val, str) and not val.strip():
            raise ValueError(f"{key} must be non-empty")

    _require_non_empty("host")
    _require_non_empty("port")
    return cfg


def _seed_server_enabled() -> bool:
    """`COHUB_SEED_SERVER`: on unless set to 0/false/no/off.

    When on, the server seeds its RNGs at start and at every `seed_episode`
    call, so an episode's sampling noise depends on its seed alone.
    """
    value = os.environ.get("COHUB_SEED_SERVER", "1").strip().lower()
    return value not in ("0", "false", "no", "off")


def seed_rngs(seed: int) -> None:
    """Seed python, numpy and torch (CPU and every CUDA device) with `seed`."""
    import random

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def seed_server(deploy_cfg) -> bool:
    """Seed this process's RNGs once at start and pin cuDNN to one algorithm.

    Uses `eval_seed` when the config has one, else `seed`. Returns whether
    seeding is on.
    """
    if not _seed_server_enabled():
        print("[server] RNGs not seeded (COHUB_SEED_SERVER=0)")
        return False
    seed = deploy_cfg.get("eval_seed", deploy_cfg.get("seed"))
    if seed is None:
        print("[server] no seed/eval_seed in the config -- seeding from 0")
        seed = 0
    seed = int(seed)
    seed_rngs(seed)
    try:
        import torch

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except ImportError:
        pass
    print(f"[server] RNGs seeded with {seed}; cudnn.benchmark=False, cudnn.deterministic=True (COHUB_SEED_SERVER)")
    return True


def attach_episode_seeding(model, enabled: bool):
    """Give the model a `seed_episode(obs)` RPC that the eval client calls per episode.

    The payload is `{"seed": int}`. When seeding is on it reseeds python, numpy
    and torch, and calls the adapter's `seed(int)` if defined. The reply says
    whether anything was seeded.
    """
    def seed_episode(obs=None):
        if not isinstance(obs, dict) or "seed" not in obs:
            raise ValueError("seed_episode expects {'seed': int}")
        seed = int(obs["seed"])
        if not enabled:
            return {"seeded": False, "seed": seed}
        seed_rngs(seed)
        hook = getattr(model, "seed", None)
        if callable(hook):
            hook(seed)
        return {"seeded": True, "seed": seed}

    model.seed_episode = seed_episode
    return model


if __name__ == "__main__":
    deploy_cfg = parse_args_and_config()
    _SEEDING = seed_server(deploy_cfg)
    main(deploy_cfg)
