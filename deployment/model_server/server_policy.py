# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].
"""WebSocket policy server.

Loads a trained checkpoint and serves ``predict_action`` requests over a WebSocket
(see ``tools/websocket_policy_server.py``); the LIBERO / LIBERO-Plus evaluation clients
connect to it with ``tools/websocket_policy_client.py``. Run from the repository root:

    python deployment/model_server/server_policy.py \\
        --ckpt_path playground/Checkpoints/<run_id>/checkpoints/steps_20000_pytorch_model.pt \\
        --port 10093 --device cuda --use_bf16
"""

import argparse
import logging
import socket
from pathlib import Path

from deployment.model_server.policy_wrapper import PolicyServerWrapper
from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer


def main(args) -> None:
    """Build the policy wrapper and start the WebSocket server.

    The wrapper owns action un-normalization and chunk-size discovery, so evaluation
    clients only forward ``examples`` and receive already-unnormalized actions.
    """
    if not Path(args.ckpt_path).is_file():
        raise SystemExit(f"--ckpt_path {args.ckpt_path!r} is not a file; pass a .pt or .safetensors checkpoint.")

    wrapper = PolicyServerWrapper(
        ckpt_path=args.ckpt_path,
        device=args.device,
        use_bf16=args.use_bf16,
        unnorm_key=args.unnorm_key,
    )

    hostname = socket.gethostname()
    logging.info("Creating server (host: %s, port: %d)", hostname, args.port)

    # wrapper.metadata is sent to every client at handshake.
    server = WebsocketPolicyServer(
        policy=wrapper,
        host=args.host,
        port=args.port,
        idle_timeout=args.idle_timeout,
        metadata=wrapper.metadata,
    )
    logging.info("Server running ... metadata=%s", wrapper.metadata)
    server.serve_forever()


def build_argparser():
    parser = argparse.ArgumentParser(description="Serve a starVLA checkpoint over WebSocket.")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Checkpoint file (<run_dir>/checkpoints/steps_*_pytorch_model.pt); "
        "config.yaml and dataset_statistics.json are read from <run_dir>.",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Interface to bind.")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--device", type=str, default="cuda", help="Torch device for the model, e.g. cuda or cuda:1.")
    parser.add_argument("--use_bf16", action="store_true", help="Cast the model to bfloat16.")
    parser.add_argument(
        "--unnorm_key",
        type=str,
        default=None,
        help="Top-level key of dataset_statistics.json used for un-normalization (auto when there is one).",
    )
    parser.add_argument("--idle_timeout", type=int, default=1800, help="Idle timeout in seconds, -1 means never close")
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(build_argparser().parse_args())
