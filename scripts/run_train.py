#!/usr/bin/env python3
"""
run_train.py — Launch LLaMA-Factory training with a torch.mps shim.

Why this wrapper exists
-----------------------
LLaMA-Factory 0.9.x calls ``torch.mps.device_count()`` unconditionally as
part of device detection at startup. That method was only added in newer
PyTorch versions; on older torch builds (notably macOS Intel, which caps
at torch 2.2.2 because PyTorch dropped x86_64 macOS support after that)
the call raises::

    AttributeError: module 'torch.mps' has no attribute 'device_count'

This wrapper monkey-patches ``torch.mps.device_count`` to return 0 when
the method is missing, then hands off to LLaMA-Factory's CLI ``main()``
exactly as if you had run ``llamafactory-cli train <config>`` directly.

The patch is a no-op on systems that already have the method (Linux with
recent torch, Apple Silicon Mac with recent torch). It's safe to use as
the standard training entrypoint everywhere — no harm done when not
needed, makes training actually run on the platforms that need it.

Usage
-----
    python3.11 scripts/run_train.py path/to/cisc187_pt.yaml

With the CPU-tuning launchers documented in README Step 4f::

    ipexrun python3.11 scripts/run_train.py path/to/cisc187_pt.yaml
    numactl --cpunodebind=0 --membind=0 python3.11 scripts/run_train.py path/to/cisc187_pt.yaml
"""

import sys


def _apply_torch_mps_shim() -> None:
    """Add torch.mps.device_count if missing. No-op when already present."""
    try:
        import torch
        if hasattr(torch, "mps") and not hasattr(torch.mps, "device_count"):
            torch.mps.device_count = lambda: 0
    except Exception:
        # If torch can't even be imported, let LLaMA-Factory raise the real error.
        pass


# Apply the shim BEFORE importing llamafactory (which transitively imports torch
# and runs its device-detection logic at module load time on some code paths).
_apply_torch_mps_shim()


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage: python run_train.py <path/to/training_config.yaml>")

    yaml_path = sys.argv[1]

    # Rewrite argv so LLaMA-Factory's CLI sees `llamafactory-cli train <yaml>`.
    sys.argv = ["llamafactory-cli", "train", yaml_path]

    from llamafactory.cli import main as llamafactory_main
    llamafactory_main()


if __name__ == "__main__":
    main()
