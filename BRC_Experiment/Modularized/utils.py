import os
from typing import Tuple

import numpy as np
import torch


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def configure_determinism(seed: int) -> None:
    """Configure deterministic settings across libraries.

    Must be called once near the start of the program, before CUDA ops.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        # Fallback for older torch versions
        pass

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    np.random.seed(seed)
    torch.manual_seed(seed)


def build_alpha_range(alpha_start: float, alpha_stop: float, alpha_step: float) -> np.ndarray:
    return np.array(np.arange(alpha_start, alpha_stop, alpha_step), dtype=float)


def parse_layer_spec(spec: str | None) -> list[int] | None:
    """Parse a layer spec string into a list of ints.

    Accepts formats:
      - None or "all" -> None (meaning use all layers)
      - comma-separated list: "0,2,5"
      - single range: "3-8" (inclusive of start, exclusive of end like range(start, end))
    """
    if spec is None:
        return None
    spec = spec.strip().lower()
    if spec == "all" or spec == "":
        return None
    if "," in spec:
        return [int(x.strip()) for x in spec.split(",") if x.strip()]
    if "-" in spec:
        start_str, end_str = spec.split("-", 1)
        start, end = int(start_str), int(end_str)
        return list(range(start, end))
    return [int(spec)]


