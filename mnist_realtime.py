from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_LIB = ROOT / "libqwen3_mnist_realtime.dylib"
DEFAULT_MODEL = ROOT / "Qwen3-2M-MNIST-GRPO.raw"


class MnistGenerator:
    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL,
        lib_path: str | Path = DEFAULT_LIB,
        *,
        threads: int = 5,
        temperature: float = 0.8,
        seed: int = 1,
    ):
        self.model_path = Path(model_path)
        self.lib_path = Path(lib_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Missing raw model: {self.model_path}")
        if not self.lib_path.is_file():
            raise FileNotFoundError(
                f"Missing shared library: {self.lib_path}. Run `make` first."
            )

        self._lib = ctypes.CDLL(str(self.lib_path))
        self._lib.mnist_create.argtypes = [
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_uint32,
        ]
        self._lib.mnist_create.restype = ctypes.c_void_p
        self._lib.mnist_destroy.argtypes = [ctypes.c_void_p]
        self._lib.mnist_destroy.restype = None
        self._lib.mnist_generate.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_ubyte),
        ]
        self._lib.mnist_generate.restype = ctypes.c_int
        self._lib.mnist_generate_with_seed.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_ubyte),
        ]
        self._lib.mnist_generate_with_seed.restype = ctypes.c_int
        self._lib.mnist_set_temperature.argtypes = [ctypes.c_void_p, ctypes.c_float]
        self._lib.mnist_set_temperature.restype = None
        self._lib.mnist_set_seed.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        self._lib.mnist_set_seed.restype = None

        self._handle = self._lib.mnist_create(
            str(self.model_path).encode("utf-8"),
            int(threads),
            float(temperature),
            int(seed) & 0xFFFFFFFF,
        )
        if not self._handle:
            raise RuntimeError("mnist_create returned NULL")

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self._lib.mnist_destroy(self._handle)
            self._handle = None

    def __enter__(self) -> "MnistGenerator":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def set_temperature(self, temperature: float) -> None:
        self._lib.mnist_set_temperature(self._handle, float(temperature))

    def set_seed(self, seed: int) -> None:
        self._lib.mnist_set_seed(self._handle, int(seed) & 0xFFFFFFFF)

    def generate(self, label: int, *, seed: int | None = None) -> np.ndarray:
        if self._handle is None:
            raise RuntimeError("Generator is closed")
        if not 0 <= int(label) <= 9:
            raise ValueError("label must be in 0..9")

        image = np.empty((28, 28), dtype=np.uint8)
        ptr = image.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
        if seed is None:
            rc = self._lib.mnist_generate(self._handle, int(label), ptr)
        else:
            rc = self._lib.mnist_generate_with_seed(
                self._handle, int(label), int(seed) & 0xFFFFFFFF, ptr
            )
        if rc == -1:
            raise RuntimeError("mnist_generate failed")
        return image

    def generate_many(self, labels, *, seed: int | None = None) -> list[np.ndarray]:
        images = []
        for i, label in enumerate(labels):
            image_seed = None if seed is None else int(seed) + i
            images.append(self.generate(int(label), seed=image_seed))
        return images
