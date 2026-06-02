# mnistRealtime

### EPILEPSY WARNING: This project's realtime demo can produce rapidly flashing images that may trigger seizures in photosensitive individuals. Viewer discretion is advised.

*Disclaimer: this is exclusively vibe-coded and is entirely GPT-5.5's work. The README is GPT-5.5 written.*

Realtime MNIST image generation using the Qwen3-2M MNIST checkpoint and the
optimized C decoder exposed through a small Python wrapper.

## Setup

Use the same conda environment as the training/benchmarking work:

```bash
cd /Users/natebreslow/Documents/mnistRealtime
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mlx-experiment
python -m pip install -r requirements.txt
make
```

`make` builds `libqwen3_mnist_realtime.dylib` from:

- `qwen3_cpu_bench.c`: optimized Qwen3/MNIST CPU decoder
- `qwen3_mnist_realtime.c`: exported C API shim used by Python

The default model is `Qwen3-2M-MNIST-GRPO.raw`.

## `mnist_realtime.py`

Python bindings for the C generator.

```python
from mnist_realtime import MnistGenerator

with MnistGenerator(seed=1234, temperature=0.8, threads=5) as gen:
    image = gen.generate(7)  # numpy uint8 array, shape (28, 28)
```

Main API:

- `MnistGenerator(...)`: loads the raw model and initializes the C runtime.
- `generate(label, seed=None)`: returns one `28x28` `uint8` image for label `0..9`.
- `generate_many(labels, seed=None)`: generates a list of images.
- `set_temperature(value)`: updates sampling temperature in the live generator.
- `set_seed(value)`: resets the generator RNG.

Defaults:

- model: `Qwen3-2M-MNIST-GRPO.raw`
- shared library: `libqwen3_mnist_realtime.dylib`
- threads: `5`
- temperature: `0.8`

## `test_realtime.py`

Small matplotlib smoke test and throughput check.

```bash
python test_realtime.py
```

Useful options:

```bash
python test_realtime.py --calls 10 --temperature 0.8 --seed 1234 --out realtime_10_samples.png
python test_realtime.py --model Qwen3-2M-MNIST-GRPO.raw --threads 5 --calls 100
```

The script prints model load time, images/sec, and tokens/sec, then writes a
sample grid PNG. Labels cycle through `0..9`.

## `realtime_gui.py`

### EPILEPSY WARNING: This project's realtime demo can produce rapidly flashing images that may trigger seizures in photosensitive individuals. Viewer discretion is advised.

Realtime Pygame viewer. It keeps one C-backed generator alive in a background
thread and displays each generated image as it arrives.

```bash
python realtime_gui.py
```

Controls:

- Click digit buttons `0..9` to choose the target label.
- Press number keys `0..9` to choose the target label.
- Press `q` or `Esc` to quit.

Useful options:

```bash
python realtime_gui.py --label 7 --temperature 0.8 --seed 1234
python realtime_gui.py --threads 5 --scale 16 --max-fps 120
python realtime_gui.py --headless-frames 20
```

`--headless-frames` uses SDL's dummy video driver and exits after rendering the
requested number of generated frames. This is useful for quick CI-style checks
without opening a window.

## Files Produced

- `realtime_10_samples.png`: default output from `test_realtime.py`
- `libqwen3_mnist_realtime.dylib`: shared library built by `make`

