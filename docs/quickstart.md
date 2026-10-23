# Quickstart

Three calls to train, save, and predict.

## 1. Install

```bash
pip install "kanx[all]"
```

## 2. ⚡ One-call magic

```python
import kanx
model = kanx.quickstart()                   # build + train + return
model.predict([[0.5, 0.2]])                 # → array([[1.04…]])
```

## 3. Adaptive Grid Update

During training, refine the B-spline grid based on observed input statistics (recommended for real-world data):

=== "TensorFlow"

    ```python
    from kanx import KAN
    import numpy as np

    X = np.random.randn(1000, 4).astype("float32")
    y = np.sin(X[:, :1]).astype("float32")
    
    model = KAN([4, 16, 8, 1])
    model.fit(X, y, epochs=10, verbose=0)
    
    # Refine grids based on input statistics
    model.update_grid_from_samples(X)
    
    # Continue training with refined grids
    model.fit(X, y, epochs=10, verbose=0)
    ```

=== "PyTorch"

    ```python
    import torch
    from kanx.torch import KAN

    model = KAN([4, 16, 8, 1])
    X = torch.randn(1000, 4)
    y = torch.sin(X[:, :1])
    
    model.fit(X, y, epochs=10, lr=1e-2)
    
    # Refine grids based on input statistics
    model.update_grid_from_samples(X)
    
    # Continue training with refined grids
    model.fit(X, y, epochs=10, lr=1e-2)
    ```

## 4. Your data

=== "TensorFlow"

    ```python
    from kanx import KAN
    import numpy as np

    X = np.random.uniform(-1, 1, (1024, 2)).astype("float32")
    y = (np.sin(np.pi * X[:, :1]) + X[:, 1:2] ** 2).astype("float32")

    model = KAN([2, 64, 64, 1])
    model.fit(X, y, epochs=30, verbose=0)   # auto-compiles
    ```

=== "PyTorch"

    ```python
    import torch
    from kanx.torch import KAN

    model = KAN([2, 64, 64, 1])
    X = torch.randn(1024, 2)
    y = torch.sin(torch.pi * X[:, :1]) + X[:, 1:2] ** 2
    model.fit(X, y, epochs=30, lr=1e-2, val_split=0.1)
    model.save("kan.pt")
    ```

## 5. Serve

```bash
# (option A) Local
uvicorn api.app:app --port 8000

# (option B) Docker
docker run --rm -p 8000:8000 \
    -e KANX_CHECKPOINT=/app/checkpoints/kanx_model.keras \
    -v $(pwd)/checkpoints:/app/checkpoints \
    ghcr.io/mattral/kanx:latest
```

## 6. Monitoring with TensorBoard

When you train with `--tensorboard`, kanx writes events to `logs/kanx` by default.

```bash
python -m kanx train --config configs/default.yaml --tensorboard
tensorboard --logdir logs/kanx
```

Open the TensorBoard UI in your browser to inspect `loss`, `val_loss`, per-layer
`grid` histograms, and `inference_latency_ms`.

## 7. Share your model

After training, publish your KAN model to the HuggingFace Hub and load it
anywhere with a single line.

```python
from kanx import KAN

model = KAN([2, 64, 1])
model(tf.zeros((1, 2)))
model.push_to_hub("username/kanx-demo", private=True)

loaded = KAN.from_pretrained("username/kanx-demo")
```

For the PyTorch backend:

```python
from kanx.torch import KAN

model = KAN([2, 64, 1])
model(torch.zeros((1, 2)))
model.push_to_hub("username/kanx-demo", private=True)

loaded = KAN.from_pretrained("username/kanx-demo")
```

```bash
curl -X POST http://localhost:8000/api/predict \
     -H 'content-type: application/json' \
     -d '{"x": [[0.1, -0.2], [0.5, 0.7]]}'
```

```json
{ "output": [[0.31], [0.84]], "shape": [2, 1], "inference_ms": 22.4 }
```

## 6. MatrixKAN

`MatrixKAN` replaces B-spline recursion with batched matrix multiplies:

```python
from kanx.torch import MatrixKAN
import torch

model = MatrixKAN([8, 32, 1])  # same interface as KAN
X = torch.randn(1024, 8).cuda()
y = model(X)
```

`MatrixKAN` replaces the Cox–de Boor loop with batched GEMMs. The quickstart notebook's saved CUDA cell, hidden width 32 and batch 4096, measured it slower: 13.82 ms/call against 2.23 ms/call for standard `KAN`. The cell prints `Device: cuda` and does not name the GPU. Use standard `KAN` when you need symbolic regression, and time `MatrixKAN` on the width and batch you will serve.

## 7. Export to ONNX

=== "From PyTorch"

    ```python
    from kanx.torch import export_onnx
    export_onnx(model, "kan.onnx", sample_input=torch.zeros(1, 2))
    ```

=== "From TensorFlow"

    ```python
    from kanx import export_onnx_tf
    export_onnx_tf(model, "kan.onnx")
    ```

Both exports include a dynamic batch dimension and have been verified to
produce outputs identical to the eager model within 1e-5.

```python
import onnxruntime as ort, numpy as np
sess = ort.InferenceSession("kan.onnx")
out = sess.run(None, {"input": np.zeros((4, 2), dtype=np.float32)})
```

## Next steps

- [System Design](system_design.md) — KAN architecture, MatrixKAN, grid adaptation
- [Benchmarks](benchmarks.md) — reproducible benchmarking methodology + real-world results
- [Architecture](architecture.md) — library structure and module organization
- [REST API](api.md) — full endpoint reference
- [Deployment](deployment.md) — production rollouts

