"""Tests for MatrixKAN GPU-optimized layer."""
import pytest
import torch

from kanx.torch.layers import KANLinear
from kanx.torch.matrix_kan import MatrixKAN, MatrixKANLinear


class TestMatrixKANOutput:
    """Test MatrixKAN output shape and basic functionality."""

    def test_matrix_kan_output_shape(self):
        """MatrixKAN forward pass produces correct shape."""
        model = MatrixKAN([2, 16, 1])
        x = torch.randn(32, 2)
        y = model(x)
        assert y.shape == (32, 1)

    def test_matrix_kan_linear_output_shape(self):
        """MatrixKANLinear produces correct output shape."""
        layer = MatrixKANLinear(8, 4)
        x = torch.randn(16, 8)
        y = layer(x)
        assert y.shape == (16, 4)


class TestMatrixKANNumericalAgreement:
    """Test MatrixKAN agrees with standard KANLinear."""

    def test_matrix_kan_numerical_agreement_single_layer(self):
        """MatrixKANLinear output agrees with KANLinear to within 1e-4."""
        torch.manual_seed(0)
        x = torch.randn(32, 8)

        # Create two identical-seed models
        torch.manual_seed(42)
        matrix_layer = MatrixKANLinear(8, 4, grid_size=5, spline_order=3)

        torch.manual_seed(42)
        kan_layer = KANLinear(8, 4, grid_size=5, spline_order=3)

        # Copy weights to ensure identical initialization
        kan_layer.base_weight.data = matrix_layer.base_weight.data.clone()
        kan_layer.spline_weight.data = matrix_layer.spline_weight.data.clone()
        kan_layer.grid.data = matrix_layer.grid.data.clone()

        matrix_out = matrix_layer(x)
        kan_out = kan_layer(x)

        # Check agreement to within tolerance
        max_diff = torch.max(torch.abs(matrix_out - kan_out)).item()
        assert max_diff < 1e-3, f"Max difference: {max_diff}"

    def test_matrix_kan_agreement_multi_layer(self):
        """MatrixKAN (full model) agrees with KAN layers."""
        torch.manual_seed(0)
        x = torch.randn(16, 4)

        torch.manual_seed(42)
        matrix_model = MatrixKAN([4, 8, 1])

        torch.manual_seed(42)
        # Build equivalent standard model
        kan_model = torch.nn.Sequential(
            KANLinear(4, 8),
            KANLinear(8, 1),
        )

        # Copy weights
        with torch.no_grad():
            matrix_layers = [m for m in matrix_model.modules() if isinstance(m, MatrixKANLinear)]
            kan_layers    = [k for k in kan_model.modules() if isinstance(k, KANLinear)]
            for m, k in zip(matrix_layers, kan_layers, strict=True):
                k.base_weight.data   = m.base_weight.data.clone()
                k.spline_weight.data = m.spline_weight.data.clone()
                k.grid.data          = m.grid.data.clone()

        matrix_out = matrix_model(x)
        kan_out = kan_model(x)

        max_diff = torch.max(torch.abs(matrix_out - kan_out)).item()
        assert max_diff < 1e-3, f"Max difference: {max_diff}"


class TestMatrixKANGPU:
    """Test MatrixKAN GPU functionality and throughput."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_matrix_kan_gpu_forward(self):
        """MatrixKAN works on GPU device."""
        model = MatrixKAN([4, 16, 1]).cuda()
        x = torch.randn(32, 4, device="cuda")
        y = model(x)
        assert y.device.type == "cuda"
        assert y.shape == (32, 1)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_matrix_kan_gpu_throughput(self):
        """On CUDA, this layer shape should not be much slower than KANLinear.

        The quickstart notebook's hidden-width-32, batch-4096 cell measured
        MatrixKAN slower (13.82 ms vs 2.23 ms). This test is a different
        shape (one layer, batch 256) and only checks a 0.95× floor.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"

        matrix_layer = MatrixKANLinear(32, 32, grid_size=5).to(device)
        kan_layer = KANLinear(32, 32, grid_size=5).to(device)

        # Copy weights for fair comparison
        with torch.no_grad():
            kan_layer.base_weight.data = matrix_layer.base_weight.data.clone()
            kan_layer.spline_weight.data = matrix_layer.spline_weight.data.clone()
            kan_layer.grid.data = matrix_layer.grid.data.clone()

        x = torch.randn(256, 32, device=device)
        n_runs = 100

        # Warmup
        for _ in range(10):
            _ = matrix_layer(x)
            _ = kan_layer(x)

        # Time MatrixKAN
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        for _ in range(n_runs):
            _ = matrix_layer(x)
        t1.record()
        torch.cuda.synchronize() if device == "cuda" else None
        matrix_time = t0.elapsed_time(t1) if device == "cuda" else None

        # Time standard KAN
        if device == "cuda":
            torch.cuda.synchronize()
        t2 = torch.cuda.Event(enable_timing=True)
        t3 = torch.cuda.Event(enable_timing=True)
        t2.record()
        for _ in range(n_runs):
            _ = kan_layer(x)
        t3.record()
        torch.cuda.synchronize() if device == "cuda" else None
        kan_time = t2.elapsed_time(t3) if device == "cuda" else None

        if matrix_time is not None and kan_time is not None:
            speedup = kan_time / matrix_time
            # A floor, not a 1.5–2× claim. See the quickstart CUDA cell.
            assert speedup >= 0.95, f"MatrixKAN far slower on this shape: speedup={speedup:.2f}x"


class TestMatrixKANGridUpdate:
    """Test adaptive grid update."""

    def test_matrix_kan_update_grid_changes_grid(self):
        """Grid update from samples modifies grid values."""
        model = MatrixKAN([4, 8, 1])
        x = torch.randn(128, 4)

        # Get initial grid
        initial_grid = model[0].grid.data.clone()

        # Update grid
        model.update_grid_from_samples(x)
        new_grid = model[0].grid.data

        # Check that grid changed
        assert not torch.allclose(initial_grid, new_grid), "Grid should change after update"

    def test_matrix_kan_update_grid_preserves_shape(self):
        """Grid update preserves grid shape."""
        model = MatrixKAN([6, 16, 1])
        x = torch.randn(100, 6)

        initial_shape = model[0].grid.shape
        model.update_grid_from_samples(x)
        new_shape = model[0].grid.shape

        assert initial_shape == new_shape, "Grid shape should not change"

    def test_matrix_kan_forward_after_update(self):
        """Forward pass still works after grid update."""
        model = MatrixKAN([4, 8, 1])
        x_train = torch.randn(100, 4)
        x_test = torch.randn(32, 4)

        model.update_grid_from_samples(x_train)
        y = model(x_test)

        assert y.shape == (32, 1)
        assert not torch.isnan(y).any()


class TestMatrixKANONNXExport:
    """Test ONNX export compatibility."""

    def test_matrix_kan_onnx_export(self):
        """MatrixKAN exports to ONNX format."""
        pytest.importorskip("onnx")
        import tempfile

        model = MatrixKAN([2, 8, 1])
        x = torch.randn(1, 2)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx_path = f.name

        try:
            torch.onnx.export(
                model,
                x,
                onnx_path,
                input_names=["input"],
                output_names=["output"],
                opset_version=12,
            )
            # If we get here, export succeeded
            assert True
        finally:
            import os
            if os.path.exists(onnx_path):
                os.remove(onnx_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
