"""
Pytest configuration and shared fixtures for triton-inference-kernels tests.

Fixtures are organized by scope:
  - session: computed once per test run (device detection, RNG seed)
  - function: fresh tensors per test (avoids state bleeding between tests)

GPU tests are automatically skipped when CUDA is unavailable, using the
`@pytest.mark.gpu` marker. This allows CI to pass on CPU-only machines.
"""

import logging

import pytest
import torch

logger = logging.getLogger(__name__)

# Fixed seed for reproducible test cases
TEST_SEED: int = 42


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers to suppress PytestUnknownMarkWarning.

    Args:
        config: Pytest configuration object.
    """
    config.addinivalue_line("markers", "gpu: marks tests that require a CUDA GPU (skip if none)")
    config.addinivalue_line("markers", "slow: marks tests with long runtime")


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Skip GPU-marked tests when CUDA is not available.

    Args:
        item: The test item about to run.
    """
    if item.get_closest_marker("gpu") and not torch.cuda.is_available():
        pytest.skip("CUDA GPU required — skipping on CPU-only machine")


@pytest.fixture(scope="session")
def device() -> str:
    """Return the best available device for testing.

    Returns:
        'cuda' if a GPU is available, otherwise 'cpu'.
    """
    d = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Test device: %s", d)
    return d


@pytest.fixture(scope="function", autouse=True)
def set_random_seed() -> None:
    """Reset random seeds before each test for reproducibility."""
    torch.manual_seed(TEST_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(TEST_SEED)


@pytest.fixture(scope="function")
def small_softmax_input(device: str) -> torch.Tensor:
    """Small softmax input for fast unit tests.

    Returns:
        Tensor of shape (8, 128), float32, on the test device.
    """
    return torch.randn(8, 128, device=device, dtype=torch.float32)


@pytest.fixture(scope="function")
def large_softmax_input(device: str) -> torch.Tensor:
    """Larger softmax input for testing memory coalescing path.

    Returns:
        Tensor of shape (64, 2048), float32, on the test device.
    """
    return torch.randn(64, 2048, device=device, dtype=torch.float32)


@pytest.fixture(scope="function")
def attention_inputs_small(device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Small attention Q,K,V for correctness tests.

    Returns:
        Tuple (Q, K, V) each of shape (2, 4, 128, 64), float32.
    """
    dtype = torch.float16 if device == "cuda" else torch.float32
    q = torch.randn(2, 4, 128, 64, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    return q, k, v


@pytest.fixture(scope="function")
def attention_inputs_medium(device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Medium attention inputs for seq_len tests.

    Returns:
        Tuple (Q, K, V) each of shape (2, 8, 512, 64), float16 on GPU / float32 on CPU.
    """
    dtype = torch.float16 if device == "cuda" else torch.float32
    q = torch.randn(2, 8, 512, 64, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    return q, k, v
