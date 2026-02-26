import os
import pytest

@pytest.fixture(autouse=True)
def mock_settings_env():
    """
    App config to force when inside ci tests.
    """
    os.environ["ESM2_MOCK_GPU"] = "True"
    os.environ["ESM2_MOCK_GPU_COUNT"] = "1"
    os.environ["ESM2_MAX_SEQUENCE_LENGTH"] = "1022"
