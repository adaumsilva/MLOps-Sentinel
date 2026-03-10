"""
Shared pytest configuration.
Sets asyncio_mode for the entire test session.
"""


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
