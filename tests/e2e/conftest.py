import os

import pytest

URL = os.environ.get("VIGILEX_E2E_URL")


@pytest.fixture(scope="session")
def base_url():
    if not URL:
        pytest.skip("VIGILEX_E2E_URL not set (run `make app` then `make e2e`)")
    return URL
