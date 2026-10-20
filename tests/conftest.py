import helpers
import pytest


@pytest.fixture
def write_release():
    return helpers.write_release
