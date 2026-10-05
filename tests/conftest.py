import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.fixture(autouse=True)
def disable_live_summarizer(monkeypatch):
    """No test may silently send an auxiliary/provider request."""
    monkeypatch.setenv("OPTCHAT_SUMMARIZER", "none")


@pytest.fixture
def chat_dir(tmp_path):
    return tmp_path / "optchat"
