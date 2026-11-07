import pytest
from pydantic import ValidationError

from app.config import Settings


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, chunk_size=100, chunk_overlap=100)


def test_cors_origins_are_split_and_trimmed():
    s = Settings(_env_file=None, cors_origins="http://a.com, http://b.com")
    assert s.cors_origins_list == ["http://a.com", "http://b.com"]


def test_empty_cors_falls_back_to_wildcard():
    assert Settings(_env_file=None, cors_origins="").cors_origins_list == ["*"]


def test_log_level_is_normalized():
    assert Settings(_env_file=None, log_level="debug").log_level == "DEBUG"


def test_has_openai_key():
    assert Settings(_env_file=None, openai_api_key="abc").has_openai_key is True
    assert Settings(_env_file=None, openai_api_key="  ").has_openai_key is False


def test_max_upload_bytes():
    assert Settings(_env_file=None, max_upload_mb=2).max_upload_bytes == 2 * 1024 * 1024
