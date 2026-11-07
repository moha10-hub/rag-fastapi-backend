import pytest
from pydantic import ValidationError

from app.schemas import QueryRequest


def test_question_is_stripped():
    assert QueryRequest(question="  ciao  ").question == "ciao"


def test_blank_question_is_rejected():
    with pytest.raises(ValidationError):
        QueryRequest(question="   ")


def test_question_too_long_is_rejected():
    with pytest.raises(ValidationError):
        QueryRequest(question="a" * 2001)


@pytest.mark.parametrize("top_k", [0, 21])
def test_top_k_out_of_range_is_rejected(top_k):
    with pytest.raises(ValidationError):
        QueryRequest(question="ciao", top_k=top_k)
