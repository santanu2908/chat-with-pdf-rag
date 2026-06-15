"""Evaluation harness: runs 19 questions against the RAG pipeline and checks answers."""
import pytest
from tests.conftest import load_eval_questions

questions = load_eval_questions()


def check_answer(answer: str, expected_keywords: list[str], negative: bool) -> bool:
    """Check if the answer contains the expected keywords.

    Normal questions: ALL keywords must appear (AND).
    Negative questions: ANY refusal keyword must appear (OR).
    """
    answer_lower = answer.lower()
    if negative:
        return any(kw.lower() in answer_lower for kw in expected_keywords)
    return all(kw.lower() in answer_lower for kw in expected_keywords)


def question_id(q):
    """Generate a readable test ID like '01-ceo-of-zentara'."""
    short = q["question"][:40].lower().replace(" ", "-").replace("?", "").replace("'", "")
    return f"{q['id']:02d}-{short}"

""" The test function is parameterized with all questions, and each test case is identified by a readable string."""
@pytest.mark.parametrize("q", questions, ids=[question_id(q) for q in questions])
def test_eval(client, q):
    resp = client.post("/query", json={"question": q["question"], "top_k": q["top_k"]})
    assert resp.status_code == 200, f"Query failed: {resp.text}"

    data = resp.json()
    answer = data["answer"]
    passed = check_answer(answer, q["expected_keywords"], q["negative"])

    assert passed, (
        f"\n  Question:  {q['question']}"
        f"\n  Category:  {q['category']}"
        f"\n  Expected:  {q['expected_keywords']}"
        f"\n  Got:       {answer}"
    )
