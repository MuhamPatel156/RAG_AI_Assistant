from types import SimpleNamespace
import json

import pytest

from langchain_core.documents import Document

from self_rag import GUARD, IndexedPDF, NO_ANSWER, SelfRAG, citation_feedback, normalize_citations, obvious_overview, overview_passages, parse_review


class FakeLLM:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.messages = []

    def invoke(self, messages):
        self.messages.append(messages)
        return SimpleNamespace(content=next(self.responses))


class FakeStore:
    def __init__(self, documents):
        self.documents = documents
        self.queries = []
        self.retriever_options = {}

    def as_retriever(self, **kwargs):
        self.retriever_options = kwargs
        return self

    def invoke(self, question):
        self.queries.append(question)
        return list(self.documents)


PLAN = '{"mode": "focused", "search_query": "What do trees absorb?"}'
ANSWER = "Trees absorb carbon dioxide [Page 3]."


def run(responses, documents=None, question="What do trees absorb?", plan=PLAN):
    engine = SelfRAG.__new__(SelfRAG)
    engine.llm = FakeLLM(([plan] if plan is not None else []) + responses)
    if documents is None:
        documents = [Document(page_content="Trees absorb carbon dioxide.", metadata={"page": 2})]
    engine.store = FakeStore(documents)
    indexed = IndexedPDF(engine.build_workflow(engine.store, "notes.pdf", documents), 3, len(documents))
    return engine.ask(indexed, question), engine


def test_supported_answer_includes_page_metadata_and_guard():
    result, engine = run(["true", ANSWER, "true"])
    assert result["supported"]
    assert result["sources"][0]["page"] == 3
    assert "page 3" in engine.llm.messages[2][1][1]
    assert all(messages[0][1].startswith(GUARD) for messages in engine.llm.messages)
    assert result["revisions"] == 0


def test_empty_retrieval_always_returns_an_answer():
    result, _ = run([], documents=[])
    assert result["answer"] == NO_ANSWER
    assert not result["supported"]


def test_irrelevant_context_stops_before_generation():
    result, engine = run(["false"])
    assert result["answer"] == NO_ANSWER
    assert len(engine.llm.messages) == 2


def test_two_revisions_are_the_hard_limit():
    result, engine = run(["true", "first", "false", "second", "false", "third", "false"])
    assert result["revisions"] == 2
    assert result["answer"] == NO_ANSWER
    assert len(engine.llm.messages) == 8


def test_successful_revision_reaches_usefulness():
    result, _ = run(["true", "first", "false", ANSWER, "true"])
    assert result["answer"] == ANSWER
    assert result["supported"]
    assert result["revisions"] == 1


def test_unhelpful_answer_is_not_presented_as_supported():
    result, _ = run(["true", "I don't know.", "false", "I don't know.", "false", "I don't know.", "false"])
    assert result["answer"] == NO_ANSWER
    assert not result["supported"]


def test_negated_true_is_not_accepted_as_true():
    result, _ = run(["not true", "false"])
    assert result["answer"] == NO_ANSWER


@pytest.mark.parametrize("question", [
    "What is this pdf all about ?",
    "What is this document about? Give me a concise summary.",
    "What are the most important takeaways from this document?",
    "Explain the main concept in this document in simple terms.",
])
def test_frontend_examples_use_document_coverage_not_similarity(question):
    documents = [Document(page_content=f"Topic {i} is explained here.", metadata={"page": i}) for i in range(12)]
    result, engine = run(["The PDF covers several topics [Source 1, page 1] [Source 12, page 12].", "True."],
                         documents, question, plan=None)
    assert result["supported"]
    assert result["mode"] == "overview"
    assert len(result["sources"]) == 12
    assert not engine.store.queries
    assert len(engine.llm.messages) == 2  # Generate + grounded review; no irrelevant broad-query rejection.
    payload = json.loads(engine.llm.messages[0][1][1])
    assert "Topic 0" in payload["context"] and "Topic 11" in payload["context"]


def test_informal_overview_is_understood_by_planner():
    result, _ = run([ANSWER, "true"], question="Can you tell me what all this is getting at?",
                    plan='{"mode":"overview","search_query":"document topics"}')
    assert result["mode"] == "overview"
    assert result["supported"]


@pytest.mark.parametrize("question", ["Summarize chapter 3", "Explain the main concept on page 5 of this PDF", "What does the PDF say about trees?"])
def test_section_and_topic_questions_are_not_forced_into_overview(question):
    assert not obvious_overview(question)


@pytest.mark.parametrize("raw,expected", [
    ("True.", True), ("**TRUE**", True), ("False!", False),
    ('```json\n{"passed": true, "feedback": "Grounded synthesis."}\n```', True),
    ('{"passed": "true"}', None), ("not true", None), ("true or false", None),
])
def test_review_parser_is_tolerant_without_substring_false_positives(raw, expected):
    assert parse_review(raw)[0] is expected


def test_review_feedback_reaches_revision_prompt():
    feedback = "Remove the unsupported claim about nitrogen."
    result, engine = run(["true", "Trees absorb nitrogen [Source 1, page 3].",
                          json.dumps({"passed": False, "feedback": feedback}), ANSWER, "true"])
    assert result["supported"]
    assert result["revisions"] == 1
    revision_payload = json.loads(engine.llm.messages[4][1][1])
    assert feedback in revision_payload["reviewer_feedback"]


def test_rewritten_and_original_queries_are_both_searched_and_deduplicated():
    result, engine = run(["true", ANSWER, "true"], plan='{"mode":"focused","search_query":"trees carbon uptake"}')
    assert engine.store.queries == ["What do trees absorb?", "trees carbon uptake"]
    assert len(result["sources"]) == 1
    assert engine.store.retriever_options["search_type"] == "mmr"
    assert engine.store.retriever_options["search_kwargs"]["k"] == 8


def test_bad_planner_output_preserves_original_question():
    result, engine = run(["true", ANSWER, "true"], plan="unknown")
    assert result["supported"]
    assert engine.store.queries == ["What do trees absorb?"]


def test_incorrect_page_citation_triggers_revision_even_if_model_approves():
    result, engine = run(["true", "Carbon dioxide [Source 1, page 99].", "true", ANSWER, "true"])
    assert result["revisions"] == 1
    assert result["supported"]
    assert "not provided" in json.loads(engine.llm.messages[4][1][1])["reviewer_feedback"]


def test_large_document_overview_is_bounded_diverse_and_disclosed():
    documents = [Document(page_content=f"Passage {i}: " + "x" * 2450, metadata={"page": i}) for i in range(100)]
    selected, note = overview_passages(documents, budget=10400)
    assert len(selected) == 4
    assert [doc.metadata["page"] for doc in selected] == [0, 33, 66, 99]
    assert sum(len(doc.page_content) + 100 for doc in selected) <= 10400
    assert "4 representative passages" in note
    assert "100 passages" in note
    assert "unsampled" in note
    assert len(documents[0].page_content) > 2400  # Originals are not modified.


def test_small_document_overview_includes_all_text():
    documents = [Document(page_content="Beginning", metadata={"page": 0}),
                 Document(page_content="Ending", metadata={"page": 1})]
    selected, note = overview_passages(documents)
    assert selected == documents
    assert note == ""


@pytest.mark.parametrize("citation", [
    "[Source 1, page 1] [Source 2, page 2] [Source 3, page 3]",
    "[Source 1, page 1; Source 2, page 2; Source 3, page 3]",
    "[Source 1-3, pages 1-3]", "[Sources 1–3, pages 1–3]",
])
def test_valid_grouped_or_range_citations_do_not_discard_a_summary(citation):
    sources = [{"id": i, "page": i} for i in range(1, 4)]
    assert citation_feedback(f"Supported summary {citation}.", sources) == ""


@pytest.mark.parametrize("citation", ["[Source 1-3, pages 4-6]", "[Pages 1-999999999]", "[Source 3-1, pages 3-1]"])
def test_invalid_citation_ranges_are_rejected_without_expanding_them(citation):
    assert citation_feedback(citation, [{"id": i, "page": i} for i in range(1, 4)])


@pytest.mark.parametrize("citation", ["[Source 1, pages 1-3]", "[Pages 1, 2 and 3]", "[Source 1, page 1; Source 2, page 2; Source 3, page 3]"])
def test_page_normalization_removes_ambiguous_document_vs_excerpt_numbers(citation):
    sources = [{"id": i, "page": i} for i in range(1, 4)]
    assert normalize_citations(f"Summary {citation}", sources) == "Summary [Page 1] [Page 2] [Page 3]"


def test_normalization_does_not_hide_a_page_that_was_not_retrieved():
    answer = "Invented [Page 99]"
    sources = [{"id": 1, "page": 1}]
    assert normalize_citations(answer, sources) == answer
    assert citation_feedback(answer, sources)
