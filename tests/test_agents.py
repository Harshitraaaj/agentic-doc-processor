"""
Updated unit tests for Agentic Document Processor.

These tests are aligned with the current schema model:
- financial_document / resume / job_offer / medical_record / id_document / academic / unknown
"""

from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from agents import (
    classifier_agent,
    extractor_agent,
    redactor_agent,
    reporter_agent,
    self_repair_node,
    validator_agent,
)
from graph.state import DocumentState
from schemas.document_schemas import (
    ClassificationResult,
    DocumentType,
    RedactionResult,
    ValidationResult,
    ValidationStatus,
)
from utils.document_loader import document_loader


def _build_state(raw_text: str, file_path: str = "/test/doc.txt") -> DocumentState:
    return DocumentState(
        file_path=file_path,
        raw_text=raw_text,
        ground_truth_pii=None,
        doc_type=None,
        classification_result=None,
        extracted_fields=None,
        extraction_result=None,
        validation_status=None,
        validation_result=None,
        needs_repair=False,
        repair_attempts=0,
        current_accuracy=0.0,
        missing_schema_fields=[],
        redacted_text=None,
        redaction_result=None,
        metrics=None,
        trace_log=[],
        start_time=datetime.utcnow(),
        agent_timings={},
        errors=[],
        retry_count=0,
        success=False,
        hitl_required=False,
        hitl_type=None,
        hitl_resolution=None,
        hitl_corrections=None,
        custom_doc_type=None,
        supervisor_mode=None,
        supervisor_classification_confidence=0.0,
        supervisor_classification_decision=None,
        supervisor_classification_reason=None,
        supervisor_validation_decision=None,
        supervisor_validation_reason=None,
    )


@pytest.fixture
def sample_financial_text() -> str:
    return """
    INVOICE
    Invoice Number: INV-2024-001
    Date: 2024-01-15
    Vendor: ACME Corporation
    Customer: John Smith
    Subtotal: $500.00
    Tax: $40.00
    Total: $540.00
    """


@pytest.fixture
def sample_state(sample_financial_text: str) -> DocumentState:
    return _build_state(sample_financial_text, "/test/invoice.txt")


@pytest.fixture
def mock_classifier_response() -> dict:
    return {
        "content": '{"doc_type": "financial_document", "confidence": 0.95, "reasoning": "Contains invoice-like financial fields"}',
        "provider": "bedrock_claude",
        "model": "claude-3-haiku",
        "latency": 0.5,
        "tokens": {"input": 100, "output": 50},
    }


@pytest.fixture(autouse=True)
def disable_knowledge_lookup(monkeypatch):
    """Keep tests lightweight and deterministic by disabling FAISS knowledge init."""
    monkeypatch.setattr(validator_agent, "_ensure_knowledge_lookup", lambda: None)
    monkeypatch.setattr(validator_agent, "knowledge_lookup", None)


def test_classifier_happy_path(sample_state: DocumentState, mock_classifier_response: dict):
    with patch("agents.classifier_agent.llm_client.generate", return_value=mock_classifier_response):
        result_state = classifier_agent.classify(sample_state)

    assert result_state["doc_type"] == DocumentType.FINANCIAL_DOCUMENT
    assert result_state["classification_result"] is not None
    assert result_state["classification_result"].confidence >= 0.9
    assert len(result_state["trace_log"]) > 0


def test_classifier_unknown_document():
    state = _build_state("Random text with no structured cues", "/test/unknown.txt")
    mock_response = {
        "content": '{"doc_type": "unknown", "confidence": 0.3, "reasoning": "No clear structure"}',
        "provider": "bedrock_claude",
        "model": "claude-3-haiku",
        "latency": 0.4,
        "tokens": {"input": 40, "output": 20},
    }

    with patch("agents.classifier_agent.llm_client.generate", return_value=mock_response):
        result_state = classifier_agent.classify(state)

    assert result_state["doc_type"] == DocumentType.UNKNOWN


def test_classifier_llm_failure(sample_state: DocumentState):
    with patch("agents.classifier_agent.llm_client.generate", side_effect=Exception("LLM timeout")):
        result_state = classifier_agent.classify(sample_state)

    assert result_state["doc_type"] == DocumentType.UNKNOWN
    assert len(result_state["errors"]) > 0


def test_extractor_financial_document_extraction(sample_state: DocumentState):
    sample_state["doc_type"] = DocumentType.FINANCIAL_DOCUMENT

    mock_extraction = {
        "content": """{
            \"document_number\": \"INV-2024-001\",
            \"document_date\": \"2024-01-15\",
            \"issuer_name\": \"ACME Corporation\",
            \"recipient_name\": \"John Smith\",
            \"total_amount\": 540.00,
            \"tax_amount\": 40.00,
            \"currency\": \"USD\"
        }""",
        "provider": "bedrock_claude",
        "model": "claude-3-haiku",
        "latency": 0.9,
        "tokens": {"input": 300, "output": 90},
    }

    with patch("agents.extractor_agent.llm_client.generate", return_value=mock_extraction):
        result_state = extractor_agent.extract(sample_state)

    assert result_state["extracted_fields"] is not None
    assert result_state["extracted_fields"].get("document_number") == "INV-2024-001"
    assert result_state["extraction_result"] is not None


def test_validator_valid_data(sample_state: DocumentState):
    sample_state["doc_type"] = DocumentType.FINANCIAL_DOCUMENT
    sample_state["extracted_fields"] = {
        "document_number": "INV-2024-001",
        "document_date": "2024-01-15",
        "issuer_name": "ACME Corporation",
        "recipient_name": "John Smith",
        "total_amount": 540.00,
        "tax_amount": 40.00,
        "currency": "USD",
    }

    result_state = validator_agent.validate(sample_state)
    assert result_state["validation_result"] is not None


def test_validator_schema_mismatch(sample_state: DocumentState):
    sample_state["doc_type"] = DocumentType.FINANCIAL_DOCUMENT
    sample_state["extracted_fields"] = {
        "document_number": 12345,
        "total_amount": "not_a_number",
    }

    result_state = validator_agent.validate(sample_state)
    assert result_state["validation_result"] is not None
    assert len(result_state["validation_result"].errors) > 0


def test_self_repair_successful(sample_state: DocumentState):
    sample_state["doc_type"] = DocumentType.FINANCIAL_DOCUMENT
    sample_state["extracted_fields"] = {
        "document_number": "INV-001",
        "total_amount": "invalid",
    }
    sample_state["validation_result"] = ValidationResult(
        status=ValidationStatus.INVALID,
        is_valid=False,
        errors=["Field 'total_amount': invalid type"],
        warnings=[],
    )
    sample_state["needs_repair"] = True
    sample_state["repair_attempts"] = 0

    mock_repair = {
        "content": '{"document_number": "INV-001", "total_amount": 100.00}',
        "provider": "bedrock_claude",
        "model": "claude-3-haiku",
        "latency": 0.5,
        "tokens": {"input": 200, "output": 50},
    }

    with patch("agents.self_repair_node.llm_client.generate", return_value=mock_repair):
        result_state = self_repair_node.repair(sample_state)

    assert result_state["repair_attempts"] == 1
    assert result_state["extracted_fields"] is not None


def test_redactor_pii_detection():
    text_with_pii = """
    Patient: John Smith
    Email: john.smith@email.com
    Phone: 555-123-4567
    """
    state = _build_state(text_with_pii, "/test/medical.txt")
    state["doc_type"] = DocumentType.MEDICAL_RECORD
    state["extracted_fields"] = {}

    with patch("agents.redactor_agent.llm_client.generate") as mock_llm:
        mock_llm.return_value = {
            "content": '{"detected_pii": []}',
            "provider": "bedrock_claude",
            "model": "claude-3-haiku",
            "latency": 0.4,
            "tokens": {"input": 100, "output": 20},
        }
        result_state = redactor_agent.redact(state)

    assert result_state["redaction_result"] is not None
    assert result_state["redaction_result"].pii_count > 0
    redacted = result_state["redacted_text"] or ""
    assert "john.smith@email.com" not in redacted
    assert "555-123-4567" not in redacted
    assert "REDACTED" in redacted


def test_reporter_generates_metrics(sample_state: DocumentState):
    sample_state["doc_type"] = DocumentType.FINANCIAL_DOCUMENT
    sample_state["classification_result"] = ClassificationResult(
        doc_type=DocumentType.FINANCIAL_DOCUMENT,
        confidence=0.95,
        reasoning="financial fields detected",
    )
    sample_state["extracted_fields"] = {
        "document_number": "INV-001",
        "total_amount": 100.00,
    }
    sample_state["extraction_result"] = type(
        "ExtractionResultStub",
        (),
        {"extracted_fields": sample_state["extracted_fields"], "confidence": 0.9, "chunk_count": 1},
    )()
    sample_state["validation_result"] = ValidationResult(
        status=ValidationStatus.VALID,
        is_valid=True,
        errors=[],
        warnings=[],
    )
    sample_state["redaction_result"] = RedactionResult(
        redacted_text="text",
        pii_count=1,
        precision=0.95,
        recall=0.95,
    )

    result_state = reporter_agent.generate_report(sample_state)
    assert result_state["metrics"] is not None
    assert "extraction_accuracy" in result_state["metrics"]
    assert "workflow_success" in result_state["metrics"]


def test_document_loader_text_file(tmp_path: Path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("Sample text content", encoding="utf-8")

    documents = document_loader.load_document(file_path)
    text = document_loader.extract_text_from_documents(documents)

    assert len(documents) >= 1
    assert "Sample text content" in text


def test_full_workflow_simulation(sample_state: DocumentState, mock_classifier_response: dict):
    with patch("agents.classifier_agent.llm_client.generate", return_value=mock_classifier_response):
        state = classifier_agent.classify(sample_state)
        assert state["doc_type"] is not None

    state["doc_type"] = DocumentType.FINANCIAL_DOCUMENT

    state["extracted_fields"] = {
        "document_number": "INV-001",
        "total_amount": 100.00,
    }
    state = validator_agent.validate(state)
    assert state["validation_result"] is not None

    with patch("agents.redactor_agent.llm_client.generate") as mock_llm:
        mock_llm.return_value = {
            "content": '{"detected_pii": []}',
            "provider": "bedrock_claude",
            "model": "claude-3-haiku",
            "latency": 0.3,
            "tokens": {"input": 50, "output": 10},
        }
        state = redactor_agent.redact(state)
        assert state["redaction_result"] is not None

    state["classification_result"] = state.get("classification_result") or ClassificationResult(
        doc_type=DocumentType.FINANCIAL_DOCUMENT,
        confidence=0.9,
        reasoning="simulated",
    )
    state["extraction_result"] = state.get("extraction_result") or type(
        "ExtractionResultStub",
        (),
        {"extracted_fields": state.get("extracted_fields", {}), "confidence": 0.9, "chunk_count": 1},
    )()

    state = reporter_agent.generate_report(state)
    assert state["metrics"] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
