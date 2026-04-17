"""
HumanInLoopAgent — Proper LangGraph HITL node using interrupt().

This is a first-class agent in the LangGraph StateGraph — NOT UI logic.
LangGraph's interrupt() primitive suspends the graph, checkpoints the full
state, and resumes exactly where it left off when the human responds via
POST /thread/{thread_id}/resume.

Two checkpoints:
  1. review_classification()  — always, after ClassifierAgent
     Human confirms or overrides the predicted doc_type.

  2. review_extraction()       — conditionally, after ValidatorAgent (accuracy < 80%)
     Human reviews extracted fields and approves / corrects / rejects.
"""
import time
from typing import Any, Dict

from langgraph.types import interrupt

from graph.state import DocumentState
from schemas.document_schemas import (
    DocumentType,
    ValidationResult,
    ValidationStatus,
)
from utils.knowledge_lookup import get_knowledge_lookup
from utils.logger import logger


class HumanInLoopAgent:
    """
    Agentic HITL node — wraps LangGraph interrupt() for human-in-the-loop reviews.

    Graph integration (in HITLWorkflow):
        classify
            ↓
        human_review_classify   ← this node (always)
            ↓
        extract → validate → [repair?]
            ↓
        human_review_extract    ← this node (only when accuracy < 0.80)
            ↓
        redact → report → END

    interrupt() is the LangGraph primitive for human-in-the-loop:
    - Calling interrupt(payload) inside a node suspends the graph.
    - Full state is persisted to the checkpointer (MemorySaver / SqliteSaver).
    - graph.stream(Command(resume=human_input), config) resumes from this exact point.
    - human_input becomes the return value of interrupt().
    """

    def __init__(self):
        self.name = "HumanInLoopAgent"
        logger.info(f"{self.name} initialized")

    # ── Checkpoint 1: Classification Review ──────────────────────────────────

    def review_classification(self, state: DocumentState) -> DocumentState:
        """
        HITL node — always executed after ClassifierAgent.

        Suspends graph via interrupt(), presenting classification result to the
        human. On resume, applies override if human changed the doc_type.

        human_input schema (sent by POST /thread/{id}/resume):
          {
            "resolution": "approved" | "corrected" | "rejected",
            "doc_type_override": "resume" | "financial_document" | ...  (optional)
          }
        """
        start = time.time()
        logger.info(f"{self.name}: Interrupting graph — classification review")

        cr = state.get("classification_result")

        # ── Suspend graph; present payload to human UI ────────────────────────
        human_input: Dict[str, Any] = interrupt({
            "hitl_type":  "classify",
            "doc_type":   state.get("doc_type").value if state.get("doc_type") else "unknown",
            "confidence": float(cr.confidence) if cr else 0.0,
            "reasoning":  cr.reasoning if cr else "",
            "message":    "Please confirm the document type detected by the AI classifier.",
        })
        # ── Graph resumes here with human_input populated ─────────────────────

        resolution = human_input.get("resolution", "approved")
        state["hitl_required"]  = True
        state["hitl_type"]      = "classify"
        state["hitl_resolution"] = resolution

        if resolution == "rejected":
            state["errors"] = list(state.get("errors", [])) + [
                "Document rejected by human reviewer at classification stage."
            ]
            state["success"] = False
            logger.info(f"{self.name}: Document rejected at classification review")
            return state

        override = human_input.get("doc_type_override")
        if override:
            try:
                new_type = DocumentType(override)
                if new_type != state.get("doc_type"):
                    logger.info(
                        f"{self.name}: Doc type overridden "
                        f"{state.get('doc_type')} → {new_type}"
                    )
                    state["doc_type"] = new_type
                    state["custom_doc_type"] = None  # clear any previous custom label
                    state["extracted_fields"] = None
                    state["extraction_result"] = None
            except ValueError:
                # Custom type not in enum — store as free-text label so the
                # extractor and validator use it in their prompts.
                logger.info(
                    f"{self.name}: Custom doc type '{override}' stored for open-ended extraction"
                )
                state["doc_type"] = DocumentType.UNKNOWN
                state["custom_doc_type"] = override
                state["extracted_fields"] = None
                state["extraction_result"] = None

        state["agent_timings"] = dict(state.get("agent_timings") or {})
        state["agent_timings"][f"{self.name}_classify"] = time.time() - start
        return state

    # ── Checkpoint 2: Extraction / Validation Review ──────────────────────────

    def review_extraction(self, state: DocumentState) -> DocumentState:
        """
        HITL node — executed when extraction accuracy < 80% OR validation failed.

        Suspends graph via interrupt(), presenting extracted fields to the human.
        On resume, human corrections are applied and validation is marked approved.

        human_input schema (sent by POST /thread/{id}/resume):
          {
            "resolution":   "approved" | "corrected" | "rejected",
            "corrections":  {"field_name": "corrected_value", ...}  (optional)
          }
        """
        start = time.time()
        logger.info(f"{self.name}: Interrupting graph — extraction review")

        vr       = state.get("validation_result")
        accuracy = float(state.get("current_accuracy", 0.0))
        custom_doc_type = state.get("custom_doc_type")
        display_doc_type = (
            custom_doc_type
            if custom_doc_type
            else state.get("doc_type").value if state.get("doc_type") else "unknown"
        )
        review_message = (
            f"No predefined schema exists for custom document type '{display_doc_type}'. "
            "Please validate the extracted fields and approve or correct them."
            if custom_doc_type
            else (
                f"Extraction accuracy is {accuracy:.0%} (below 80% threshold). "
                "Please review and correct the extracted fields."
            )
        )

        # ── Suspend graph ─────────────────────────────────────────────────────
        human_input: Dict[str, Any] = interrupt({
            "hitl_type":         "extract",
            "doc_type":          display_doc_type,
            "extracted_fields":  state.get("extracted_fields") or {},
            "accuracy":          accuracy,
            "missing_fields":    list(state.get("missing_schema_fields") or []),
            "validation_errors": list(vr.errors if vr else []),
            "message": review_message,
        })
        # ── Graph resumes here ────────────────────────────────────────────────

        resolution = human_input.get("resolution", "approved")
        state["hitl_type"]       = "extract"
        state["hitl_resolution"] = resolution

        if resolution == "rejected":
            state["errors"] = list(state.get("errors", [])) + [
                "Document rejected by human reviewer at extraction review stage."
            ]
            state["success"] = False
            logger.info(f"{self.name}: Document rejected at extraction review")
            return state

        corrections = human_input.get("corrections")
        if corrections and isinstance(corrections, dict):
            state["extracted_fields"]  = corrections
            state["hitl_corrections"]  = corrections
            logger.info(
                f"{self.name}: Applied human corrections "
                f"({len(corrections)} fields)"
            )

        observed_fields = state.get("extracted_fields") if isinstance(state.get("extracted_fields"), dict) else {}
        if observed_fields:
            try:
                knowledge_lookup = get_knowledge_lookup()
                doc_type_label = state.get("doc_type").value if state.get("doc_type") else "unknown"
                knowledge_lookup.register_runtime_observed_fields(
                    doc_type=doc_type_label,
                    observed_fields=observed_fields,
                    custom_doc_type=state.get("custom_doc_type"),
                    notes="Auto-merged from HITL-approved extraction output.",
                )
                logger.info(
                    f"{self.name}: Runtime knowledge profile updated in DB "
                    f"({len(observed_fields)} observed fields)"
                )
            except Exception as e:
                logger.warning(f"{self.name}: Failed to update runtime knowledge profile: {e}")

        # Mark validation as human-approved so graph can proceed to redact
        state["validation_result"] = ValidationResult(
            status=ValidationStatus.VALID,
            is_valid=True,
            errors=[],
            warnings=["Fields reviewed and approved by human operator."],
        )
        state["validation_status"] = "valid"
        state["needs_repair"]      = False

        state["agent_timings"] = dict(state.get("agent_timings") or {})
        state["agent_timings"][f"{self.name}_extract"] = time.time() - start
        return state


# Singleton — imported by HITLWorkflow
human_review_agent = HumanInLoopAgent()
