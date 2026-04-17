"""
Supervisor Agent — Central orchestrator for the document processing pipeline.

This agent owns the full LangGraph execution graph and makes every routing
decision. Specialist agents (classify, extract, validate, repair, redact,
report) are called as graph nodes; the supervisor decides what runs next at
each checkpoint.

Orchestration flow (HITL mode)
───────────────────────────────
classify
  └─► supervise_classification   [Supervisor: always requires HITL #1]
        └─► human_review_classify  [HITL #1: human confirms / corrects doc type]
              └─► extract
                    └─► validate
                          └─► supervise_validation  [Supervisor: repair | HITL #2 | approve]
                                ├─► repair ──► validate  (self-repair loop)
                                ├─► human_review_extract  [HITL #2: human corrects fields]
                                └─► redact
                                      └─► report → END
"""
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Literal, Optional

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from graph.state import DocumentState, GraphConfig
from agents.classifier_agent import classifier_agent
from agents.extractor_agent import extractor_agent
from agents.validator_agent import validator_agent
from agents.self_repair_node import self_repair_node
from agents.redactor_agent import redactor_agent
from agents.reporter_agent import reporter_agent
from agents.human_review_agent import human_review_agent
from utils.document_loader import document_loader
from utils.logger import logger


class SupervisorAgent:
    """
    Central orchestrator for the entire document processing pipeline.

    Owns the LangGraph execution graph and makes every routing decision.
    Specialist agents are invoked as graph nodes; this supervisor controls
    what runs next at each step via explicit decision nodes and routing edges.
    """

    ACCURACY_THRESHOLD = 0.80

    def __init__(self, config: GraphConfig = None):
        self.name = "SupervisorAgent"
        self.config = config or GraphConfig(
            max_repair_attempts=1,
            enable_responsible_ai_logging=True,
            visualize=False,
        )
        self._memory = MemorySaver()
        self._graph = self._build_graph()
        self._compiled = None
        logger.info(f"{self.name} initialized")

    # ── Specialist agent node wrappers ────────────────────────────────────────
    # Each method is registered as a graph node.  The supervisor calls the
    # right specialist at the right time via the graph edges below.

    def _classify_node(self, state: DocumentState) -> DocumentState:
        return classifier_agent.classify(state)

    def _extract_node(self, state: DocumentState) -> DocumentState:
        return extractor_agent.extract(state)

    def _validate_node(self, state: DocumentState) -> DocumentState:
        return validator_agent.validate(state)

    def _repair_node(self, state: DocumentState) -> DocumentState:
        return self_repair_node.repair(state)

    def _redact_node(self, state: DocumentState) -> DocumentState:
        return redactor_agent.redact(state)

    def _report_node(self, state: DocumentState) -> DocumentState:
        return reporter_agent.generate_report(state)

    def _hitl_classify_node(self, state: DocumentState) -> DocumentState:
        return human_review_agent.review_classification(state)

    def _hitl_extract_node(self, state: DocumentState) -> DocumentState:
        return human_review_agent.review_extraction(state)

    # ── Supervisor decision nodes ─────────────────────────────────────────────
    # These are the brain of the orchestration.  They inspect the current state,
    # make a policy decision, write it into state, and let the routing edges act.

    def _supervise_classification(self, state: DocumentState) -> DocumentState:
        """
        Supervisor checkpoint after classification.

        Policy: classification HITL is always mandatory.  The human must
        confirm or correct the predicted doc type before extraction begins.
        Confidence and decision rationale are recorded for audit/metadata.
        """
        cr = state.get("classification_result")
        confidence = float(cr.confidence) if cr else 0.0
        doc_type = state.get("doc_type")
        doc_type_value = (
            doc_type.value
            if doc_type and hasattr(doc_type, "value")
            else str(doc_type or "unknown")
        )

        state["supervisor_mode"] = "hitl_policy"
        state["supervisor_classification_confidence"] = confidence
        state["supervisor_classification_decision"] = "hitl_classification_required"
        state["supervisor_classification_reason"] = (
            f"mandatory human confirmation after classification "
            f"(doc_type={doc_type_value}, confidence={confidence:.0%})"
        )
        state["hitl_required"] = True

        logger.info(
            f"{self.name} ▶ classify: mandatory HITL — "
            f"doc_type={doc_type_value}, confidence={confidence:.0%}"
        )
        return state

    def _supervise_validation(self, state: DocumentState) -> DocumentState:
        """
        Supervisor checkpoint after validation (and after each repair attempt).

        Policy (evaluated in priority order):
          1. Custom doc type (no predefined schema) → HITL #2
          2. Low accuracy + repair budget remaining  → self-repair retry
          3. Low accuracy + budget exhausted         → HITL #2
          4. Quality acceptable                      → auto-approve → redact
        """
        needs_repair    = state.get("needs_repair", False)
        repair_attempts = state.get("repair_attempts", 0)
        max_attempts    = self.config.get("max_repair_attempts", 1)
        accuracy        = state.get("current_accuracy", 1.0)
        custom_doc_type = state.get("custom_doc_type")

        if custom_doc_type:
            decision = "hitl_extract_custom_doc_type"
            reason   = (
                f"custom doc type '{custom_doc_type}' has no predefined schema; "
                f"human validation required"
            )
        elif (needs_repair or accuracy < self.ACCURACY_THRESHOLD) and repair_attempts < max_attempts:
            decision = "self_repair_retry"
            reason   = (
                f"accuracy={accuracy:.0%}, needs_repair={needs_repair}, "
                f"attempt {repair_attempts + 1}/{max_attempts}"
            )
        elif accuracy < self.ACCURACY_THRESHOLD or needs_repair:
            decision = "hitl_extract_required"
            reason   = (
                f"post-repair quality risk: accuracy={accuracy:.0%}, "
                f"needs_repair={needs_repair}, repair budget exhausted"
            )
        else:
            decision = "auto_approved_validation"
            reason   = (
                f"quality acceptable: accuracy={accuracy:.0%}, "
                f"needs_repair={needs_repair}"
            )

        state["supervisor_validation_decision"] = decision
        state["supervisor_validation_reason"]   = reason

        logger.info(f"{self.name} ▶ validate: {decision} — {reason}")
        return state

    # ── Routing edges ─────────────────────────────────────────────────────────
    # Called by LangGraph after each conditional node; return the next node name.

    def _route_after_supervise_classify(
        self, state: DocumentState
    ) -> Literal["human_review_classify", "extract"]:
        """Route to HITL in hitl mode; skip HITL in standard mode."""
        mode = str(state.get("supervisor_mode") or "hitl_policy")
        if mode == "standard":
            return "extract"
        return "human_review_classify"

    def _route_after_hitl_classify(
        self, state: DocumentState
    ) -> Literal["extract", "report"]:
        """Approved / corrected → extract.  Rejected → report (skip processing)."""
        if state.get("hitl_resolution") == "rejected":
            logger.info(f"{self.name}: Classification rejected by human → report")
            return "report"
        return "extract"

    def _route_after_supervise_validate(
        self, state: DocumentState
    ) -> Literal["repair", "human_review_extract", "redact"]:
        """Route based on decision, with standard mode bypassing HITL nodes."""
        decision = state.get("supervisor_validation_decision", "auto_approved_validation")
        mode = str(state.get("supervisor_mode") or "hitl_policy")
        mapping = {
            "self_repair_retry":            "repair",
            "hitl_extract_required":        "human_review_extract",
            "hitl_extract_custom_doc_type": "human_review_extract",
            "auto_approved_validation":     "redact",
        }
        next_node = mapping.get(decision, "redact")
        if mode == "standard" and next_node == "human_review_extract":
            return "redact"
        return next_node

    def _route_after_hitl_extract(
        self, state: DocumentState
    ) -> Literal["redact", "report"]:
        """Approved / corrected fields → redact.  Rejected → report."""
        if state.get("hitl_resolution") == "rejected":
            logger.info(f"{self.name}: Extraction rejected by human → report")
            return "report"
        return "redact"

    # ── Graph construction ─────────────────────────────────────────────────────

    def _build_graph(self) -> StateGraph:
        """
        Build the complete orchestration graph.

        The supervisor owns this graph — every node and every routing decision
        is defined here, making the full pipeline visible in one place.
        """
        g = StateGraph(DocumentState)

        # ── Specialist agent nodes ──────────────────────────────────────────
        g.add_node("classify",  self._classify_node)
        g.add_node("extract",   self._extract_node)
        g.add_node("validate",  self._validate_node)
        g.add_node("repair",    self._repair_node)
        g.add_node("redact",    self._redact_node)
        g.add_node("report",    self._report_node)

        # ── Supervisor decision nodes ───────────────────────────────────────
        g.add_node("supervise_classification", self._supervise_classification)
        g.add_node("supervise_validation",     self._supervise_validation)

        # ── HITL interrupt nodes ────────────────────────────────────────────
        g.add_node("human_review_classify", self._hitl_classify_node)
        g.add_node("human_review_extract",  self._hitl_extract_node)

        # ── Pipeline flow ───────────────────────────────────────────────────
        g.set_entry_point("classify")

        # Step 1: classify → supervisor evaluates → always HITL #1
        g.add_edge("classify", "supervise_classification")
        g.add_conditional_edges(
            "supervise_classification",
            self._route_after_supervise_classify,
            {
                "human_review_classify": "human_review_classify",
                "extract": "extract",
            },
        )

        # Step 2: HITL #1 result → extract (approved/corrected) or report (rejected)
        g.add_conditional_edges(
            "human_review_classify",
            self._route_after_hitl_classify,
            {"extract": "extract", "report": "report"},
        )

        # Step 3: extract → validate → supervisor evaluates quality
        g.add_edge("extract", "validate")
        g.add_edge("validate", "supervise_validation")

        # Step 4: supervisor routes → repair loop, HITL #2, or redact
        g.add_conditional_edges(
            "supervise_validation",
            self._route_after_supervise_validate,
            {
                "repair":               "repair",
                "human_review_extract": "human_review_extract",
                "redact":               "redact",
            },
        )

        # Repair loops back to validate (supervisor re-evaluates after each attempt)
        g.add_edge("repair", "validate")

        # Step 5: HITL #2 result → redact (approved/corrected) or report (rejected)
        g.add_conditional_edges(
            "human_review_extract",
            self._route_after_hitl_extract,
            {"redact": "redact", "report": "report"},
        )

        # Step 6: redact → report → done
        g.add_edge("redact", "report")
        g.add_edge("report", END)

        logger.info(f"{self.name}: Orchestration graph built successfully")
        return g

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def _ensure_compiled(self) -> None:
        if self._compiled is None:
            self._compiled = self._graph.compile(checkpointer=self._memory)
            logger.info(
                f"{self.name}: Orchestration graph compiled with MemorySaver checkpointing"
            )

    def compile_workflows(self) -> None:
        """Compile the supervisor orchestration graph."""
        self._ensure_compiled()
        logger.info(f"{self.name}: Workflow compiled")

    def get_graph_mermaid(self) -> str:
        """
        Return a clean, hand-crafted Mermaid diagram for the supervisor/HITL graph.
        """
        return self._generate_supervisor_mermaid()

    def _generate_supervisor_mermaid(self) -> str:
        """Generate a user-friendly Mermaid diagram of the current supervisor flow."""
        return """graph TD
    START([🚀 START]) --> classify[📋 Classify]
    classify --> supervise_classification[🧠 Supervisor Checkpoint #1]
    supervise_classification --> human_review_classify[👤 HITL Review #1<br/>Classification]

    human_review_classify -->|Approved/Corrected| extract[📤 Extract]
    human_review_classify -->|Rejected| report[📊 Report]

    extract --> validate[✓ Validate]
    validate --> supervise_validation[🧠 Supervisor Checkpoint #2]

    supervise_validation -->|Retry Needed| repair[🔧 Self Repair]
    supervise_validation -->|Human Review Needed| human_review_extract[👤 HITL Review #2<br/>Extraction]
    supervise_validation -->|Auto Approved| redact[🔒 Redact]

    repair --> validate
    human_review_extract -->|Approved/Corrected| redact
    human_review_extract -->|Rejected| report

    redact --> report
    report --> END([✔️ END])

    style START fill:#90EE90,stroke:#006400,stroke-width:3px
    style END fill:#90EE90,stroke:#006400,stroke-width:3px

    style classify fill:#87CEEB,stroke:#4682B4,stroke-width:2px
    style extract fill:#87CEEB,stroke:#4682B4,stroke-width:2px
    style validate fill:#87CEEB,stroke:#4682B4,stroke-width:2px
    style redact fill:#87CEEB,stroke:#4682B4,stroke-width:2px
    style report fill:#87CEEB,stroke:#4682B4,stroke-width:2px

    style supervise_classification fill:#D8BFD8,stroke:#8A2BE2,stroke-width:2px
    style supervise_validation fill:#D8BFD8,stroke:#8A2BE2,stroke-width:2px
    style human_review_classify fill:#FFE4B5,stroke:#FF8C00,stroke-width:2px
    style human_review_extract fill:#FFE4B5,stroke:#FF8C00,stroke-width:2px
    style repair fill:#FFD700,stroke:#FF8C00,stroke-width:2px
"""

    # ── State initialization ───────────────────────────────────────────────────

    def _initialize_state(
        self,
        file_path: str,
        raw_text: str,
        ground_truth_pii: Optional[list[Dict[str, Any]]] = None,
        supervisor_mode: str = "hitl_policy",
    ) -> DocumentState:
        return DocumentState(
            file_path=file_path,
            raw_text=raw_text,
            ground_truth_pii=ground_truth_pii,
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
            supervisor_mode=supervisor_mode,
            supervisor_classification_confidence=0.0,
            supervisor_classification_decision=None,
            supervisor_classification_reason=None,
            supervisor_validation_decision=None,
            supervisor_validation_reason=None,
        )

    # ── Internal runner ────────────────────────────────────────────────────────

    def _run_until_pause(self, input_: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Stream the graph until an interrupt (HITL checkpoint) or completion."""
        for _ in self._compiled.stream(input_, config, stream_mode="values"):
            pass

        snapshot = self._compiled.get_state(config)

        if snapshot.next:
            interrupt_data: Dict[str, Any] = {}
            for task in snapshot.tasks:
                for intr in task.interrupts:
                    interrupt_data = intr.value
                    break
                if interrupt_data:
                    break
            logger.info(
                f"{self.name}: Orchestration paused — "
                f"hitl_type={interrupt_data.get('hitl_type')}"
            )
            return {
                "status": "interrupted",
                "interrupt_data": interrupt_data,
                "state": dict(snapshot.values),
            }

        final_state = dict(snapshot.values)
        logger.info(
            f"{self.name}: Orchestration complete — success={final_state.get('success')}"
        )
        return {"status": "complete", "final_state": final_state}

    # ── Public API ─────────────────────────────────────────────────────────────

    def process_document(
        self,
        file_path: str,
        thread_id: str,
        ground_truth_pii: Optional[list[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Run standard end-to-end processing using the supervisor graph in standard mode.

        In standard mode, HITL nodes are bypassed by routing policy and the
        graph runs to completion in one pass.
        """
        self._ensure_compiled()
        file_path_obj = Path(file_path)
        raw_text = document_loader.load_and_extract_text(file_path_obj)
        if not raw_text or not raw_text.strip():
            raise ValueError(f"No text could be extracted from {file_path_obj.name}")

        initial_state = self._initialize_state(
            str(file_path_obj),
            raw_text,
            ground_truth_pii=ground_truth_pii,
            supervisor_mode="standard",
        )
        config = {"configurable": {"thread_id": thread_id}}

        logger.info(
            f"{self.name}: Standard processing (single-graph) — "
            f"thread={thread_id}, file={file_path_obj.name}"
        )
        result = self._run_until_pause(initial_state, config)
        if result.get("status") != "complete":
            raise RuntimeError("Standard mode unexpectedly paused for HITL review")
        return result.get("final_state", {})

    def start_processing(self, file_path: str, thread_id: str) -> Dict[str, Any]:
        """
        Start the orchestrated HITL pipeline.

        Runs the full graph from the beginning.  Pauses at the mandatory
        classification HITL checkpoint and returns interrupt data to the caller.
        """
        self._ensure_compiled()
        file_path_obj = Path(file_path)
        raw_text = document_loader.load_and_extract_text(file_path_obj)
        if not raw_text or not raw_text.strip():
            raise ValueError(f"No text could be extracted from {file_path_obj.name}")

        initial_state = self._initialize_state(
            str(file_path_obj),
            raw_text,
            supervisor_mode="hitl_policy",
        )
        config = {"configurable": {"thread_id": thread_id}}

        logger.info(
            f"{self.name}: Starting orchestration — "
            f"thread={thread_id}, file={file_path_obj.name}"
        )
        return self._run_until_pause(initial_state, config)

    def resume_processing(self, thread_id: str, human_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resume a paused orchestration from a HITL checkpoint.

        Injects human_input into the suspended interrupt() call and continues
        until the next checkpoint or final completion.
        """
        self._ensure_compiled()
        config = {"configurable": {"thread_id": thread_id}}
        logger.info(
            f"{self.name}: Resuming orchestration — "
            f"thread={thread_id}, resolution={human_input.get('resolution')}"
        )
        return self._run_until_pause(Command(resume=human_input), config)
