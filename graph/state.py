"""
LangGraph State Definition
"""
from typing import TypedDict, Optional, Dict, List, Any
from datetime import datetime
from schemas.document_schemas import (
    DocumentType,
    ClassificationResult,
    ExtractionResult,
    ValidationResult,
    RedactionResult,
    ResponsibleAILog,
)


class DocumentState(TypedDict):
    """
    State object passed between LangGraph nodes
    
    This defines the complete state that flows through the workflow.
    Each agent reads from and writes to this state.
    """
    # Input
    file_path: str
    raw_text: str
    ground_truth_pii: Optional[List[Dict[str, Any]]]
    
    # Classification
    doc_type: Optional[DocumentType]
    classification_result: Optional[ClassificationResult]
    
    # Extraction
    extracted_fields: Optional[Dict[str, Any]]
    extraction_result: Optional[ExtractionResult]
    
    # Validation
    validation_status: Optional[str]
    validation_result: Optional[ValidationResult]
    needs_repair: bool
    repair_attempts: int
    current_accuracy: float          # Live accuracy score (0.0–1.0); drives accuracy-based repair
    missing_schema_fields: List[str] # Fields still missing after extraction; used by self-repair
    
    # Redaction
    redacted_text: Optional[str]
    redaction_result: Optional[RedactionResult]
    
    # Metrics & Reporting
    metrics: Optional[Dict[str, Any]]
    
    # Responsible AI Logging
    trace_log: List[ResponsibleAILog]
    
    # Timing & Metadata
    start_time: datetime
    agent_timings: Dict[str, float]
    
    # Error Handling
    errors: List[str]
    retry_count: int
    
    # Success flag
    success: bool

    # HITL (Human-In-The-Loop) fields
    hitl_required: bool                          # True when human review is needed
    hitl_type: Optional[str]                     # "classify" | "extract"
    hitl_resolution: Optional[str]               # "approved" | "corrected" | "rejected"
    hitl_corrections: Optional[Dict[str, Any]]   # human-corrected field values
    custom_doc_type: Optional[str]               # free-text doc type typed by human (not in DocumentType enum)

    # Supervisor policy / orchestration metadata
    supervisor_mode: Optional[str]               # e.g. "auto_hitl_policy"
    supervisor_classification_confidence: float
    supervisor_classification_decision: Optional[str]
    supervisor_classification_reason: Optional[str]
    supervisor_validation_decision: Optional[str]
    supervisor_validation_reason: Optional[str]


class GraphConfig(TypedDict):
    """Configuration for LangGraph execution"""
    max_repair_attempts: int
    enable_responsible_ai_logging: bool
    visualize: bool
