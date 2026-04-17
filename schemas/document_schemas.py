"""
Pydantic schemas for document processing
"""
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Dict, List
from pydantic import BaseModel, Field, validator


class DocumentType(str, Enum):
    """Supported document types (OPTIMIZED - 6 CORE TYPES + UNKNOWN)"""
    FINANCIAL_DOCUMENT = "financial_document"
    RESUME = "resume"
    JOB_OFFER = "job_offer"
    MEDICAL_RECORD = "medical_record"
    ID_DOCUMENT = "id_document"
    ACADEMIC = "academic"
    UNKNOWN = "unknown"  # Fallback for classification failures


class ValidationStatus(str, Enum):
    """Validation status"""
    VALID = "valid"
    INVALID = "invalid"
    NEEDS_REPAIR = "needs_repair"
    REPAIRED = "repaired"
    VALID_AFTER_REPAIR = "valid_after_repair"  # Validated successfully after self-repair


class PIIType(str, Enum):
    """PII field types (internationally compatible)"""
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"  # National ID (US SSN, India Aadhaar, UK NI, etc.)
    CREDIT_CARD = "credit_card"
    ADDRESS = "address"
    NAME = "name"
    DATE_OF_BIRTH = "date_of_birth"
    MEDICAL_ID = "medical_id"
    BANK_ACCOUNT = "bank_account"
    TAX_ID = "tax_id"  # Tax IDs (India PAN/GSTIN, US EIN, UK UTR, EU VAT, etc.)
    GENDER = "gender"  # Gender (Male/Female/Other) — personal demographic info


# ==================== Extraction Schemas ====================

class FinancialDocumentFields(BaseModel):
    """Structured fields for financial documents (invoices, receipts, bills, etc.)"""
    document_number: Optional[str] = None  # Invoice/receipt/bill number
    document_date: Optional[str] = None
    due_date: Optional[str] = None
    issuer_name: Optional[str] = None  # Vendor/merchant name
    issuer_address: Optional[str] = None
    recipient_name: Optional[str] = None  # Customer name
    recipient_address: Optional[str] = None
    total_amount: Optional[float] = None
    tax_amount: Optional[float] = None
    currency: Optional[str] = "USD"
    payment_method: Optional[str] = None
    line_items: List[Dict[str, Any]] = Field(default_factory=list)


class ContractFields(BaseModel):
    """Structured fields for contract documents"""
    contract_title: Optional[str] = None
    contract_date: Optional[str] = None
    effective_date: Optional[str] = None
    expiration_date: Optional[str] = None
    party_a: Optional[str] = None
    party_b: Optional[str] = None
    contract_value: Optional[float] = None
    key_terms: List[str] = Field(default_factory=list)


class ResumeFields(BaseModel):
    """Structured fields for resumes and CVs"""
    candidate_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    linkedin_url: Optional[str] = None
    summary: Optional[str] = None
    education: List[Dict[str, Any]] = Field(default_factory=list)  # degree, institution, year
    work_experience: List[Dict[str, Any]] = Field(default_factory=list)  # title, company, dates
    skills: List[str] = Field(default_factory=list)
    certifications: List[str] = Field(default_factory=list)
    languages: List[str] = Field(default_factory=list)


class JobOfferFields(BaseModel):
    """Structured fields for job offers and internship offers"""
    candidate_name: Optional[str] = None
    company_name: Optional[str] = None
    position_title: Optional[str] = None
    offer_date: Optional[str] = None
    start_date: Optional[str] = None
    salary: Optional[str] = None
    employment_type: Optional[str] = None  # full-time, part-time, internship, contract
    work_location: Optional[str] = None
    department: Optional[str] = None
    reporting_to: Optional[str] = None
    benefits: List[str] = Field(default_factory=list)
    conditions: List[str] = Field(default_factory=list)
    deadline_to_accept: Optional[str] = None


class MedicalRecordFields(BaseModel):
    """Structured fields for medical records"""
    patient_name: Optional[str] = None
    patient_id: Optional[str] = None
    date_of_birth: Optional[str] = None
    visit_date: Optional[str] = None
    physician_name: Optional[str] = None
    department: Optional[str] = None
    diagnosis: Optional[str] = None
    prescribed_medications: List[str] = Field(default_factory=list)
    lab_results: List[Dict[str, Any]] = Field(default_factory=list)
    follow_up_date: Optional[str] = None
    notes: Optional[str] = None


class IdDocumentFields(BaseModel):
    """Structured fields for identity documents"""
    document_type: Optional[str] = None  # passport, driver_license, visa, ID card, etc.
    document_number: Optional[str] = None
    full_name: Optional[str] = None
    date_of_birth: Optional[str] = None
    gender: Optional[str] = None
    nationality: Optional[str] = None
    place_of_birth: Optional[str] = None
    address: Optional[str] = None
    issue_date: Optional[str] = None
    expiration_date: Optional[str] = None
    issuing_authority: Optional[str] = None


class AcademicFields(BaseModel):
    """Structured fields for academic documents"""
    document_type: Optional[str] = None  # transcript, diploma, certificate, research paper
    student_name: Optional[str] = None
    student_id: Optional[str] = None
    institution_name: Optional[str] = None
    degree_program: Optional[str] = None
    graduation_date: Optional[str] = None
    gpa: Optional[float] = None
    doi: Optional[str] = None  # Digital Object Identifier for research papers
    courses: List[Dict[str, Any]] = Field(default_factory=list)  # course name, grade
    honors: List[str] = Field(default_factory=list)


class GeneralDocumentFields(BaseModel):
    """Fallback schema for general documents"""
    title: Optional[str] = None
    author: Optional[str] = None
    date: Optional[str] = None
    summary: Optional[str] = None
    key_entities: List[str] = Field(default_factory=list)
    key_dates: List[str] = Field(default_factory=list)
    key_amounts: List[float] = Field(default_factory=list)


# ==================== Agent Output Schemas ====================

class ClassificationResult(BaseModel):
    """Output from ClassifierAgent"""
    doc_type: DocumentType
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ExtractionResult(BaseModel):
    """Output from ExtractorAgent"""
    doc_type: DocumentType
    extracted_fields: Dict[str, Any]
    confidence: float = Field(ge=0.0, le=1.0)
    chunk_count: int = 1
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ValidationResult(BaseModel):
    """Output from ValidatorAgent"""
    status: ValidationStatus
    is_valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    validated_fields: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class PIIDetection(BaseModel):
    """Detected PII field"""
    field_name: str = ""
    pii_type: PIIType
    original_text: str
    redacted_text: str
    detection_source: str = "regex"  # "regex" or "llm"
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)


class RedactionResult(BaseModel):
    """Output from RedactorAgent"""
    redacted_text: str
    pii_detections: List[PIIDetection] = Field(default_factory=list)
    detected_pii: List[PIIDetection] = Field(default_factory=list)  # Alias for backward compatibility
    pii_count: int = 0
    precision: float = Field(ge=0.0, le=1.0, default=0.0)
    recall: float = Field(ge=0.0, le=1.0, default=0.0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class MetricsReport(BaseModel):
    """Metrics computed by ReporterAgent"""
    extraction_accuracy: float = Field(ge=0.0, le=1.0)
    pii_recall: float = Field(ge=0.0, le=1.0)
    pii_precision: float = Field(ge=0.0, le=1.0)
    workflow_success: bool
    total_processing_time: float  # seconds
    agent_latencies: Dict[str, float] = Field(default_factory=dict)
    error_count: int = 0
    retry_count: int = 0
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ResponsibleAILog(BaseModel):
    """Responsible AI trace log entry with comprehensive details"""
    agent_name: str
    
    # Model Information
    llm_model_used: str
    llm_provider: Optional[str] = None  # e.g., "bedrock", "local_llama"
    
    # Input Details
    input_data: str  # Summarized input
    system_prompt: Optional[str] = None  # System instructions sent to LLM
    user_prompt: Optional[str] = None  # User message sent to LLM
    context_data: Optional[Dict[str, Any]] = None  # Additional context (e.g., doc_type, previous results)
    
    # Output Details
    output_data: str  # Summarized output
    raw_output: Optional[str] = None  # Complete LLM response
    
    # Execution Metadata
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    latency_ms: float
    tokens_used: Optional[int] = None
    tokens_input: Optional[int] = None
    tokens_output: Optional[int] = None
    
    # Error Tracking
    error_occurred: bool = False
    error_message: Optional[str] = None
    retry_attempt: Optional[int] = None  # Which retry attempt (if applicable)


# ==================== Final Output Schema ====================

class ProcessingResult(BaseModel):
    """Final output from the entire pipeline"""
    file_path: str
    doc_type: DocumentType
    classification: ClassificationResult
    extraction: ExtractionResult
    validation: ValidationResult
    redaction: RedactionResult
    metrics: MetricsReport
    responsible_ai_logs: List[ResponsibleAILog] = Field(default_factory=list)
    success: bool
    error_message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
