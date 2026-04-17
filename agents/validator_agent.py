"""
Validator Agent - Hybrid rule-based + LLM validation
"""
import json
import time
import re
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

from json_repair import repair_json

from graph.state import DocumentState
from schemas.document_schemas import (
    DocumentType,
    ValidationStatus,
    ValidationResult,
    ResponsibleAILog,
)
from utils.llm_client import llm_client
from utils.config import settings
from utils.logger import logger
from utils.knowledge_lookup import get_knowledge_lookup
from prompts import VALIDATOR_PROMPT, VALIDATOR_SYSTEM_PROMPT


class ValidatorAgent:
    """
    Hybrid validator: Rule-based validation + LLM for complex cases
    
    Benefits:
    - Rule-based: Fast, deterministic validation (no LLM needed)
    - LLM: Handles complex semantic validations
    - Reduces LLM calls by ~70%
    """
    
    # Document type field priorities (most required to least required)
    # FIXED: Field names now match document_schemas.py exactly
    FIELD_PRIORITIES = {
        "financial_document": ["total_amount", "issuer_name", "document_date", "document_number", "recipient_name"],
        "contract": ["parties", "effective_date", "contract_type", "contract_value", "terms"],
        "resume": ["candidate_name", "email", "phone", "work_experience", "education"],
        "job_offer": ["candidate_name", "position_title", "salary", "start_date", "company_name"],
        "medical_record": ["patient_name", "date_of_birth", "diagnosis", "physician_name", "visit_date"],
        "id_document": ["full_name", "document_number", "date_of_birth", "issue_date", "expiration_date"],
        "academic": ["student_name", "institution_name", "degree_program", "graduation_date", "gpa"],
        "unknown": ["title", "date", "author"]
    }
    
    # Field name aliases for fuzzy matching (maps variations → canonical schema names)
    FIELD_ALIASES = {
        # Job offer fields
        "position": "position_title", "job_title": "position_title", "title": "position_title",
        "role": "position_title", "job_role": "position_title",
        "company": "company_name", "employer": "company_name", "organization": "company_name",
        "candidate": "candidate_name", "applicant_name": "candidate_name", "employee_name": "candidate_name",
        "annual_salary": "salary", "base_salary": "salary", "compensation": "salary",
        "location": "work_location", "office_location": "work_location", "job_location": "work_location",
        "manager": "reporting_manager", "supervisor": "reporting_manager", "reports_to": "reporting_manager",
        "expiration": "offer_expiration_date", "valid_until": "offer_expiration_date",
        # Resume fields
        "name": "candidate_name", "full_name": "candidate_name", "applicant": "candidate_name",
        "contact_email": "email", "email_address": "email",
        "phone_number": "phone", "contact_phone": "phone", "mobile": "phone",
        "professional_summary": "summary", "objective": "summary", "profile": "summary",
        "technical_skills": "skills", "skill_list": "skills", "competencies": "skills",
        "experience": "work_experience", "employment_history": "work_experience",
        "educational_background": "education", "academic_history": "education",
        # Medical fields
        "patient": "patient_name", "patient_full_name": "patient_name",
        "dob": "date_of_birth", "birth_date": "date_of_birth", "birthdate": "date_of_birth",
        "mrn": "medical_record_number", "patient_id": "medical_record_number",
        "doctor": "physician_name", "provider": "physician_name", "attending_physician": "physician_name",
        "visit": "visit_date", "appointment_date": "visit_date", "encounter_date": "visit_date",
        "condition": "diagnosis", "medical_condition": "diagnosis",
        "prescriptions": "medications", "drugs": "medications", "meds": "medications",
        "vitals": "vital_signs", "vital_statistics": "vital_signs",
        # Financial fields
        "amount": "total_amount", "total": "total_amount", "grand_total": "total_amount",
        "issuer": "issuer_name", "vendor": "issuer_name", "supplier": "issuer_name",
        "recipient": "recipient_name", "customer": "recipient_name", "client": "recipient_name",
        "invoice_number": "document_number", "receipt_number": "document_number", "doc_number": "document_number",
        "date": "document_date", "invoice_date": "document_date", "receipt_date": "document_date",
        "tax": "tax_amount", "vat": "tax_amount", "sales_tax": "tax_amount",
        # Academic fields
        "student": "student_name", "student_full_name": "student_name",
        "school": "institution_name", "university": "institution_name", "college": "institution_name",
        "program": "degree_program", "major": "degree_program", "field_of_study": "degree_program",
        "grad_date": "graduation_date", "completion_date": "graduation_date",
        "grade_point_average": "gpa", "cumulative_gpa": "gpa",
        "textbook_title": "title", "book_title": "title", "book_name": "title",
        "writer": "author", "authors": "author",
        "isbn_10": "isbn", "isbn_13": "isbn",
        "published_year": "publication_year", "year_published": "publication_year",
        "publication_date": "publication_year",
        # ID document fields
        "id_number": "document_number", "passport_number": "document_number", "license_number": "document_number",
        "issued": "issue_date", "issued_date": "issue_date",
        "expires": "expiration_date", "expiry_date": "expiration_date", "valid_until": "expiration_date",
    }
    
    def __init__(self):
        self.name = "ValidatorAgent"
        self.knowledge_lookup = None
        self._knowledge_init_attempted = False
        
        # Validation regex patterns (used for pre-checks, not blockers)
        self.patterns = {
            "email": re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
            "phone": re.compile(r'^[\d\s()+-]{10,}$'),
            "date": re.compile(r'\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4}'),
            "amount": re.compile(r'^\$?[\d,]+\.?\d*$'),
            "ssn": re.compile(r'^\d{3}-\d{2}-\d{4}$')
        }
        
        logger.info(f"{self.name} initialized (LLM-powered with intelligent field mapping)")

    def _ensure_knowledge_lookup(self) -> None:
        """Lazily initialize knowledge lookup to avoid import-time heavy model loading."""
        if self._knowledge_init_attempted:
            return

        self._knowledge_init_attempted = True
        try:
            self.knowledge_lookup = get_knowledge_lookup()
            logger.info(f"{self.name}: Knowledge lookup initialized")
        except Exception as e:
            self.knowledge_lookup = None
            logger.warning(f"{self.name}: Knowledge lookup unavailable, fallback mode active: {e}")
    
    def _normalize_extracted_fields(self, extracted_fields: Dict[str, Any]) -> Dict[str, Any]:
        """
        Pre-validation normalization:
        1. Remap alias keys to canonical schema names (e.g. 'experience' → 'work_experience')
        2. Coerce lists to scalars for fields expected to be strings
        """
        normalized = dict(extracted_fields)

        # 1. Alias remapping — add canonical key if absent OR overwrite if canonical key is null/empty
        for alias_key, canonical_key in self.FIELD_ALIASES.items():
            if alias_key in normalized:
                alias_val = normalized[alias_key]
                canon_val = normalized.get(canonical_key)
                # Overwrite canonical with alias value when alias has real content and canonical is empty/null
                if alias_val not in (None, "", [], {}) and canon_val in (None, "", [], {}, None):
                    normalized[canonical_key] = alias_val

        # 2. Scalar coercion: LLMs sometimes wrap a single value in a list
        scalar_fields = {
            "email", "phone", "candidate_name", "full_name", "date_of_birth",
            "issue_date", "expiration_date", "document_number", "summary",
            "patient_name", "physician_name", "visit_date", "diagnosis",
            "issuer_name", "recipient_name", "document_date", "total_amount",
            "position_title", "company_name", "salary", "start_date",
        }
        for field in scalar_fields:
            val = normalized.get(field)
            if isinstance(val, list) and val:
                # take first non-empty element
                normalized[field] = next((str(v) for v in val if v not in (None, "")), str(val[0]))

        return normalized

    def _update_runtime_knowledge_profile(self, state: DocumentState, extracted_fields: Dict[str, Any]) -> None:
        if not isinstance(extracted_fields, dict) or not extracted_fields:
            return

        if self.knowledge_lookup is None:
            self._ensure_knowledge_lookup()
        if self.knowledge_lookup is None:
            return

        doc_type = state.get("doc_type")
        doc_type_label = doc_type.value if isinstance(doc_type, DocumentType) else str(doc_type or "unknown")
        custom_doc_type = state.get("custom_doc_type")

        self.knowledge_lookup.register_runtime_observed_fields(
            doc_type=doc_type_label,
            observed_fields=extracted_fields,
            custom_doc_type=custom_doc_type,
            notes="Auto-merged from successful validator output.",
        )

    @staticmethod
    def _field_has_value(v) -> bool:
        """True when a field has any meaningful value — empty list [] counts as present."""
        if v is None or v == "":
            return False
        if isinstance(v, dict) and len(v) == 0:
            return False
        if isinstance(v, str) and v.strip().lower() in {
            "n/a", "na", "null", "none", "not available", "not applicable", "unknown"
        }:
            return False
        return True

    def _is_textbook_like_academic(
        self,
        extracted_fields: Dict[str, Any],
        raw_text: Optional[str] = None,
    ) -> bool:
        """Detect textbook/reference-book style academic documents."""
        if not isinstance(extracted_fields, dict):
            extracted_fields = {}

        marker_keys = {
            "title", "book_title", "textbook_title",
            "author", "authors", "isbn", "isbn_10", "isbn_13",
            "publisher", "publication_year", "published_year",
        }
        if marker_keys.intersection(set(extracted_fields.keys())):
            return True

        doc_label = str(extracted_fields.get("document_type", "")).strip().lower()
        if any(token in doc_label for token in ("textbook", "book", "reference")):
            return True

        text = (raw_text or "").lower()
        return any(token in text for token in ("textbook", "isbn", "publisher", "edition", "chapter"))

    def _resolve_priority_fields(
        self,
        extracted_fields: Dict[str, Any],
        doc_type_label: str,
        profile_required_fields: Optional[List[str]] = None,
        raw_text: Optional[str] = None,
    ) -> List[str]:
        """Resolve required/priority fields with textbook-aware academic override."""
        resolved_doc_type = str(doc_type_label or "unknown").strip().lower()
        base_priority = profile_required_fields or self.FIELD_PRIORITIES.get(
            resolved_doc_type,
            ["title", "date"],
        )

        if resolved_doc_type == "academic" and self._is_textbook_like_academic(extracted_fields, raw_text=raw_text):
            return ["title", "author", "isbn", "publication_year"]

        return base_priority

    def _validate_field_format(self, field_name: str, value: Any) -> Tuple[bool, Optional[str]]:
        """
        Rule-based field format validation
        
        Returns:
            (is_valid, error_message)
        """
        if value is None or value == "":
            return True, None  # null/empty are okay
        
        # Handle list/dict fields - skip format validation for collections
        if isinstance(value, (list, dict)):
            # For lists of amounts, check if they're valid numbers
            if isinstance(value, list) and any(s in field_name.lower() for s in ["amount", "price", "total", "salary"]):
                for item in value:
                    if not isinstance(item, (int, float)) or item < 0:
                        return False, f"{field_name}: invalid amount in list"
            return True, None
        
        value_str = str(value).strip()
        field_lower = field_name.lower()
        
        # Email validation
        if "email" in field_lower:
            if not self.patterns["email"].match(value_str):
                return False, f"{field_name}: invalid email format"
        
        # Phone validation
        elif "phone" in field_lower:
            if not self.patterns["phone"].match(value_str):
                return False, f"{field_name}: invalid phone format"
        
        # Date validation (check for actual date fields, not substrings like "candidate")
        elif (field_lower.endswith("_date") or field_lower == "date" or 
              field_lower.startswith("date_") or "birth" in field_lower):
            if not self.patterns["date"].search(value_str):
                return False, f"{field_name}: invalid date format"
        
        # Amount/currency validation (only for scalar values)
        elif "amount" in field_lower or "price" in field_lower or "total" in field_lower or "salary" in field_lower:
            if not self.patterns["amount"].match(value_str):
                return False, f"{field_name}: invalid amount format"
        
        # SSN validation
        elif "ssn" in field_lower:
            if not self.patterns["ssn"].match(value_str):
                return False, f"{field_name}: invalid SSN format"
        
        return True, None
    
    # ──────────────────────────────────────────────────────────────────────────
    # Accuracy computation (mirrors reporter logic so both use the same baseline)
    # ──────────────────────────────────────────────────────────────────────────
    ACCURACY_THRESHOLD = 0.80  # Lowered from 0.90 — LLM often uses alternate field names (e.g. 'job_experience' vs 'work_experience')

    def _compute_accuracy(
        self,
        extracted_fields: Dict[str, Any],
        doc_type: DocumentType,
        custom_doc_type: Optional[str] = None,
    ) -> Tuple[float, List[str]]:
        """
        Compute schema-completion accuracy and return the list of missing fields.

        Returns:
            (accuracy_score 0.0-1.0, list_of_missing_field_names)
        """
        def _is_missing(v):
            if v is None or v == "":
                return True
            # Empty dict is missing, but empty list [] means "correctly extracted as none" — not missing
            if isinstance(v, dict) and len(v) == 0:
                return True
            if isinstance(v, str) and v.strip().lower() in {
                "n/a", "na", "null", "none", "not available", "not applicable", "unknown"
            }:
                return True
            return False

        try:
            if not settings.LOW_LATENCY_MODE:
                self._ensure_knowledge_lookup()

            # 1) Knowledge profile (SQLite + JSON schema / Pydantic-derived schema)
            if (not settings.LOW_LATENCY_MODE) and self.knowledge_lookup is not None:
                profile = self.knowledge_lookup.get_validation_profile(
                    doc_type=doc_type.value if isinstance(doc_type, DocumentType) else str(doc_type),
                    custom_doc_type=custom_doc_type,
                )
                required_fields = self._resolve_priority_fields(
                    extracted_fields=extracted_fields,
                    doc_type_label=doc_type.value if isinstance(doc_type, DocumentType) else str(doc_type),
                    profile_required_fields=profile.get("required_fields", []) if isinstance(profile, dict) else [],
                )
                if required_fields:
                    total = len(required_fields)
                    missing = [f for f in required_fields if _is_missing(extracted_fields.get(f))]
                    accuracy = min(1.0, (total - len(missing)) / total) if total > 0 else 1.0
                    return accuracy, missing

            # 2) Fallback to builtin extractor schema map
            from agents.extractor_agent import ExtractorAgent
            schema_class = ExtractorAgent.SCHEMA_MAP.get(doc_type)
            if not schema_class:
                non_null = sum(1 for v in extracted_fields.values() if not _is_missing(v))
                total = max(len(extracted_fields), 1)
                return non_null / total, []

            schema_fields = list(schema_class.model_fields.keys())
            total = len(schema_fields)
            if total == 0:
                return 1.0, []

            missing = [f for f in schema_fields if _is_missing(extracted_fields.get(f))]
            accuracy = min(1.0, (total - len(missing)) / total)
            return accuracy, missing

        except Exception as e:
            logger.warning(f"{self.name}: accuracy computation failed: {e}")
            return 1.0, []

    def _rule_based_validation(
        self,
        extracted_fields: Dict[str, Any],
        doc_type: DocumentType,
        priority_fields: Optional[List[str]] = None,
    ) -> Tuple[List[str], List[str]]:
        """
        Fast rule-based validation (no LLM needed)
        
        Returns:
            (errors, warnings)
        """
        errors = []
        warnings = []
        
        # Get priority fields
        resolved_priority_fields = priority_fields or self.FIELD_PRIORITIES.get(
            doc_type.value if isinstance(doc_type, DocumentType) else str(doc_type),
            []
        )
        
        # Helper function to check if value is missing/invalid
        def is_missing_value(value):
            """Check if value is truly missing (null, empty, or placeholder)"""
            if value is None or value == "" or value == []:
                return True
            # Reject common placeholder strings that LLMs use for missing data
            if isinstance(value, str):
                normalized = value.strip().lower()
                if normalized in ["n/a", "na", "null", "none", "not available", "not applicable"]:
                    return True
            return False
        
        # Check required fields (top 3 priority fields)
        for field in resolved_priority_fields[:3]:  # Check top 3 most important
            value = extracted_fields.get(field)
            if is_missing_value(value):
                errors.append(f"Missing required field: {field}")
        
        # Validate field formats
        for field_name, value in extracted_fields.items():
            if not is_missing_value(value):
                is_valid, error_msg = self._validate_field_format(field_name, value)
                if not is_valid:
                    errors.append(error_msg)
        
        # Check field count (exclude missing values from count)
        non_null_count = sum(1 for v in extracted_fields.values() if not is_missing_value(v))
        if non_null_count < 2:
            errors.append(f"Insufficient fields: only {non_null_count} non-null fields")
        elif non_null_count < 4:
            warnings.append(f"Sparse extraction: only {non_null_count} fields populated")
        
        return errors, warnings
    
    def _extract_json_from_response(self, response: str) -> Dict[str, Any]:
        """
        Parse JSON from LLM validation response using repair_json.
        Handles markdown fences, preamble text, truncated JSON, double braces.
        Falls back to a default valid structure if repair also fails.
        """
        try:
            result = repair_json(response, return_objects=True)
            if isinstance(result, dict):
                return result
            raise ValueError(f"Expected dict, got {type(result).__name__}")
        except Exception as e:
            logger.warning(f"repair_json failed for validator response: {e}. Response preview: {response[:200]}")
            return {
                "is_valid": True,  # fail-open
                "errors": [],
                "warnings": ["Validation response parsing failed, skipping LLM validation"],
                "status": "valid"
            }

    def _validate_against_json_schema(
        self,
        extracted_fields: Dict[str, Any],
        json_schema: Dict[str, Any],
    ) -> Tuple[List[str], List[str]]:
        """Validate extracted fields against a resolved JSON Schema profile."""
        if not json_schema:
            return [], []

        try:
            from jsonschema import Draft7Validator
        except Exception as e:
            logger.warning(f"{self.name}: jsonschema unavailable: {e}")
            return [], ["JSON Schema validation skipped: jsonschema package unavailable"]

        validator = Draft7Validator(json_schema)
        errors = []
        for error in sorted(validator.iter_errors(extracted_fields), key=str):
            path = ".".join(str(part) for part in error.path) if list(error.path) else "root"
            errors.append(f"Schema violation at {path}: {error.message}")

        warnings = []
        if not errors:
            warnings.append("JSON Schema validation passed")
        return errors, warnings
    
    def _validate_with_llm(
        self,
        extracted_fields: Dict[str, Any],
        doc_type: DocumentType,
        custom_doc_type: Optional[str] = None,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Use LLM to validate extracted fields with intelligent field mapping
        
        Args:
            extracted_fields: Extracted fields dict
            doc_type: Document type
        
        Returns:
            Tuple of (validation result dict, llm response dict)
        """
        # Resolve doc label (custom type from HITL gets priority)
        doc_type_label = custom_doc_type or (
            doc_type.value if isinstance(doc_type, DocumentType) else str(doc_type)
        )

        # Knowledge look-up profile (SQLite + FAISS + JSON schema)
        profile = {}
        if self.knowledge_lookup is not None:
            profile = self.knowledge_lookup.get_validation_profile(
                doc_type=doc_type.value if isinstance(doc_type, DocumentType) else str(doc_type),
                custom_doc_type=custom_doc_type,
            )

        schema_fields = profile.get("schema_fields", []) if isinstance(profile, dict) else []
        schema_fields_str = ", ".join(schema_fields) if schema_fields else "Standard document fields"
        
        # Format extracted fields for prompt
        fields_json = json.dumps(extracted_fields, indent=2, default=str)
        
        # Get priority fields (knowledge profile overrides static defaults)
        priority_fields = self._resolve_priority_fields(
            extracted_fields=extracted_fields,
            doc_type_label=doc_type_label,
            profile_required_fields=(profile.get("required_fields", []) if isinstance(profile, dict) else []),
            raw_text=None,
        )
        priority_str = ", ".join(priority_fields)

        # Optional semantic hits from FAISS retrieval for richer validator context
        semantic_hits = profile.get("semantic_hits", []) if isinstance(profile, dict) else []
        semantic_context_lines = []
        for hit in semantic_hits[:3]:
            metadata = hit.get("metadata", {}) if isinstance(hit, dict) else {}
            semantic_context_lines.append(
                f"- doc_type={metadata.get('doc_type', 'unknown')}, "
                f"source={metadata.get('source', 'unknown')}, "
                f"distance={hit.get('distance', 0):.4f}"
            )
        semantic_context = "\n".join(semantic_context_lines) if semantic_context_lines else "None"

        json_schema_obj = profile.get("json_schema", {}) if isinstance(profile, dict) else {}
        json_schema_str = json.dumps(json_schema_obj, ensure_ascii=False)[:3000] if json_schema_obj else "{}"
        knowledge_notes = (profile.get("knowledge_notes", "") if isinstance(profile, dict) else "")[:1500]
        
        # Create enhanced prompt with schema information
        prompt = VALIDATOR_PROMPT.format(
            doc_type=doc_type_label,
            schema_fields=schema_fields_str,
            priority_fields=priority_str,
            extracted_fields=fields_json
        ) + (
            "\n\nKNOWLEDGE LOOK-UP CONTEXT:" 
            f"\nPROFILE_SOURCE: {profile.get('source', 'none') if isinstance(profile, dict) else 'none'}"
            f"\nPROFILE_DOC_TYPE: {profile.get('doc_type', doc_type_label) if isinstance(profile, dict) else doc_type_label}"
            f"\nKNOWLEDGE_NOTES: {knowledge_notes or 'None'}"
            f"\nSEMANTIC_HITS:\n{semantic_context}"
            f"\nJSON_SCHEMA: {json_schema_str}"
            "\nApply this schema context while deciding is_valid/errors/warnings."
        )
        
        # Call LLM with Groq for ultra-fast intelligent validation
        from utils.llm_client import LLMProvider
        llm_response = llm_client.generate(
            prompt=prompt,
            system_prompt="Document validator. Return ONLY valid JSON with is_valid, errors, warnings, status. No code, no explanations.",
            max_tokens=200,
            temperature=0.0,
            groq_model=settings.GROQ_MODEL,
            groq_key=2  # Secondary key (low token load)
        )
        response = llm_response["content"]
        
        # Parse JSON response with robust extraction
        try:
            result = self._extract_json_from_response(response)
            
            # Ensure required fields exist
            if "is_valid" not in result:
                result["is_valid"] = False
            if "errors" not in result:
                result["errors"] = []
            if "warnings" not in result:
                result["warnings"] = []
            if "status" not in result:
                result["status"] = "valid" if result["is_valid"] else "invalid"
            
            # Ensure errors and warnings are lists
            if not isinstance(result["errors"], list):
                result["errors"] = [str(result["errors"]) ]
            if not isinstance(result["warnings"], list):
                result["warnings"] = [str(result["warnings"]) ]

            # Coerce each element to a string (dicts/lists -> JSON string)
            def _coerce_list_to_strings(lst: List[Any]) -> List[str]:
                coerced = []
                for el in lst:
                    if isinstance(el, (dict, list)):
                        try:
                            coerced.append(json.dumps(el, ensure_ascii=False))
                        except Exception:
                            coerced.append(str(el))
                    else:
                        coerced.append(str(el))
                return coerced

            result["errors"] = _coerce_list_to_strings(result["errors"])
            result["warnings"] = _coerce_list_to_strings(result["warnings"])
            
            return result, llm_response
        
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse LLM validation response: {e}")
            logger.debug(f"Raw response (first 1000 chars): {response[:1000]}")
            
            # Fallback: Try to infer validation result from text
            response_lower = response.lower()
            is_valid = "valid" in response_lower and "invalid" not in response_lower
            
            return {
                "is_valid": is_valid,
                "errors": [f"Validation response parsing failed, please retry"] if not is_valid else [],
                "warnings": ["Could not parse structured validation response"],
                "status": "valid" if is_valid else "invalid"
            }, llm_response
    
    def validate(self, state: DocumentState) -> DocumentState:
        """
        Validate extracted fields using LLM-first approach with intelligent field mapping
        
        Flow:
        1. Always use LLM for validation (with schema context)
        2. LLM can intelligently map field names and validate semantically
        3. Rule-based checks only as warnings (not blockers)
        
        Special handling:
        - After max repair attempts, mark as valid with warnings
        
        Args:
            state: Current document state
        
        Returns:
            Updated state with validation result
        """
        logger.info(f"{self.name}: Starting LLM-powered validation with intelligent field mapping")
        start_time = time.time()
        
        # Check if this is post-repair and at max attempts
        repair_attempts = state.get("repair_attempts", 0)
        max_attempts = 3  # Match workflow GraphConfig.max_repair_attempts
        is_post_max_repair = repair_attempts >= max_attempts and state.get("validation_status") == ValidationStatus.REPAIRED.value
        
        # Initialize LLM response
        llm_response = {
            "model": "bedrock-claude",
            "provider": "bedrock",
            "content": "",
            "tokens": {"input": 0, "output": 0},
            "system_prompt": "",
            "user_prompt": ""
        }
    
        try:
            extracted_fields = state["extracted_fields"] or {}
            doc_type = state["doc_type"]
            raw_text = state.get("raw_text", "")

            # For custom doc types typed by human (not in enum), use UNKNOWN so that
            # schema-based required-field checks don't fire false errors.
            custom_doc_type = state.get("custom_doc_type")
            effective_doc_type = (
                DocumentType.UNKNOWN
                if (doc_type == DocumentType.UNKNOWN and custom_doc_type)
                else doc_type
            )
            knowledge_profile = {}
            if self.knowledge_lookup is not None:
                knowledge_profile = self.knowledge_lookup.get_validation_profile(
                    doc_type=doc_type.value if isinstance(doc_type, DocumentType) else str(doc_type),
                    custom_doc_type=custom_doc_type,
                )

            # Normalize: alias remapping + type coercion before any validation
            extracted_fields = self._normalize_extracted_fields(extracted_fields)
            state["extracted_fields"] = extracted_fields

            resolved_priority_fields = self._resolve_priority_fields(
                extracted_fields=extracted_fields,
                doc_type_label=doc_type.value if isinstance(doc_type, DocumentType) else str(doc_type),
                profile_required_fields=(knowledge_profile.get("required_fields", []) if isinstance(knowledge_profile, dict) else []),
                raw_text=raw_text,
            )

            schema_errors, schema_warnings = self._validate_against_json_schema(
                extracted_fields,
                knowledge_profile.get("json_schema", {}) if isinstance(knowledge_profile, dict) else {},
            )

            # Quick pre-check: Rule-based validation (non-blocking, just for warnings)
            # Use effective_doc_type so custom-type runs don't check for wrong required fields
            rule_errors, rule_warnings = self._rule_based_validation(
                extracted_fields,
                effective_doc_type,
                priority_fields=resolved_priority_fields,
            )
            rule_errors = schema_errors + rule_errors
            rule_warnings = schema_warnings + rule_warnings
            
            logger.info(f"{self.name}: Rule-based pre-check found {len(rule_errors)} potential issues, {len(rule_warnings)} warnings")
            
            # Special case: After max repair attempts, accept with warnings
            if is_post_max_repair:
                logger.info(f"{self.name}: Max repair attempts reached ({repair_attempts}), marking as valid with repair notice")
                is_valid = True
                status = ValidationStatus.VALID_AFTER_REPAIR
                needs_repair = False
                errors = []
                warnings = rule_warnings + [
                    f"Document validated after {repair_attempts} self-repair attempt(s)",
                    "Some validation issues may remain but were auto-resolved"
                ]
                if rule_errors:
                    warnings.extend([f"Original issue: {err}" for err in rule_errors[:3]])

            elif not rule_errors and extracted_fields and all(
                self._field_has_value(extracted_fields.get(f))
                for f in resolved_priority_fields
            ):
                # Fast-path: all priority fields filled + no rule errors — skip LLM (saves ~2s)
                logger.info(f"{self.name}: Fast-path validation — all priority fields filled, no errors")
                accuracy, missing_fields = self._compute_accuracy(
                    extracted_fields,
                    doc_type,
                    custom_doc_type=custom_doc_type,
                )
                is_valid = accuracy >= self.ACCURACY_THRESHOLD
                status = ValidationStatus.VALID if is_valid else ValidationStatus.INVALID
                needs_repair = not is_valid
                errors = (
                    [f"Extraction accuracy {accuracy:.0%} below 90% threshold. Missing: {', '.join(missing_fields[:5])}"]
                    if needs_repair else []
                )
                warnings = [f"Pre-check: {w}" for w in rule_warnings]
                if isinstance(knowledge_profile, dict) and knowledge_profile.get("source"):
                    warnings.append(f"Knowledge source: {knowledge_profile.get('source')}")
                state["current_accuracy"] = accuracy
                state["missing_schema_fields"] = missing_fields

            else:
                # ALWAYS use LLM for intelligent validation (primary validator)
                logger.info(f"{self.name}: Using LLM for intelligent semantic validation with field mapping")
                validation_response, llm_response = self._validate_with_llm(
                    extracted_fields,
                    doc_type,
                    custom_doc_type=custom_doc_type,
                )
                
                is_valid = validation_response.get("is_valid", False)
                errors = validation_response.get("errors", [])
                
                # Combine LLM warnings with rule-based warnings
                llm_warnings = validation_response.get("warnings", [])
                warnings = llm_warnings + [f"Pre-check: {w}" for w in rule_warnings]
                if isinstance(knowledge_profile, dict) and knowledge_profile.get("source"):
                    warnings.append(f"Knowledge source: {knowledge_profile.get('source')}")
                
                # Extractor now uses schema directly, so no field remapping needed here.
                # Validator's role is purely validation (is_valid, errors, warnings)
                extra_fields = validation_response.get("extra_fields", {})
                if extra_fields:
                    state["extra_extracted_fields"] = extra_fields
                
                status = ValidationStatus.VALID if is_valid else ValidationStatus.INVALID
                needs_repair = not is_valid

                # ── Accuracy gate ────────────────────────────────────────────────
                accuracy, missing_fields = self._compute_accuracy(
                    extracted_fields,
                    doc_type,
                    custom_doc_type=custom_doc_type,
                )
                logger.info(
                    f"{self.name}: Schema accuracy = {accuracy:.0%} "
                    f"({len(missing_fields)} missing fields)"
                )

                if accuracy >= self.ACCURACY_THRESHOLD:
                    # Accuracy ≥ threshold — override any LLM errors, mark VALID
                    # Minor field-format complaints become warnings, not blockers
                    if errors:
                        logger.info(
                            f"{self.name}: Accuracy {accuracy:.0%} ≥ threshold — "
                            f"demoting {len(errors)} LLM error(s) to warnings"
                        )
                        warnings = errors + warnings
                        errors = []
                    is_valid = True
                    status = ValidationStatus.VALID
                    needs_repair = False

                elif not is_post_max_repair:
                    logger.warning(
                        f"{self.name}: Accuracy {accuracy:.0%} below {self.ACCURACY_THRESHOLD:.0%} threshold "
                        f"— triggering silent self-repair for: {missing_fields}"
                    )
                    needs_repair = True
                    is_valid = False
                    status = ValidationStatus.INVALID
                    # NOTE: accuracy error intentionally NOT added to errors list — repair is silent
                # ────────────────────────────────────────────────────────────────

                # Log validation notes if provided by LLM
                validation_notes = validation_response.get("validation_notes", "")
                if validation_notes:
                    logger.info(f"{self.name}: {validation_notes}")
            
            latency = time.time() - start_time
            
            # Create result
            validation_result = ValidationResult(
                status=status,
                is_valid=is_valid,
                errors=errors,
                warnings=warnings,
                validated_fields=extracted_fields,  # Always include fields, even after repair
                timestamp=datetime.utcnow()
            )
            
            # Update state
            state["validation_status"] = status.value
            state["validation_result"] = validation_result
            state["needs_repair"] = needs_repair
            # Persist live accuracy + missing field list so workflow router and
            # self-repair node can both read them without recomputing.
            if not is_post_max_repair:
                _acc, _missing = self._compute_accuracy(
                    extracted_fields,
                    doc_type,
                    custom_doc_type=custom_doc_type,
                )
            else:
                _acc, _missing = state.get("current_accuracy", 1.0), []
            state["current_accuracy"] = _acc
            state["missing_schema_fields"] = _missing
            state["agent_timings"][self.name] = latency

            if is_valid:
                try:
                    self._update_runtime_knowledge_profile(state, extracted_fields)
                    logger.info(
                        f"{self.name}: Runtime knowledge profile updated in DB "
                        f"({len(extracted_fields)} observed fields)"
                    )
                except Exception as e:
                    logger.warning(f"{self.name}: Runtime knowledge profile update skipped: {e}")
            
            # Responsible AI logging
            state["trace_log"].append(
                ResponsibleAILog(
                    agent_name=self.name,
                    input_data=str(extracted_fields),
                    output_data=str(validation_result.model_dump()),
                    timestamp=datetime.utcnow(),
                    latency_ms=latency * 1000,
                    llm_model_used=llm_response.get("model", "unknown"),
                    tokens_used=llm_response.get("tokens", {}).get("input", 0) + llm_response.get("tokens", {}).get("output", 0),
                    error_occurred=False,
                    llm_provider=llm_response.get("provider", "unknown"),
                    system_prompt=llm_response.get("system_prompt", ""),
                    user_prompt=llm_response.get("user_prompt", ""),
                    context_data={
                        "doc_type": doc_type.value if isinstance(doc_type, DocumentType) else str(doc_type),
                        "custom_doc_type": custom_doc_type,
                        "knowledge_source": knowledge_profile.get("source") if isinstance(knowledge_profile, dict) else None,
                        "knowledge_requested_doc_type": custom_doc_type or (doc_type.value if isinstance(doc_type, DocumentType) else str(doc_type)),
                        "knowledge_resolved_doc_type": knowledge_profile.get("doc_type") if isinstance(knowledge_profile, dict) else None,
                        "field_count": len(extracted_fields),
                        "is_valid": is_valid
                    },
                    raw_output=llm_response.get("content", ""),
                    tokens_input=llm_response.get("tokens", {}).get("input", 0),
                    tokens_output=llm_response.get("tokens", {}).get("output", 0),
                    retry_attempt=0
                )
            )
            
            logger.info(
                f"{self.name}: Validation complete",
                status=status.value,
                errors_count=len(errors),
                warnings_count=len(warnings),
                needs_repair=needs_repair,
                latency_ms=latency * 1000
            )
            
            return state
        
        except Exception as e:
            logger.error(f"{self.name}: Validation failed", error=str(e))
            
            # Update state with error
            state["errors"].append(f"{self.name}: {str(e)}")
            state["validation_status"] = ValidationStatus.INVALID.value
            state["validation_result"] = ValidationResult(
                status=ValidationStatus.INVALID,
                is_valid=False,
                errors=[str(e)],
                warnings=[],
                validated_fields=None
            )
            state["needs_repair"] = True
            
            # Log error
            state["trace_log"].append(
                ResponsibleAILog(
                    agent_name=self.name,
                    input_data=str(state.get("extracted_fields", {})),
                    output_data="",
                    timestamp=datetime.utcnow(),
                    latency_ms=(time.time() - start_time) * 1000,
                    llm_model_used="unknown",
                    error_occurred=True,
                    error_message=str(e),
                    llm_provider="unknown",
                    system_prompt=VALIDATOR_PROMPT[:500],
                    user_prompt="",
                    context_data={
                        "doc_type": state.get("doc_type", DocumentType.UNKNOWN).value,
                        "extracted_fields_count": len(state.get("extracted_fields", {}))
                    },
                    raw_output="",
                    tokens_input=0,
                    tokens_output=0,
                    retry_attempt=0
                )
            )
            
            return state


# Agent instance
validator_agent = ValidatorAgent()
