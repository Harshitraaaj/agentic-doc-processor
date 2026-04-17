"""
Extractor Agent - Extracts structured fields from documents
"""
import time
import json
from datetime import datetime
from typing import Dict, Any, Optional

from json_repair import repair_json
from langchain_text_splitters import RecursiveCharacterTextSplitter

from graph.state import DocumentState
from schemas.document_schemas import (
    DocumentType,
    ExtractionResult,
    ResponsibleAILog,
    FinancialDocumentFields,
    ResumeFields,
    JobOfferFields,
    MedicalRecordFields,
    IdDocumentFields,
    AcademicFields,
)
from utils.llm_client import llm_client
from utils.logger import logger
from utils.config import settings
from prompts import EXTRACTOR_PROMPT, EXTRACTOR_SYSTEM_PROMPT


class ExtractorAgent:
    """
    Agent responsible for extracting structured fields from documents
    
    Uses Claude Haiku + optional FAISS semantic lookup for context.
    Supports chunking for long documents.
    """
    
    # Schema definitions for each document type (OPTIMIZED - 6 CORE TYPES)
    SCHEMA_MAP = {
        DocumentType.FINANCIAL_DOCUMENT: FinancialDocumentFields,
        DocumentType.RESUME: ResumeFields,
        DocumentType.JOB_OFFER: JobOfferFields,
        DocumentType.MEDICAL_RECORD: MedicalRecordFields,
        DocumentType.ID_DOCUMENT: IdDocumentFields,
        DocumentType.ACADEMIC: AcademicFields,
    }
    
    # Few-shot examples for better extraction accuracy — one full example per doc type
    # showing EVERY schema field so the LLM knows exactly what to return.
    FEW_SHOT_EXAMPLES = {
        DocumentType.FINANCIAL_DOCUMENT: """
EXAMPLE - Financial Document (Invoice):
Input: "INVOICE #INV-2024-001 | Date: March 15, 2024 | Due: April 15, 2024
FROM: CloudTech LLC, 500 Tech Park, San Jose CA 95110
TO: Acme Corp, 200 Business Ave, New York NY 10001
Items: Cloud Hosting x1 = $1,200.00, Support Plan x1 = $300.00
Subtotal: $1,500.00 | Tax (8%): $120.00 | Total: $1,620.00
Payment Method: Bank Transfer"
Output: {
  "document_number": "INV-2024-001",
  "document_date": "2024-03-15",
  "due_date": "2024-04-15",
  "issuer_name": "Cloudtech Llc",
  "issuer_address": "500 Tech Park, San Jose CA 95110",
  "recipient_name": "Acme Corp",
  "recipient_address": "200 Business Ave, New York NY 10001",
  "total_amount": 1620.00,
  "tax_amount": 120.00,
  "currency": "USD",
  "payment_method": "Bank Transfer",
  "line_items": [
    {"description": "Cloud Hosting", "quantity": 1, "unit_price": 1200.00, "total": 1200.00},
    {"description": "Support Plan", "quantity": 1, "unit_price": 300.00, "total": 300.00}
  ]
}
""",
        DocumentType.RESUME: """
EXAMPLE - Resume / CV:
Input: "JOHN DOE | john@email.com | (555) 123-4567 | New York, NY
LinkedIn: linkedin.com/in/johndoe
SUMMARY: Experienced data scientist specializing in NLP and predictive modeling.
SKILLS: Python, ML, AWS, SQL
EDUCATION: MS Computer Science, Stanford University, May 2019, GPA: 3.9
EXPERIENCE: Senior Data Scientist, TechCorp (Jan 2020 - Present)
  - Built ML pipelines, Led team of 4 engineers
CERTIFICATIONS: AWS Certified ML Specialist, Google Cloud Professional
LANGUAGES: English (Native), Spanish (Intermediate)"
Output: {
  "candidate_name": "John Doe",
  "email": "john@email.com",
  "phone": "(555) 123-4567",
  "address": "New York, NY",
  "linkedin_url": "linkedin.com/in/johndoe",
  "summary": "Experienced data scientist specializing in NLP and predictive modeling.",
  "education": [
    {"degree": "Master Of Science In Computer Science", "institution": "Stanford University", "graduation_date": "2019-05-01", "gpa": 3.9}
  ],
  "work_experience": [
    {"job_title": "Senior Data Scientist", "employer": "Techcorp", "start_date": "2020-01-01", "end_date": null, "responsibilities": ["Built ML pipelines", "Led team of 4 engineers"]}
  ],
  "skills": ["Python", "ML", "AWS", "SQL"],
  "certifications": ["AWS Certified ML Specialist", "Google Cloud Professional"],
  "languages": ["English (Native)", "Spanish (Intermediate)"]
}
""",
        DocumentType.MEDICAL_RECORD: """
EXAMPLE - Medical Record:
Input: "Patient: Sarah Williams | DOB: 03/22/1985 | Patient ID: MED-98765
Department: Endocrinology | Visit Date: December 5, 2023
Physician: Dr. James Anderson
Diagnosis: Type 2 Diabetes Mellitus (E11.9)
Prescribed: Metformin 500mg twice daily, Lisinopril 10mg daily
Lab: HbA1c 7.2%, Fasting Glucose 145 mg/dL
Follow-up: January 10, 2024
Notes: Patient advised lifestyle modifications and dietary changes."
Output: {
  "patient_name": "Sarah Williams",
  "patient_id": "MED-98765",
  "date_of_birth": "1985-03-22",
  "visit_date": "2023-12-05",
  "physician_name": "Dr. James Anderson",
  "department": "Endocrinology",
  "diagnosis": "Type 2 Diabetes Mellitus (E11.9)",
  "prescribed_medications": ["Metformin 500mg twice daily", "Lisinopril 10mg daily"],
  "lab_results": [
    {"test": "HbA1c", "result": "7.2%"},
    {"test": "Fasting Glucose", "result": "145 mg/dL"}
  ],
  "follow_up_date": "2024-01-10",
  "notes": "Patient advised lifestyle modifications and dietary changes."
}
""",
        DocumentType.JOB_OFFER: """
EXAMPLE - Job Offer:
Input: "TechCorp Inc. | OFFER LETTER
Candidate: Michael Chen | Position: Senior Software Engineer
Start Date: January 15, 2024 | Salary: $145,000/year
Location: San Francisco, CA | Reports to: VP Engineering
Benefits: Health, Dental, Vision, 401k (4% match), 20 days PTO
Offer valid until: December 31, 2023"
Output: {
  "company_name": "TechCorp Inc.",
  "candidate_name": "Michael Chen",
  "position_title": "Senior Software Engineer",
  "start_date": "2024-01-15",
  "salary": "145000",
  "employment_type": "full-time",
  "work_location": "San Francisco, CA",
  "reporting_to": "VP Engineering",
  "department": null,
  "offer_date": null,
  "deadline_to_accept": "2023-12-31",
  "benefits": ["Health", "Dental", "Vision", "401k (4% match)", "20 days PTO"],
  "conditions": []
}
""",
        DocumentType.ID_DOCUMENT: """
EXAMPLE - Identity Document (Passport):
Input: "PASSPORT | United States of America
Surname: JOHNSON  Given Names: EMILY ROSE
Date of Birth: 15 APR 1990  Sex: F
Nationality: USA  Place of Birth: New York, NY
Personal No: 987-65-4321
Passport No: A12345678  Issued: 10 JAN 2020  Expires: 09 JAN 2030
Issuing Authority: U.S. Department of State
Address: 123 Main Street, New York, NY 10001"
Output: {
  "document_type": "Passport",
  "document_number": "A12345678",
  "full_name": "Emily Rose Johnson",
  "date_of_birth": "1990-04-15",
  "gender": "Female",
  "nationality": "American",
  "place_of_birth": "New York, NY",
  "address": "123 Main Street, New York, NY 10001",
  "issue_date": "2020-01-10",
  "expiration_date": "2030-01-09",
  "issuing_authority": "U.S. Department of State"
}
""",
        DocumentType.ACADEMIC: """
EXAMPLE - Academic Document (Transcript):
Input: "OFFICIAL TRANSCRIPT | STANFORD UNIVERSITY
Student: Alex Rivera  |  ID: STU-20190456
Program: Bachelor of Science in Computer Science
Graduation: June 15, 2023  |  GPA: 3.87 / 4.0
Honors: Magna Cum Laude, Dean's List (4 semesters)
Courses: CS101 Intro to Programming A, CS201 Data Structures A-, MATH301 Linear Algebra B+"
Output: {
  "document_type": "Transcript",
  "student_name": "Alex Rivera",
  "student_id": "STU-20190456",
  "institution_name": "Stanford University",
  "degree_program": "Bachelor of Science in Computer Science",
  "graduation_date": "2023-06-15",
  "gpa": 3.87,
  "doi": null,
  "courses": [
    {"course": "CS101 Intro to Programming", "grade": "A"},
    {"course": "CS201 Data Structures", "grade": "A-"},
    {"course": "MATH301 Linear Algebra", "grade": "B+"}
  ],
  "honors": ["Magna Cum Laude", "Dean's List (4 semesters)"]
}

EXAMPLE - Academic Document (Textbook):
Input: "TEXTBOOK | Introduction to Machine Learning (2nd Edition)
Authors: Ethem Alpaydin
Publisher: MIT Press
ISBN: 978-0262043793
Publication Year: 2020
Course: CS-540
Institution: University Department Library"
Output: {
    "document_type": "Textbook",
    "title": "Introduction to Machine Learning",
    "author": "Ethem Alpaydin",
    "isbn": "978-0262043793",
    "publication_year": 2020,
    "institution_name": "University Department Library",
    "degree_program": "CS-540"
}
"""
    }

    def __init__(self):
        self.name = "ExtractorAgent"
        logger.info(f"{self.name} initialized")
    
    def _camel_to_snake(self, name: str) -> str:
        """
        Convert camelCase or PascalCase to snake_case
        
        Args:
            name: Field name in camelCase/PascalCase
        
        Returns:
            Field name in snake_case
        """
        import re
        # Insert underscore before uppercase letters (except at start)
        snake = re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()
        return snake
    
    def _normalize_field_names(self, data: Any) -> Any:
        """
        Recursively normalize all dictionary keys from camelCase to snake_case
        
        Args:
            data: Dictionary, list, or primitive value
        
        Returns:
            Data structure with normalized field names
        """
        if isinstance(data, dict):
            # Normalize keys and recursively process values
            return {self._camel_to_snake(key): self._normalize_field_names(value) 
                    for key, value in data.items()}
        elif isinstance(data, list):
            # Recursively process list items
            return [self._normalize_field_names(item) for item in data]
        else:
            # Primitive value - return as-is
            return data
    
    def _normalize_extracted_values(self, data: Any) -> Any:
        """
        Normalize extracted values for better evaluation matching
        Handles dates, amounts, names, etc.
        
        Args:
            data: Extracted data structure
        
        Returns:
            Normalized data structure
        """
        from datetime import datetime
        import re
        
        if isinstance(data, dict):
            # Recursively normalize dict values
            return {key: self._normalize_extracted_values(value) for key, value in data.items()}
        
        elif isinstance(data, list):
            # Recursively normalize list items
            return [self._normalize_extracted_values(item) for item in data]
        
        elif isinstance(data, str):
            # Normalize string values
            
            # 1. Date normalization: Try to parse and format as YYYY-MM-DD
            date_formats = [
                '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d',
                '%B %d, %Y', '%b %d, %Y', '%B %d %Y', '%b %d %Y',
                '%d %B %Y', '%d %b %Y', '%Y-%m-%dT%H:%M:%S'
            ]
            for fmt in date_formats:
                try:
                    parsed_date = datetime.strptime(data.strip(), fmt)
                    # Return in standard YYYY-MM-DD format
                    return parsed_date.strftime('%Y-%m-%d')
                except (ValueError, AttributeError):
                    continue
            
            # 2. Name normalization: Title Case for names
            # Check if this looks like a name (contains only letters and spaces)
            if data.strip() and all(c.isalpha() or c.isspace() for c in data.strip()):
                # Convert to Title Case
                return data.strip().title()
            
            # 3. Return cleaned string (strip whitespace)
            return data.strip()
        
        else:
            # Return primitive values as-is (int, float, bool, None)
            return data
    
    def _chunk_text(self, text: str, max_length: int = 6000, overlap: int = 500) -> list[str]:
        """
        Split text into chunks using LangChain's RecursiveCharacterTextSplitter.
        Splits on natural boundaries (paragraphs → lines → words) before falling
        back to character-level splitting, with configurable overlap.

        Args:
            text: Full text
            max_length: Maximum chunk size in characters (default 6000)
            overlap: Overlap between consecutive chunks (default 500)

        Returns:
            List of text chunks with overlap
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_length,
            chunk_overlap=overlap,
            length_function=len,
        )
        chunks = splitter.split_text(text)
        logger.info(f"Text split into {len(chunks)} chunks with {overlap}-char overlap")
        return chunks
    
    def _extract_from_chunk(
        self,
        chunk: str,
        doc_type: DocumentType,
        force_provider=None,
        effective_label: Optional[str] = None,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Extract fields from a single text chunk

        Args:
            chunk: Text chunk
            doc_type: Document type
            effective_label: Override label for LLM prompt (custom doc types typed by human)

        Returns:
            Tuple of (extracted fields dict, llm response dict)
        """
        # Resolve the label used in the prompt:
        # - custom human-typed type (effective_label) takes priority
        # - otherwise use the enum value (e.g. "resume", "financial_document")
        prompt_label = effective_label or doc_type.value

        # Few-shot examples only exist for known enum types
        few_shot_examples = self.FEW_SHOT_EXAMPLES.get(doc_type, "")

        prompt = EXTRACTOR_PROMPT.format(
            doc_type=prompt_label,
            few_shot_examples=few_shot_examples,
            document_text=chunk
        )
        
        response = llm_client.generate(
            prompt=prompt,
            system_prompt=EXTRACTOR_SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=500 if settings.BEDROCK_ONLY_MODE else 1200,
            force_provider=force_provider,
            groq_model="llama-3.3-70b-versatile",
            groq_key=3  # Tertiary key for heavy extraction load
        )
        
        # Parse JSON response using json-repair
        # Handles: preamble text, markdown fences, double braces, truncated/unclosed JSON
        content = response["content"].strip()
        logger.debug(f"Extractor LLM response length: {len(content)} chars")

        try:
            repaired = repair_json(content, return_objects=True)
            if not isinstance(repaired, dict):
                raise ValueError(f"repair_json returned {type(repaired).__name__}, expected dict")
            logger.debug("Parsed JSON using repair_json")
            return repaired, response
        except Exception as e:
            logger.error(f"Failed to parse extraction response: {e}")
            logger.error(f"First 500 chars: {content[:500]}")
            raise ValueError(f"Could not parse JSON from response: {e}")
    
    def _merge_extracted_fields(
        self,
        chunks_results: list[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Merge extraction results from multiple chunks
        
        Args:
            chunks_results: List of extraction results from chunks
        
        Returns:
            Merged fields dict
        """
        if len(chunks_results) == 1:
            return chunks_results[0]
        
        # Simple merge strategy: take non-null values, prefer later chunks
        merged = {}
        for chunk_result in chunks_results:
            for key, value in chunk_result.items():
                if value is not None:
                    if isinstance(value, list):
                        # Extend lists
                        # Extend lists only when existing value is list-like; otherwise replace.
                        existing = merged.get(key)
                        if isinstance(existing, list):
                            existing.extend(value)
                        elif existing is None:
                            merged[key] = list(value)
                        else:
                            merged[key] = list(value)
                    else:
                        # Overwrite with non-null values
                        merged[key] = value
        
        return merged
    
    def extract(self, state: DocumentState) -> DocumentState:
        """
        Extract structured fields from document
        
        Args:
            state: Current document state
        
        Returns:
            Updated state with extraction result
        """
        logger.info(f"{self.name}: Starting extraction")
        start_time = time.time()
        
        try:
            doc_type = state["doc_type"]
            raw_text = state["raw_text"]

            # When the human typed a custom doc type (not in the enum), use that
            # label in the LLM prompt so extraction targets the right document kind.
            custom_doc_type = state.get("custom_doc_type")  # e.g. "bank_statement"
            effective_label = (
                custom_doc_type
                if (doc_type == DocumentType.UNKNOWN and custom_doc_type)
                else None
            )
            if effective_label:
                logger.info(
                    f"{self.name}: Custom doc type '{effective_label}' — "
                    "using open-ended extraction"
                )
            
            text_excerpt = raw_text[:500]
            
            # Chunk text if needed
            chunks = self._chunk_text(raw_text)
            logger.info(f"Processing {len(chunks)} chunks")
            
            # Extract from each chunk
            chunk_results = []
            llm_responses = []
            total_tokens = 0
            
            # Get schema definition only needed for self-repair — not sent in initial extraction
            from utils.llm_client import LLMProvider
            for i, chunk in enumerate(chunks, 1):
                logger.debug(f"Extracting from chunk {i}/{len(chunks)}")
                extracted, response = self._extract_from_chunk(
                    chunk=chunk,
                    doc_type=doc_type,
                    effective_label=effective_label,
                )
                chunk_results.append(extracted)
                llm_responses.append(response)
                total_tokens += response["tokens"]["input"] + response["tokens"]["output"]
            
            # Merge results
            merged_fields = self._merge_extracted_fields(chunk_results)
            
            # Normalize field names: camelCase -> snake_case
            normalized_fields = self._normalize_field_names(merged_fields)
            
            # Normalize values: dates to YYYY-MM-DD, names to Title Case, etc.
            normalized_fields = self._normalize_extracted_values(normalized_fields)
            
            # Apply FIELD_ALIASES to catch any remaining field name variations
            # e.g. "company" → "company_name", "role" → "position_title"
            from agents.validator_agent import ValidatorAgent
            aliases = ValidatorAgent.FIELD_ALIASES
            aliased_fields = {}
            for key, value in normalized_fields.items():
                canonical = aliases.get(key.lower(), key)
                # Always store under the canonical name.
                # A non-null value may overwrite a previously stored null for the same canonical.
                if canonical not in aliased_fields or value not in (None, "", [], {}):
                    aliased_fields[canonical] = value
            normalized_fields = aliased_fields
            
            latency = time.time() - start_time
            
            # Use last response for logging (representative)
            last_response = llm_responses[-1] if llm_responses else {}
            
            # Create result
            extraction_result = ExtractionResult(
                doc_type=doc_type,
                extracted_fields=normalized_fields,
                confidence=0.85,  # TODO: compute actual confidence
                chunk_count=len(chunks),
                timestamp=datetime.utcnow()
            )
            
            # Update state
            state["extracted_fields"] = normalized_fields
            state["extraction_result"] = extraction_result
            state["agent_timings"][self.name] = latency
            
            # Responsible AI logging
            state["trace_log"].append(
                ResponsibleAILog(
                    agent_name=self.name,
                    input_data=text_excerpt,
                    output_data=json.dumps(merged_fields),
                    timestamp=datetime.utcnow(),
                    latency_ms=latency * 1000,
                    llm_model_used=last_response.get("model", "claude-haiku"),
                    tokens_used=total_tokens,
                    error_occurred=False,
                    llm_provider=last_response.get("provider", "unknown"),
                    system_prompt=EXTRACTOR_SYSTEM_PROMPT,
                    user_prompt=last_response.get("user_prompt", ""),
                    context_data={
                        "doc_type": effective_label or doc_type.value,
                        "chunk_count": len(chunks),
                        "text_length": len(raw_text)
                    },
                    raw_output=last_response.get("content", ""),
                    tokens_input=sum(r["tokens"]["input"] for r in llm_responses),
                    tokens_output=sum(r["tokens"]["output"] for r in llm_responses),
                    retry_attempt=0
                )
            )
            
            logger.info(
                f"{self.name}: Extraction complete",
                doc_type=effective_label or doc_type.value,
                fields_count=len(merged_fields),
                chunks=len(chunks),
                latency_ms=latency * 1000
            )
            
            return state
        
        except Exception as e:
            logger.error(f"{self.name}: Extraction failed", error=str(e))
            
            # Update state with error
            state["errors"].append(f"{self.name}: {str(e)}")
            state["extracted_fields"] = {}
            state["extraction_result"] = ExtractionResult(
                doc_type=state["doc_type"],
                extracted_fields={},
                confidence=0.0,
                chunk_count=0
            )
            
            # Log error
            state["trace_log"].append(
                ResponsibleAILog(
                    agent_name=self.name,
                    input_data=state["raw_text"][:500],
                    output_data="",
                    timestamp=datetime.utcnow(),
                    latency_ms=(time.time() - start_time) * 1000,
                    llm_model_used="unknown",
                    error_occurred=True,
                    error_message=str(e),
                    llm_provider="unknown",
                    system_prompt=EXTRACTOR_SYSTEM_PROMPT,
                    user_prompt="",
                    context_data={
                        "doc_type": state.get("doc_type", DocumentType.FINANCIAL_DOCUMENT).value if state.get("doc_type") else "unknown",
                        "text_length": len(state.get("raw_text", ""))
                    },
                    raw_output="",
                    tokens_input=0,
                    tokens_output=0,
                    retry_attempt=0
                )
            )
            
            return state


# Agent instance
extractor_agent = ExtractorAgent()
