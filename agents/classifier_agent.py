"""
Classifier Agent - Classifies document type using LLM
"""
import time
import json
from datetime import datetime
from typing import Dict, Any
from graph.state import DocumentState
from schemas.document_schemas import DocumentType, ClassificationResult, ResponsibleAILog
from utils.llm_client import llm_client
from utils.config import settings
from utils.logger import logger
from prompts import CLASSIFIER_PROMPT, CLASSIFIER_SYSTEM_PROMPT

class ClassifierAgent:
    """
    Agent responsible for classifying document type
    
    Uses Claude Haiku to determine document category from raw text.
    """
    
    def __init__(self):
        self.name = "ClassifierAgent"
        logger.info(f"{self.name} initialized")
    
    def _prepare_text_excerpt(self, text: str, max_chars: int = 600) -> str:
        """
        Prepare text excerpt for classification.
        600 chars (up from 300) so Work Experience / Skills sections are visible.
        """
        if len(text) <= max_chars:
            return text
        
        # Take excerpt from beginning and end for better classification
        half = max_chars // 2
        excerpt = text[:half] + "\n...\n" + text[-half:]
        return excerpt
    
    def _parse_llm_response(self, content: str) -> Dict[str, Any]:
        """
        Parse LLM JSON response
        
        Args:
            content: LLM response content
        
        Returns:
            Parsed classification dict
        
        Raises:
            ValueError if parsing fails
        """
        try:
            # Try direct JSON parsing
            result = json.loads(content)
            return result
        except json.JSONDecodeError:
            # Try extracting JSON from markdown code blocks
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                json_str = content[json_start:json_end].strip()
                result = json.loads(json_str)
                return result
            elif "```" in content:
                json_start = content.find("```") + 3
                json_end = content.find("```", json_start)
                json_str = content[json_start:json_end].strip()
                result = json.loads(json_str)
                return result
            else:
                raise ValueError(f"Could not parse JSON from response: {content[:200]}")
    
    def _normalize_doc_type(self, doc_type_str: str) -> DocumentType:
        """
        Normalize document type string to match enum values
        
        Args:
            doc_type_str: Document type string from LLM
        
        Returns:
            Normalized DocumentType enum value
        """
        # Clean up the string
        doc_type_clean = doc_type_str.lower().strip()
        
        # Direct match attempt
        try:
            return DocumentType(doc_type_clean)
        except ValueError:
            pass
        
        # Handle common variations and aliases
        type_mapping = {
            # Resume/CV variations
            "cv": DocumentType.RESUME,
            "curriculum vitae": DocumentType.RESUME,
            "job application": DocumentType.RESUME,
            "biodata": DocumentType.RESUME,  # Indian format
            # Job offer variations
            "job offer": DocumentType.JOB_OFFER,
            "offer letter": DocumentType.JOB_OFFER,
            "employment offer": DocumentType.JOB_OFFER,
            "internship offer": DocumentType.JOB_OFFER,
            "appointment letter": DocumentType.JOB_OFFER,  # Indian format
            # Financial variations
            "financial": DocumentType.FINANCIAL_DOCUMENT,
            "invoice": DocumentType.FINANCIAL_DOCUMENT,
            "receipt": DocumentType.FINANCIAL_DOCUMENT,
            "gst invoice": DocumentType.FINANCIAL_DOCUMENT,  # Indian
            "tax invoice": DocumentType.FINANCIAL_DOCUMENT,
            "bill": DocumentType.FINANCIAL_DOCUMENT,
            # Medical variations
            "medical": DocumentType.MEDICAL_RECORD,
            "health record": DocumentType.MEDICAL_RECORD,
            "prescription": DocumentType.MEDICAL_RECORD,
            "lab report": DocumentType.MEDICAL_RECORD,
            # ID document variations
            "id": DocumentType.ID_DOCUMENT,
            "identification": DocumentType.ID_DOCUMENT,
            "passport": DocumentType.ID_DOCUMENT,
            "driver license": DocumentType.ID_DOCUMENT,
            "aadhaar": DocumentType.ID_DOCUMENT,  # Indian
            "aadhar": DocumentType.ID_DOCUMENT,  # Common misspelling
            "pan card": DocumentType.ID_DOCUMENT,  # Indian
            "id card": DocumentType.ID_DOCUMENT,
            "id_document": DocumentType.ID_DOCUMENT,
            "college id": DocumentType.ID_DOCUMENT,
            "college id card": DocumentType.ID_DOCUMENT,
            "student id": DocumentType.ID_DOCUMENT,
            "student id card": DocumentType.ID_DOCUMENT,
            "student card": DocumentType.ID_DOCUMENT,
            "employee id": DocumentType.ID_DOCUMENT,
            "employee card": DocumentType.ID_DOCUMENT,
            "national id": DocumentType.ID_DOCUMENT,
            "voter id": DocumentType.ID_DOCUMENT,
            "driving license": DocumentType.ID_DOCUMENT,
            # Contract variations
            "agreement": DocumentType.CONTRACT,
            "legal document": DocumentType.CONTRACT,
            "nda": DocumentType.CONTRACT,
            # Academic variations
            "transcript": DocumentType.ACADEMIC,
            "diploma": DocumentType.ACADEMIC,
            "certificate": DocumentType.ACADEMIC,
            "mark sheet": DocumentType.ACADEMIC,  # Indian
            "marksheet": DocumentType.ACADEMIC,
            "textbook": DocumentType.ACADEMIC,
            "course book": DocumentType.ACADEMIC,
            "reference book": DocumentType.ACADEMIC,
            "syllabus": DocumentType.ACADEMIC,
            "lecture notes": DocumentType.ACADEMIC,
        }
        
        # Try exact match in mapping
        if doc_type_clean in type_mapping:
            logger.info(f"Normalized '{doc_type_str}' to '{type_mapping[doc_type_clean].value}'")
            return type_mapping[doc_type_clean]
        
        # Try partial match (contains)
        for key, value in type_mapping.items():
            if key in doc_type_clean or doc_type_clean in key:
                logger.info(f"Partial match: normalized '{doc_type_str}' to '{value.value}'")
                return value
        
        # If no match found, default to UNKNOWN
        logger.warning(f"Could not normalize doc_type '{doc_type_str}', defaulting to UNKNOWN")
        return DocumentType.UNKNOWN
    
    def classify(self, state: DocumentState) -> DocumentState:
        """
        Classify document type
        
        Args:
            state: Current document state
        
        Returns:
            Updated state with classification result
        """
        logger.info(f"{self.name}: Starting classification")
        start_time = time.time()
        
        try:
            # Prepare prompt
            text_excerpt = self._prepare_text_excerpt(state["raw_text"])
            prompt = CLASSIFIER_PROMPT.format(
                text_excerpt=text_excerpt
            )
            
            # Call LLM with optimized settings
            # Removed force_provider - allows Groq→Claude→HF fallback if rate limit hit
            response = llm_client.generate(
                prompt=prompt,
                system_prompt=CLASSIFIER_SYSTEM_PROMPT,
                temperature=0.0,
                max_tokens=150,
                groq_model="llama-3.1-8b-instant",  # 8b-instant: fast enough for 6-category classification
                groq_key=1  # Primary key
            )
            
            latency = time.time() - start_time
            
            # Parse response
            classification = self._parse_llm_response(response["content"])
            
            # Normalize and validate doc_type
            doc_type = self._normalize_doc_type(classification["doc_type"])
            
            # Create result
            classification_result = ClassificationResult(
                doc_type=doc_type,
                confidence=float(classification.get("confidence", 0.8)),
                reasoning=classification.get("reasoning", ""),
                timestamp=datetime.utcnow()
            )
            
            # Update state
            state["doc_type"] = doc_type
            state["classification_result"] = classification_result
            state["agent_timings"][self.name] = latency
            
            # Responsible AI logging
            if state.get("trace_log") is None:
                state["trace_log"] = []
            
            state["trace_log"].append(
                ResponsibleAILog(
                    agent_name=self.name,
                    input_data=text_excerpt[:500],
                    output_data=str(classification_result.model_dump()),
                    timestamp=datetime.utcnow(),
                    latency_ms=latency * 1000,
                    llm_model_used=response["model"],
                    tokens_used=response["tokens"]["input"] + response["tokens"]["output"],
                    error_occurred=False,
                    llm_provider=response["provider"],
                    system_prompt=CLASSIFIER_SYSTEM_PROMPT,
                    user_prompt=response.get("user_prompt", ""),
                    context_data={
                        "file_path": state.get("file_path", ""),
                        "text_length": len(state.get("raw_text", "")),
                        "excerpt_length": len(text_excerpt)
                    },
                    raw_output=response["content"],
                    tokens_input=response["tokens"]["input"],
                    tokens_output=response["tokens"]["output"],
                    retry_attempt=0
                )
            )
            
            logger.info(
                f"{self.name}: Classification complete",
                doc_type=doc_type.value,
                confidence=classification_result.confidence,
                latency_ms=latency * 1000
            )
            
            return state
        
        except Exception as e:
            logger.error(f"{self.name}: Classification failed", error=str(e))
            
            # Update state with error - use UNKNOWN as fallback
            state["errors"].append(f"{self.name}: {str(e)}")
            state["doc_type"] = DocumentType.UNKNOWN  # Now valid after enum update
            state["classification_result"] = ClassificationResult(
                doc_type=DocumentType.UNKNOWN,  # Now valid after enum update
                confidence=0.0,
                reasoning=f"Classification failed: {str(e)}"
            )
            
            # Log error in trace
            if state.get("trace_log") is None:
                state["trace_log"] = []
            
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
                    system_prompt=CLASSIFIER_SYSTEM_PROMPT,
                    user_prompt="",
                    context_data={
                        "file_path": state.get("file_path", ""),
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
classifier_agent = ClassifierAgent()
