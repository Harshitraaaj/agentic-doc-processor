"""
Self-Repair Node - Fixes invalid JSON using LLM
"""
import time
import json
from datetime import datetime
from typing import Dict, Any

from json_repair import repair_json

from graph.state import DocumentState
from schemas.document_schemas import ValidationStatus, ResponsibleAILog
from utils.llm_client import llm_client
from utils.config import settings
from utils.logger import logger
from prompts import SELF_REPAIR_PROMPT, SELF_REPAIR_RE_EXTRACTION_PROMPT, SELF_REPAIR_SYSTEM_PROMPT


class SelfRepairNode:
    """
    Self-repair node that uses LLM to fix validation errors or re-extract fields
    
    Two modes:
    1. Repair Mode: Fix validation errors in existing extracted fields
    2. Re-extraction Mode: Extract fields when none/few were extracted initially
    """
    
    def __init__(self):
        self.name = "SelfRepairNode"
        # Import schema map here to avoid circular imports
        from agents.extractor_agent import ExtractorAgent
        self.schema_map = ExtractorAgent.SCHEMA_MAP
        logger.info(f"{self.name} initialized")
        # Load config values (use utils.config.Env for direct instantiation)
        from utils.config import Env
        self.env = Env()
        self.max_attempts = int(self.env.get('workflow', 'max_repair_attempts', fallback=3))
        self.llm_model = 'llama-3.3-70b-versatile'
        self.llm_key = 3
    
    def _parse_llm_response(self, content: str) -> Dict[str, Any]:
        """
        Parse LLM JSON response using repair_json.
        Handles markdown fences, preamble text, double braces, truncated JSON.
        """
        try:
            result = repair_json(content, return_objects=True)
            if isinstance(result, dict):
                logger.debug("Parsed repair response using repair_json")
                return result
            raise ValueError(f"repair_json returned {type(result).__name__}, expected dict")
        except Exception as e:
            logger.error(f"repair_json failed for self-repair response: {e}")
            logger.error(f"First 500 chars: {content[:500]}")
            raise ValueError(f"Could not parse JSON from repair response: {e}")
    
    def _get_schema_fields(self, doc_type) -> str:
        """
        Get schema field names for document type
        
        Args:
            doc_type: Document type
        
        Returns:
            Formatted string of expected field names
        """
        from schemas.document_schemas import DocumentType
        
        schema_class = self.schema_map.get(doc_type)
        if not schema_class:
            return "No specific schema available"
        
        # Get field names from Pydantic model
        try:
            fields = schema_class.model_fields.keys()
            return "\n".join(f"  - {field}" for field in fields)
        except:
            return "Unable to extract schema fields"
    
    def _should_re_extract(self, extracted_fields: Dict[str, Any], state: Dict[str, Any]) -> bool:
        """
        Determine if fields are too sparse or below accuracy threshold — needs re-extraction.

        Triggers re-extraction when:
        - fewer than 3 non-null fields  (always), OR
        - validator stored accuracy < 90 % in state
        """
        if not extracted_fields:
            return True

        # Count non-null fields
        non_null_fields = sum(1 for v in extracted_fields.values() if v not in [None, "", [], {}])
        if non_null_fields < 3:
            logger.info(f"Only {non_null_fields} non-null fields extracted, triggering re-extraction")
            return True

        # Accuracy-based trigger
        current_accuracy = state.get("current_accuracy", 1.0)
        if current_accuracy < 0.90:
            logger.info(
                f"Accuracy {current_accuracy:.0%} below 90% threshold, triggering re-extraction"
            )
            return True

        return False
    
    def repair(self, state: DocumentState) -> DocumentState:
        """
        Attempt to repair invalid extracted fields
        
        Args:
            state: Current document state
        
        Returns:
            Updated state with repaired fields
        """
        logger.info(f"{self.name}: Starting self-repair")
        start_time = time.time()
        # Use config-driven max_attempts
        current_attempts = state.get("repair_attempts", 0)
        if current_attempts >= self.max_attempts:
            logger.warning(f"{self.name}: Max repair attempts reached ({self.max_attempts})")
            state["validation_status"] = ValidationStatus.FAILED
            state["validation_errors"].append("Max repair attempts reached")
            return state
        state["repair_attempts"] = current_attempts + 1
        logger.info(f"{self.name}: Repair attempt {state['repair_attempts']} of {self.max_attempts}")
        
        try:
            extracted_fields = state["extracted_fields"] or {}
            validation_result = state.get("validation_result")
            errors = validation_result.errors if validation_result else []
            doc_type = state["doc_type"]
            
            # Determine if we need re-extraction or just repair
            needs_re_extraction = self._should_re_extract(extracted_fields, state)
            
            # Prepare text excerpt (longer for re-extraction)
            text_length = 1000 if needs_re_extraction else 500
            text_excerpt = " ".join(state.get("raw_text", state.get("text", "")).split()[:text_length])
            
            if needs_re_extraction:
                logger.info(f"{self.name}: Using RE-EXTRACTION mode (accuracy below 90% or sparse fields)")
                # Get schema fields for guidance
                schema_fields = self._get_schema_fields(doc_type)

                # Retrieve missing field list stored by validator (or compute fresh)
                missing_fields = state.get("missing_schema_fields", [])
                current_accuracy = state.get("current_accuracy", 0.0)
                accuracy_pct = int(round(current_accuracy * 100))

                if missing_fields:
                    missing_fields_list = "\n".join(f"  - {f}" for f in missing_fields)
                else:
                    missing_fields_list = "  (unknown — extract all schema fields)"

                prompt = SELF_REPAIR_RE_EXTRACTION_PROMPT.format(
                    doc_type=doc_type.value if hasattr(doc_type, 'value') else str(doc_type),
                    current_accuracy=accuracy_pct,
                    document_text=text_excerpt,
                    schema_fields=schema_fields,
                    extracted_fields=json.dumps(extracted_fields, indent=2),
                    missing_fields_list=missing_fields_list,
                    validation_errors="\n".join(f"- {error}" for error in errors) if errors else "No specific errors - fields are missing"
                )
            else:
                logger.info(f"{self.name}: Using REPAIR mode (fixing validation errors)")
                prompt = SELF_REPAIR_PROMPT.format(
                    extracted_fields=json.dumps(extracted_fields, indent=2),
                    errors="\n".join(f"- {error}" for error in errors),
                    text_excerpt=text_excerpt
                )
            
            # Use config-driven LLM model and key
            response = llm_client.generate(
                prompt=prompt,
                system_prompt=SELF_REPAIR_SYSTEM_PROMPT,
                temperature=0.0,
                max_tokens=500 if settings.BEDROCK_ONLY_MODE else 1200,
                groq_model=self.llm_model,
                groq_key=self.llm_key
            )
            
            latency = time.time() - start_time
            
            # Log response for debugging
            logger.debug(f"{self.name}: LLM response length: {len(response['content'])} chars")
            
            # Parse repaired fields
            try:
                repaired_fields = self._parse_llm_response(response["content"])
            except ValueError as parse_error:
                logger.error(f"{self.name}: JSON parsing failed: {parse_error}")
                logger.error(f"{self.name}: Full LLM response:\n{response['content']}")
                raise
            
            # Update state
            state["extracted_fields"] = repaired_fields
            state["repair_attempts"] = current_attempts + 1
            state["validation_status"] = ValidationStatus.REPAIRED.value
            state["agent_timings"][self.name] = latency
            
            # Add repair info to state
            if "repair_history" not in state:
                state["repair_history"] = []
            state["repair_history"].append({
                "attempt": current_attempts + 1,
                "errors_fixed": errors,
                "mode": "re_extraction" if needs_re_extraction else "repair"
            })
            
            # Responsible AI logging
            state["trace_log"].append(
                ResponsibleAILog(
                    agent_name=self.name,
                    input_data=json.dumps(extracted_fields),
                    output_data=json.dumps(repaired_fields),
                    timestamp=datetime.utcnow(),
                    latency_ms=latency * 1000,
                    llm_model_used=response["model"],
                    tokens_used=response["tokens"]["input"] + response["tokens"]["output"],
                    error_occurred=False,
                    llm_provider=response["provider"],
                    system_prompt=SELF_REPAIR_SYSTEM_PROMPT,
                    user_prompt=response.get("user_prompt", ""),
                    context_data={
                        "repair_attempt": current_attempts + 1,
                        "max_attempts": self.max_attempts,
                        "validation_errors": errors,
                        "field_count": len(extracted_fields),
                        "mode": "re_extraction" if needs_re_extraction else "repair",
                        "current_accuracy": state.get("current_accuracy", 0.0),
                        "missing_fields": state.get("missing_schema_fields", []),
                        "doc_type": doc_type.value if hasattr(doc_type, 'value') else str(doc_type)
                    },
                    raw_output=response["content"],
                    tokens_input=response["tokens"]["input"],
                    tokens_output=response["tokens"]["output"],
                    retry_attempt=current_attempts + 1
                )
            )
            
            logger.info(
                f"{self.name}: Repair complete",
                attempt=current_attempts + 1,
                latency_ms=latency * 1000
            )
            
            # Route back to validator
            state["needs_repair"] = False  # Will be re-evaluated by validator
            
            return state
        
        except Exception as e:
            logger.error(f"{self.name}: Repair failed", error=str(e))
            
            # Update state with error
            state["errors"].append(f"{self.name}: {str(e)}")
            state["repair_attempts"] = current_attempts + 1
            state["needs_repair"] = False  # Stop trying after error
            
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
                    system_prompt=SELF_REPAIR_SYSTEM_PROMPT,
                    user_prompt="",
                    context_data={
                        "repair_attempt": current_attempts + 1,
                        "validation_errors": state.get("validation_result", {}).errors if hasattr(state.get("validation_result", {}), "errors") else []
                    },
                    raw_output="",
                    tokens_input=0,
                    tokens_output=0,
                    retry_attempt=current_attempts + 1
                )
            )
            
            return state


# Node instance
self_repair_node = SelfRepairNode()
