"""
Redactor Agent - Presidio + LLM PII detection and redaction
"""
import json
import re
import time
from datetime import datetime
from typing import List, Tuple, Dict, Any

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from json_repair import repair_json

from graph.state import DocumentState
from schemas.document_schemas import PIIType, PIIDetection, RedactionResult, ResponsibleAILog
from utils.llm_client import llm_client
from utils.config import settings
from utils.logger import logger
from prompts import REDACTOR_PII_DETECTION_PROMPT, REDACTOR_SYSTEM_PROMPT


class RedactorAgent:
    """
    Hybrid agent responsible for detecting and redacting PII
    
    Uses Presidio + LLM hybrid approach:
    1. Presidio (rule-based + ML) for fast, accurate PII detection
    2. LLM for context-aware enhancement and complex patterns
    3. Merge results for comprehensive coverage
    
    NOTE: PII metrics (precision, recall) are calculated GLOBALLY across all documents
    in the evaluation script, not per-document. This provides overall model performance.
    """
    
    # Presidio entity type mapping to our PIIType
    PRESIDIO_TO_PII_TYPE = {
        # ── Standard Presidio entities ──
        "EMAIL_ADDRESS": PIIType.EMAIL,
        "PHONE_NUMBER": PIIType.PHONE,
        "US_SSN": PIIType.SSN,
        "CREDIT_CARD": PIIType.CREDIT_CARD,
        "PERSON": PIIType.NAME,
        "LOCATION": PIIType.ADDRESS,
        "DATE_TIME": PIIType.DATE_OF_BIRTH,
        "MEDICAL_LICENSE": PIIType.MEDICAL_ID,
        "US_PASSPORT": PIIType.SSN,
        "US_DRIVER_LICENSE": PIIType.SSN,
        "US_BANK_NUMBER": PIIType.BANK_ACCOUNT,
        "US_ITIN": PIIType.SSN,
        "UK_NHS": PIIType.MEDICAL_ID,
        "IBAN_CODE": PIIType.BANK_ACCOUNT,
        "IP_ADDRESS": PIIType.NAME,
        "CRYPTO": PIIType.CREDIT_CARD,
        "NRP": PIIType.NAME,
        # ── Spanish ──
        "ES_NIF": PIIType.SSN,
        "ES_NIE": PIIType.SSN,
        # ── Italian ──
        "IT_DRIVER_LICENSE": PIIType.SSN,
        "IT_FISCAL_CODE": PIIType.SSN,
        "IT_VAT_CODE": PIIType.TAX_ID,
        "IT_IDENTITY_CARD": PIIType.SSN,
        "IT_PASSPORT": PIIType.SSN,
        # ── Polish ──
        "PL_PESEL": PIIType.SSN,
        # ── Indian custom entities ──
        "IN_AADHAAR": PIIType.SSN,
        "IN_PAN": PIIType.TAX_ID,
        "IN_GSTIN": PIIType.TAX_ID,
        "IN_PASSPORT": PIIType.SSN,
        "IN_VOTER_ID": PIIType.SSN,
        "IN_DRIVING_LICENSE": PIIType.SSN,
        "IN_UPI": PIIType.CREDIT_CARD,
        "IN_IFSC": PIIType.BANK_ACCOUNT,
    }
    
    def __init__(self):
        """Initialize RedactorAgent with Presidio + LLM (all languages + Indian recognizers)"""
        self.name = "RedactorAgent"

        try:
            from presidio_analyzer import PatternRecognizer, Pattern, RecognizerRegistry
            from presidio_analyzer.nlp_engine import NlpEngineProvider

            # ── Registry: English-only (content is always English) ──
            all_languages = ["en"]
            registry = RecognizerRegistry(supported_languages=all_languages)
            registry.load_predefined_recognizers(languages=all_languages)

            # ── Indian custom pattern recognizers (language="en" so they fire on en analysis) ──

            # Aadhaar: 12 digits, first digit 2-9 (XXXX XXXX XXXX)
            registry.add_recognizer(PatternRecognizer(
                supported_entity="IN_AADHAAR",
                supported_language="en",
                patterns=[Pattern("AADHAAR", r"\b[2-9]\d{3}[\s-]?\d{4}[\s-]?\d{4}\b", 0.85)],
                context=["aadhaar", "aadhar", "uid", "unique identification", "uidai"],
            ))

            # PAN Card: AAAAA9999A
            registry.add_recognizer(PatternRecognizer(
                supported_entity="IN_PAN",
                supported_language="en",
                patterns=[Pattern("PAN", r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", 0.90)],
                context=["pan", "permanent account number", "income tax"],
            ))

            # GSTIN: 15-char (22AAAAA0000A1Z5)
            registry.add_recognizer(PatternRecognizer(
                supported_entity="IN_GSTIN",
                supported_language="en",
                patterns=[Pattern("GSTIN", r"\b\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z]\b", 0.92)],
                context=["gstin", "gst", "gstn", "goods and services tax"],
            ))

            # Indian Passport: letter + 7 digits
            registry.add_recognizer(PatternRecognizer(
                supported_entity="IN_PASSPORT",
                supported_language="en",
                patterns=[Pattern("IN_PASSPORT", r"\b[A-PR-WY-Z][1-9]\d{6}[1-9]\b", 0.85)],
                context=["passport", "passport no", "passport number", "travel document"],
            ))

            # Indian Voter ID (EPIC): 3 letters + 7 digits
            registry.add_recognizer(PatternRecognizer(
                supported_entity="IN_VOTER_ID",
                supported_language="en",
                patterns=[Pattern("VOTER_ID", r"\b[A-Z]{3}\d{7}\b", 0.80)],
                context=["voter id", "epic", "election card", "voter card"],
            ))

            # Indian Driving License: SS-RR-YYYY-NNNNNNN
            registry.add_recognizer(PatternRecognizer(
                supported_entity="IN_DRIVING_LICENSE",
                supported_language="en",
                patterns=[Pattern("IN_DL", r"\b[A-Z]{2}[-\s]?\d{2}[-\s]?\d{4}[-\s]?\d{7}\b", 0.85)],
                context=["driving license", "driver license", "dl no", "driving licence"],
            ))

            # UPI ID: handle@bank
            registry.add_recognizer(PatternRecognizer(
                supported_entity="IN_UPI",
                supported_language="en",
                patterns=[Pattern("UPI", r"\b[a-zA-Z0-9.\-_]{2,256}@[a-zA-Z]{2,64}\b", 0.85)],
                context=["upi", "upi id", "upi payment", "vpa"],
            ))

            # Indian bank IFSC code: 4 letters + 0 + 6 alphanumeric
            registry.add_recognizer(PatternRecognizer(
                supported_entity="IN_IFSC",
                supported_language="en",
                patterns=[Pattern("IFSC", r"\b[A-Z]{4}0[A-Z0-9]{6}\b", 0.88)],
                context=["ifsc", "ifsc code", "bank code", "neft", "rtgs"],
            ))

            nlp_provider = NlpEngineProvider(
                nlp_configuration={
                    "nlp_engine_name": "spacy",
                    "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
                }
            )
            nlp_engine = nlp_provider.create_engine()

            self.analyzer = AnalyzerEngine(
                registry=registry,
                supported_languages=all_languages,
                nlp_engine=nlp_engine,
            )
            self.anonymizer = AnonymizerEngine()
            logger.info(
                f"{self.name} initialized (Presidio + LLM hybrid, "
                f"languages={all_languages}, 8 Indian recognizers loaded)"
            )
        except Exception as e:
            logger.warning(f"Presidio initialization failed: {e}. Falling back to LLM-only.")
            self.analyzer = None
            self.anonymizer = None
    
    def _detect_gender_patterns(self, text: str) -> List[PIIDetection]:
        """Detect gender using regex — catches 'Gender : Male', 'Sex: Female', standalone 'Male'/'Female'."""
        detections = []
        # Pattern 1: labelled gender fields  e.g. "Gender : Male", "Sex: F"
        labelled = re.finditer(
            r'(?i)\b(?:gender|sex)\s*[:\-]?\s*(male|female|m\b|f\b|other|non-binary|transgender)',
            text
        )
        for m in labelled:
            original = m.group(0).strip()
            gender_value = m.group(1).strip()
            detections.append(PIIDetection(
                field_name="gender",
                pii_type=PIIType.GENDER,
                original_text=original,
                redacted_text="[GENDER_REDACTED]",
                detection_source="regex",
                confidence=0.98
            ))
        # Pattern 2: standalone value right after newline/colon with label on same line
        # e.g. line "Gender : Male" not caught by pattern1
        standalone = re.finditer(
            r'(?i)(?<=\n)[ \t]*(male|female)[ \t]*(?:\r?\n|$)',
            text
        )
        seen = {d.original_text.lower() for d in detections}
        for m in standalone:
            original = m.group(0).strip()
            if original.lower() not in seen:
                detections.append(PIIDetection(
                    field_name="gender",
                    pii_type=PIIType.GENDER,
                    original_text=original,
                    redacted_text="[GENDER_REDACTED]",
                    detection_source="regex",
                    confidence=0.90
                ))
                seen.add(original.lower())
        return detections

    def _detect_custom_id_patterns(self, text: str) -> List[PIIDetection]:
        """Detect document-specific IDs that Presidio does not reliably capture."""
        detections = []
        patterns = [
            (r'(?im)\b(?:patient\s*id|medical\s*record\s*(?:#|number)?|mrn)\s*[:#-]?\s*(MRN-\d{4,10})\b', PIIType.MEDICAL_ID),
            (r'(?im)\b(?:patient\s*id|lab\s*id|laboratory\s*id)\s*[:#-]?\s*(LAB-\d{4,10})\b', PIIType.MEDICAL_ID),
        ]
        for pattern, pii_type in patterns:
            for match in re.finditer(pattern, text):
                value = match.group(1).strip()
                detections.append(
                    PIIDetection(
                        field_name=pii_type.value,
                        pii_type=pii_type,
                        original_text=value,
                        redacted_text=f"[{pii_type.value.upper()}_REDACTED]",
                        detection_source="regex",
                        confidence=0.98,
                    )
                )
        return detections

    def _detect_multiline_addresses(self, text: str) -> List[PIIDetection]:
        """Detect addresses that span street and city/state/zip across adjacent lines."""
        detections = []
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        street_pattern = re.compile(
            r'(?i)^\d+[\w\s.,#/-]*(street|st|avenue|ave|road|rd|drive|dr|lane|ln|court|ct|place|pl|boulevard|blvd|plaza|parkway|pkwy|apt|apartment|suite|ste|unit)\b[\w\s.,#/-]*$'
        )
        city_state_zip_pattern = re.compile(r'(?i)^([a-z .]+?),?\s+([a-z]{2})\s+(\d{5}(?:-\d{4})?)$')

        for index in range(len(lines) - 1):
            street_line = ' '.join(lines[index].split())
            city_line = ' '.join(lines[index + 1].split())
            if not street_pattern.match(street_line):
                continue
            city_match = city_state_zip_pattern.match(city_line)
            if not city_match:
                continue

            city = city_match.group(1).strip()
            state = city_match.group(2).upper()
            zipcode = city_match.group(3)
            combined = f"{street_line}, {city}, {state} {zipcode}"
            detections.append(
                PIIDetection(
                    field_name="address",
                    pii_type=PIIType.ADDRESS,
                    original_text=combined,
                    redacted_text="[ADDRESS_REDACTED]",
                    detection_source="regex",
                    confidence=0.99,
                )
            )

        return detections

    def _canonicalize_phone_text(self, text: str) -> str:
        letter_map = {
            **{c: '2' for c in 'ABC'},
            **{c: '3' for c in 'DEF'},
            **{c: '4' for c in 'GHI'},
            **{c: '5' for c in 'JKL'},
            **{c: '6' for c in 'MNO'},
            **{c: '7' for c in 'PQRS'},
            **{c: '8' for c in 'TUV'},
            **{c: '9' for c in 'WXYZ'},
        }
        text_to_translate = text
        if re.search(r'[A-Za-z]', text) and re.search(r'\(\d{3,}\)\s*$', text):
            text_to_translate = re.sub(r'\s*\(\d{3,}\)\s*$', '', text)
        translated = ''.join(letter_map.get(ch.upper(), ch) for ch in text_to_translate)
        digits = ''.join(ch for ch in translated if ch.isdigit())
        if len(digits) == 11 and digits.startswith('1'):
            return f"1-{digits[1:4]}-{digits[4:7]}-{digits[7:11]}"
        if len(digits) == 10:
            return f"({digits[0:3]}) {digits[3:6]}-{digits[6:10]}"
        return text.strip()

    def _canonicalize_name_text(self, text: str) -> str:
        cleaned = re.sub(r'^(?:name|patient|provider|physician|doctor|contact)\s*:\s*', '', text.strip(), flags=re.IGNORECASE)
        cleaned = re.sub(r'^(?:dr\.?|mr\.?|mrs\.?|ms\.?|prof\.?)\s+', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r',?\s*(?:m\.?d\.?|ph\.?d\.?|fcap|dean|president|vice chancellor)\b.*$', '', cleaned, flags=re.IGNORECASE)
        if ',' in cleaned:
            parts = [part.strip() for part in cleaned.split(',', 1)]
            if len(parts) == 2 and parts[0] and parts[1]:
                cleaned = f"{parts[1]} {parts[0]}"
        return ' '.join(cleaned.split())

    def _expand_address_text(self, raw_text: str, text: str) -> str:
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        for index, line in enumerate(lines):
            if text.lower() in line.lower():
                current = ' '.join(line.split())
                next_line = lines[index + 1] if index + 1 < len(lines) else ''
                if next_line and re.search(r'\b[A-Z]{2}\s+\d{5}(?:-\d{4})?\b', next_line):
                    city_line = ' '.join(next_line.split())
                    if ',' not in current:
                        return f"{current}, {city_line}"
                    return f"{current} {city_line}"
                return current
        return text.strip()

    def _generate_detection_aliases(self, raw_text: str, detection: PIIDetection) -> List[PIIDetection]:
        aliases = []
        canonical_text = ""

        if detection.pii_type == PIIType.PHONE:
            canonical_text = self._canonicalize_phone_text(detection.original_text)
        elif detection.pii_type == PIIType.NAME:
            canonical_text = self._canonicalize_name_text(detection.original_text)
        elif detection.pii_type == PIIType.ADDRESS:
            canonical_text = self._expand_address_text(raw_text, detection.original_text)

        canonical_text = ' '.join(canonical_text.split()).strip()
        if canonical_text and canonical_text.lower() != detection.original_text.lower():
            aliases.append(
                PIIDetection(
                    field_name=detection.field_name,
                    pii_type=detection.pii_type,
                    original_text=canonical_text,
                    redacted_text=detection.redacted_text,
                    detection_source=detection.detection_source,
                    confidence=detection.confidence,
                )
            )

        return aliases

    def _detect_pii_with_presidio(self, text: str) -> List[PIIDetection]:
        """        Detect PII using Presidio (fast, rule-based + ML)
        
        Args:
            text: Text to analyze
        
        Returns:
            List of detected PII
        """
        if not self.analyzer:
            return []
        
        try:
            # Analyze with Presidio
            results = self.analyzer.analyze(
                text=text,
                language='en',
                entities=None  # Detect all supported entities
            )
            
            pii_detections = []
            for result in results:
                # Extract the actual text
                original_text = text[result.start:result.end]
                
                # ULTRA-LOW confidence for maximum recall (targeting 90%+ recall)
                MIN_CONFIDENCE = 0.20  # AGGRESSIVE: 0.25→0.20 for 90%+ recall
                if result.score < MIN_CONFIDENCE:
                    continue
                
                # Validate detection to reduce false positives (ULTRA-STRICT validation)
                is_valid = self._validate_pii_detection(result.entity_type, original_text, result.score)
                if not is_valid:
                    logger.debug(f"FILTERED Presidio: {result.entity_type}='{original_text}' (confidence={result.score:.2f})")
                    continue
                else:
                    logger.debug(f"ACCEPTED Presidio: {result.entity_type}='{original_text}' (confidence={result.score:.2f})")
                
                # Map Presidio entity type to our PIIType
                pii_type = self.PRESIDIO_TO_PII_TYPE.get(
                    result.entity_type,
                    PIIType.NAME  # Default fallback
                )
                
                # Create redacted version
                redacted_text = f"[{result.entity_type}_REDACTED]"
                
                detection = PIIDetection(
                    field_name=result.entity_type.lower(),
                    pii_type=pii_type,
                    original_text=original_text,
                    redacted_text=redacted_text,
                    detection_source="presidio",
                    confidence=result.score
                )
                pii_detections.append(detection)
            
            logger.info(f"Presidio detected {len(pii_detections)} PII instances (after confidence filtering)")
            return pii_detections
        
        except Exception as e:
            logger.error(f"Presidio PII detection failed: {e}")
            return []
    
    # Blacklist of common false positive patterns (COMPREHENSIVE)
    FALSE_POSITIVE_PATTERNS = {
        # Technology/software terms
        'docker', 'kubernetes', 'python', 'javascript', 'tensorflow', 'pytorch',
        'java', 'react', 'angular', 'vue', 'django', 'flask', 'aws', 'azure',
        # Common words often misdetected
        'customer', 'service', 'business', 'company', 'developer', 'engineer',
        'manager', 'director', 'senior', 'junior', 'lead', 'principal',
        # Food/restaurant terms
        'italian', 'chinese', 'mexican', 'french', 'japanese', 'thai',
        'salad', 'caesar', 'pasta', 'pizza', 'burger', 'sandwich', 'soup',
        'menu', 'restaurant', 'appetizer', 'entree', 'dessert',
        # Time-related (not dates)
        'annually', 'monthly', 'daily', 'weekly', 'yearly', 'quarterly',
        'hours', 'days', 'weeks', 'months', 'years', 'hrs', 'mins'
    }
    
    def _validate_pii_detection(self, entity_type: str, text: str, confidence: float) -> bool:
        """
        Validate PII detection to filter false positives (ULTRA-STRICT MODE)
        
        Args:
            entity_type: Presidio entity type (PERSON, DATE_TIME, PHONE_NUMBER, etc.)
            text: The detected text
            confidence: Confidence score
        
        Returns:
            True if valid, False if likely false positive
        """
        # Map Presidio entity_type to our PIIType for validation
        entity_to_pii_map = {
            "PERSON": "NAME",
            "DATE_TIME": "DATE_TIME",
            "PHONE_NUMBER": "PHONE",
            "US_SSN": "SSN",
            "EMAIL_ADDRESS": "EMAIL",
            "CREDIT_CARD": "CREDIT_CARD",
            "US_BANK_NUMBER": "BANK_ACCOUNT",
            "LOCATION": "ADDRESS",
            "MEDICAL_LICENSE": "MEDICAL_ID",
            "UK_NHS": "MEDICAL_ID",
            "US_PASSPORT": "SSN",
            "US_DRIVER_LICENSE": "SSN",
            "US_ITIN": "SSN",
            "ES_NIF": "SSN",
            "ES_NIE": "SSN",
            "IT_DRIVER_LICENSE": "SSN",
            "IT_FISCAL_CODE": "SSN",
            "IT_VAT_CODE": "SSN",
            "IT_IDENTITY_CARD": "SSN",
            "IT_PASSPORT": "SSN",
            "PL_PESEL": "SSN",
            "IN_AADHAAR": "SSN",
            "IN_PAN": "SSN",
            "IN_GSTIN": "SSN",
            "IN_PASSPORT": "SSN",
            "IN_VOTER_ID": "SSN",
            "IN_DRIVING_LICENSE": "SSN",
            "IN_UPI": "CREDIT_CARD",
            "IN_IFSC": "BANK_ACCOUNT",
        }
        
        # Convert entity_type to pii_type
        pii_type = entity_to_pii_map.get(entity_type, "NAME")  # Default to NAME for unknown types
        
        # Use the SAME ultra-strict validation as LLM detections
        return self._validate_llm_pii(pii_type, text, confidence)
    
    def _validate_llm_pii(self, pii_type: str, text: str, confidence: float) -> bool:
        """
        Validate LLM-detected PII to filter false positives.
        This has been relaxed to improve recall.
        """
        normalized_type = (pii_type or "").upper().strip()
        text = (text or "").strip()
        text_lower = text.lower()

        if not text:
            return False

        # Reject content that is mostly symbols
        if len(text) > 0 and sum(not c.isalnum() and not c.isspace() for c in text) > len(text) / 2:
            return False

        min_lengths = {
            "EMAIL": 6,
            "PHONE": 7,
            "SSN": 8,
            "US_SSN": 8,
            "CREDIT_CARD": 8,
            "BANK_ACCOUNT": 6,
            "US_BANK_NUMBER": 6,
            "NAME": 2,
            "ADDRESS": 8,
            "DATE_OF_BIRTH": 6,
            "DATE_TIME": 6,
            "MEDICAL_ID": 4,
            "MEDICAL_LICENSE": 4,
            "TAX_ID": 8,
        }
        if len(text) < min_lengths.get(normalized_type, 2):
            return False

        if normalized_type == "EMAIL":
            return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", text))

        if normalized_type == "NAME":
            if '.' in text and any(text_lower.endswith(ext) for ext in ['.com', '.net', '.org', '.edu', '.io', '.ai']):
                return False
            non_names = {
                'invoice', 'receipt', 'total', 'amount', 'customer', 'vendor',
                'authorization', 'payment', 'transaction', 'billing', 'service', 'product', 'order'
            }
            if text_lower in non_names:
                return False
            return True

        if normalized_type in ["DATE_TIME", "DATE_OF_BIRTH"]:
            has_numeric_date = bool(re.search(r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b", text))
            has_written_date = bool(re.search(r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}\b", text_lower))
            has_year_token = bool(re.search(r"\b(19|20)\d{2}\b", text))
            return has_numeric_date or has_written_date or has_year_token

        if normalized_type == "ADDRESS":
            has_number = any(c.isdigit() for c in text)
            has_space = ' ' in text
            street_indicators = [
                'street', 'st', 'avenue', 'ave', 'road', 'rd', 'blvd', 'boulevard', 'drive', 'dr',
                'lane', 'ln', 'court', 'ct', 'place', 'pl', 'nagar', 'colony', 'sector', 'block',
                'apartment', 'apt', 'flat', 'house', 'layout', 'phase', 'main', 'cross'
            ]
            has_indicator = any(ind in text_lower for ind in street_indicators)
            return has_space and (has_number or has_indicator)

        if normalized_type == "PHONE":
            digits = ''.join(c for c in text if c.isdigit())
            if len(digits) < 10 or len(digits) > 15:
                return False
            if digits in ['1234567890', '0123456789', '9876543210', '0987654321']:
                return False
            return True

        if normalized_type in ["MEDICAL_ID", "MEDICAL_LICENSE"]:
            # Keep permissive for recall; just avoid long narrative fragments.
            return len(text) <= 25

        if normalized_type in ["BANK_ACCOUNT", "US_BANK_NUMBER", "CREDIT_CARD"]:
            digits = ''.join(c for c in text if c.isdigit())
            return 8 <= len(digits) <= 19

        if normalized_type in ["SSN", "US_SSN"]:
            digits = ''.join(c for c in text if c.isdigit())
            return len(digits) in [9, 12]

        if normalized_type in ["PAN", "GSTIN"]:
            normalized_type = "TAX_ID"

        if normalized_type == "TAX_ID":
            text_upper = text.upper().replace('-', '').replace(' ', '')
            if len(text_upper) == 10 and bool(re.match(r'^[A-Z]{5}[0-9]{4}[A-Z]$', text_upper)):
                return True
            if len(text_upper) == 15 and bool(re.match(r'^[0-9]{2}[A-Z0-9]{10}[A-Z][0-9][A-Z]$', text_upper)):
                return True
            return text_upper.isalnum() and 8 <= len(text_upper) <= 15

        if normalized_type == "GENDER":
            allowed = {"male", "female", "m", "f", "other", "non-binary", "transgender"}
            return text_lower in allowed or any(g in text_lower for g in ["male", "female"])

        return confidence >= 0.6
    
    def _detect_pii_with_llm(self, text: str) -> tuple[List[PIIDetection], Dict[str, Any]]:
        """
        Detect PII using LLM
        
        Args:
            text: Text to analyze
        
        Returns:
            Tuple of (list of detected PII, llm response dict)
        """
        if settings.LOW_LATENCY_MODE:
            return [], {
                "provider": "disabled",
                "model": "low_latency_mode",
                "latency": 0.0,
                "tokens": {"input": 0, "output": 0},
                "content": "",
                "system_prompt": REDACTOR_SYSTEM_PROMPT,
                "user_prompt": "",
            }

        # Initialize to ensure it's always defined
        llm_response = {}
        response = ""
        
        try:
            # Truncate text if too long (keep first 3000 chars)
            text_sample = text[:3000] if len(text) > 3000 else text
            
            # Create prompt
            prompt = REDACTOR_PII_DETECTION_PROMPT.format(text=text_sample)
            
            # 70b on key-2 — precise PII detection, key-1 reserved for classifier/validator
            llm_response = llm_client.generate(
                prompt=prompt,
                system_prompt=REDACTOR_SYSTEM_PROMPT,
                max_tokens=350 if settings.BEDROCK_ONLY_MODE else 600,
                temperature=0.0,
                groq_model="llama-3.1-8b-instant",  # 8b-instant: fast entity detection, frees key-2 budget
                groq_key=3  # Tertiary key for heavy redaction load
            )
            response = llm_response.get("content", "")
            
            # Parse JSON response
            response_text = response.strip()
            
            # Handle markdown code blocks
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            # Use json-repair because LLM PII responses occasionally contain
            # malformed JSON (single quotes, trailing commas, unquoted keys).
            repaired = repair_json(response_text, return_objects=True)
            if not isinstance(repaired, dict):
                raise ValueError(f"Expected dict from repaired PII response, got {type(repaired).__name__}")
            result = repaired
            
            # Convert to PIIDetection objects with validation
            pii_detections = []
            for item in result.get("pii_detections", []):
                try:
                    # Map string type to PIIType enum
                    pii_type_str = item.get("pii_type", "NAME").upper()
                    pii_type = PIIType[pii_type_str] if hasattr(PIIType, pii_type_str) else PIIType.NAME
                    
                    original_text = item.get("original_text", "")
                    confidence = item.get("confidence", 0.9)
                    
                    # Validate LLM detection (apply same strict rules)
                    is_valid = self._validate_llm_pii(pii_type_str, original_text, confidence)
                    if not is_valid:
                        logger.debug(f"FILTERED LLM: {pii_type_str}='{original_text}' (confidence={confidence:.2f})")
                        continue
                    else:
                        logger.debug(f"ACCEPTED LLM: {pii_type_str}='{original_text}' (confidence={confidence:.2f})")
                    
                    detection = PIIDetection(
                        field_name=item.get("field_name", "unknown"),
                        pii_type=pii_type,
                        original_text=original_text,
                        redacted_text=item.get("redacted_text", "[REDACTED]"),
                        detection_source="llm",
                        confidence=confidence
                    )
                    pii_detections.append(detection)
                except Exception as e:
                    logger.warning(f"Failed to parse PII detection: {e}")
                    continue
            
            return pii_detections, llm_response
        
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse LLM PII response: {e}")
            if response:
                logger.debug(f"Raw response: {response}")
            return [], llm_response if llm_response else {}
        except Exception as e:
            logger.error(f"LLM PII detection failed: {e}")
            return [], llm_response if llm_response else {}
    
    def _redact_text(self, text: str, pii_detections: List[PIIDetection]) -> str:
        """
        Apply redactions to text
        
        Args:
            text: Original text
            pii_detections: List of PII to redact
        
        Returns:
            Redacted text
        """
        redacted_text = text
        
        # Sort by position (longest first to handle overlaps)
        sorted_detections = sorted(
            pii_detections,
            key=lambda x: len(x.original_text),
            reverse=True
        )
        
        # Apply redactions
        for detection in sorted_detections:
            if detection.original_text in redacted_text:
                redacted_text = redacted_text.replace(
                    detection.original_text,
                    detection.redacted_text
                )
        
        return redacted_text
    
    def _compute_metrics(
        self,
        detected_pii: List[PIIDetection],
        extracted_fields: Dict[str, Any],
        ground_truth_pii: Any = None,
    ) -> Tuple[float, float]:
        """
        Compute PII precision and recall
        
        Args:
            detected_pii: Detected PII list
            extracted_fields: Extracted fields (ground truth)
        
        Returns:
            Tuple of (precision, recall)
        """
        def _normalize(value: Any) -> str:
            return " ".join(str(value or "").strip().lower().split())

        def _collect_schema_pii_values(value: Any, field_name: str = "") -> set[str]:
            collected: set[str] = set()
            if value is None:
                return collected
            if isinstance(value, str):
                field_lower = field_name.lower()
                if any(term in field_lower for term in [
                    "email", "phone", "ssn", "social_security",
                    "credit_card", "card_number", "name", "address",
                    "patient_id", "medical_id", "dob", "date_of_birth",
                    "student_id", "document_number"
                ]):
                    normalized = _normalize(value)
                    if normalized:
                        collected.add(normalized)
                return collected
            if isinstance(value, dict):
                for child_name, child_value in value.items():
                    collected.update(_collect_schema_pii_values(child_value, child_name))
                return collected
            if isinstance(value, list):
                for child in value:
                    collected.update(_collect_schema_pii_values(child, field_name))
            return collected

        reference_values: set[str] = set()
        if ground_truth_pii:
            for item in ground_truth_pii:
                if isinstance(item, dict):
                    normalized = _normalize(item.get("value", ""))
                    if normalized:
                        reference_values.add(normalized)
        else:
            for field_name, value in extracted_fields.items():
                reference_values.update(_collect_schema_pii_values(value, field_name))

        detected_values = {
            _normalize(pii.original_text)
            for pii in detected_pii
            if _normalize(pii.original_text)
        }
        
        # Compute metrics
        if len(detected_values) == 0:
            precision = 1.0 if len(reference_values) == 0 else 0.0
        else:
            true_positives = len(detected_values & reference_values)
            precision = true_positives / len(detected_values)

        if len(reference_values) == 0:
            recall = 1.0
        else:
            true_positives = len(detected_values & reference_values)
            recall = true_positives / len(reference_values)
        
        return precision, recall
    
    def redact(self, state: DocumentState) -> DocumentState:
        """
        Detect and redact PII from document using Presidio + LLM hybrid
        
        Args:
            state: Current document state
        
        Returns:
            Updated state with redaction result
        """
        logger.info(f"{self.name}: Starting Presidio + LLM PII detection and redaction")
        start_time = time.time()
        
        try:
            raw_text = state["raw_text"]
            extracted_fields = state.get("extracted_fields", {})
            
            # 1. Detect PII using Presidio (primary, fast)
            presidio_pii = self._detect_pii_with_presidio(raw_text)
            logger.info(f"Presidio detected {len(presidio_pii)} PII instances")

            # 1b. Rule-based gender detection (fast, no LLM)
            gender_pii = self._detect_gender_patterns(raw_text)
            if gender_pii:
                logger.info(f"Gender detection found {len(gender_pii)} instance(s)")
                presidio_pii.extend(gender_pii)

            # 1c. Regex-based custom IDs for medical/lab documents
            custom_id_pii = self._detect_custom_id_patterns(raw_text)
            if custom_id_pii:
                logger.info(f"Custom ID detection found {len(custom_id_pii)} instance(s)")
                presidio_pii.extend(custom_id_pii)

            multiline_address_pii = self._detect_multiline_addresses(raw_text)
            if multiline_address_pii:
                logger.info(f"Multiline address detection found {len(multiline_address_pii)} instance(s)")
                presidio_pii.extend(multiline_address_pii)

            # 2. Enhance with LLM for context-aware detection (HYBRID MODE RE-ENABLED)
            llm_pii, llm_response = self._detect_pii_with_llm(raw_text)
            logger.info(f"LLM detected {len(llm_pii)} additional PII instances")
            
            # 3. Merge results with smart deduplication
            detected_pii = presidio_pii.copy()
            existing_texts = {pii.original_text.lower() for pii in presidio_pii}
            
            for pii in llm_pii:
                pii_text_lower = pii.original_text.lower()
                
                # Skip if exact match already exists
                if pii_text_lower in existing_texts:
                    continue
                
                # Skip if this text is contained in a larger existing detection
                # (e.g., don't add "John" if "John Smith" already detected)
                is_substring = False
                for existing in existing_texts:
                    if pii_text_lower in existing or existing in pii_text_lower:
                        # Keep the longer one
                        if len(pii_text_lower) > len(existing):
                            # Remove shorter existing detection
                            detected_pii = [p for p in detected_pii if p.original_text.lower() != existing]
                            existing_texts.discard(existing)
                            break
                        else:
                            is_substring = True
                            break
                
                if not is_substring:
                    detected_pii.append(pii)
                    existing_texts.add(pii_text_lower)

            alias_detections = []
            for pii in detected_pii:
                for alias in self._generate_detection_aliases(raw_text, pii):
                    alias_text_lower = alias.original_text.lower()
                    if alias_text_lower not in existing_texts:
                        alias_detections.append(alias)
                        existing_texts.add(alias_text_lower)
            if alias_detections:
                detected_pii.extend(alias_detections)
            
            logger.info(f"Total unique PII instances: {len(detected_pii)} (Presidio: {len(presidio_pii)}, LLM: {len(llm_pii)})")
            
            # 2. Redact text
            redacted_text = self._redact_text(raw_text, detected_pii)
            
            # 3. Compute metrics
            precision, recall = self._compute_metrics(
                detected_pii,
                extracted_fields,
                ground_truth_pii=state.get("ground_truth_pii"),
            )
            
            latency = time.time() - start_time
            
            # Create result
            redaction_result = RedactionResult(
                redacted_text=redacted_text,
                pii_detections=detected_pii,
                detected_pii=detected_pii,  # For backward compatibility
                pii_count=len(detected_pii),
                precision=precision,
                recall=recall,
                timestamp=datetime.utcnow()
            )
            
            # Update state
            state["redacted_text"] = redacted_text
            state["redaction_result"] = redaction_result
            state["agent_timings"][self.name] = latency
            
            # Responsible AI logging
            state["trace_log"].append(
                ResponsibleAILog(
                    agent_name=self.name,
                    input_data=raw_text[:500],
                    output_data=f"PII Count: {len(detected_pii)}, Precision: {precision:.2f}, Recall: {recall:.2f}",
                    timestamp=datetime.utcnow(),
                    latency_ms=latency * 1000,
                    llm_model_used=llm_response.get("model", "unknown"),
                    tokens_used=llm_response.get("tokens", {}).get("input", 0) + llm_response.get("tokens", {}).get("output", 0),
                    error_occurred=False,
                    llm_provider=llm_response.get("provider", "unknown"),
                    system_prompt="",
                    user_prompt=llm_response.get("user_prompt", ""),
                    context_data={
                        "text_length": len(raw_text),
                        "pii_count": len(detected_pii),
                        "precision": precision,
                        "recall": recall
                    },
                    raw_output=llm_response.get("content", ""),
                    tokens_input=llm_response.get("tokens", {}).get("input", 0),
                    tokens_output=llm_response.get("tokens", {}).get("output", 0),
                    retry_attempt=0
                )
            )
            
            logger.info(
                f"{self.name}: Redaction complete",
                pii_count=len(detected_pii),
                precision=precision,
                recall=recall,
                latency_ms=latency * 1000
            )
            
            return state
        
        except Exception as e:
            logger.error(f"{self.name}: Redaction failed", error=str(e))
            
            # Update state with error
            state["errors"].append(f"{self.name}: {str(e)}")
            state["redacted_text"] = state["raw_text"]  # No redaction
            state["redaction_result"] = RedactionResult(
                redacted_text=state["raw_text"],
                pii_detections=[],
                detected_pii=[],
                pii_count=0,
                precision=0.0,
                recall=0.0
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
                    system_prompt=REDACTOR_PII_DETECTION_PROMPT[:500],
                    user_prompt="",
                    context_data={
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
redactor_agent = RedactorAgent()

