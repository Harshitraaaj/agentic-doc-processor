"""
Central repository for all LLM prompts used by agents.
Organized by agent type for easy maintenance and version control.
"""

# ============================================================================
# CLASSIFIER AGENT PROMPTS
# ============================================================================

CLASSIFIER_PROMPT = """Classify this document into ONE category.

DECISION (first match wins):
1. Payment/billing/invoice present? → financial_document
2. Has Work Experience OR Skills OR career objective? → resume
3. Formal offer letter from employer to candidate? → job_offer
4. Medical diagnosis/prescription/lab (no billing)? → medical_record
5. Physical ID card (passport, driver license, Aadhaar, PAN, voter ID, college ID card, student ID card, employee ID)? → id_document
6. Academic/educational content (transcript/mark sheet/diploma/degree certificate/textbook/research paper/journal article/syllabus)? → academic

KEY RULES:
- Work Experience/Skills present → ALWAYS resume (even if Education section exists)
- Written BY a person about themselves → resume
- College ID card / Student ID card → id_document (NOT academic)
- Issued BY an institution showing grades/scores OR educational textbook/research content → academic
- Medical receipt with charges → financial_document

DOCUMENT:
{text_excerpt}

Return JSON only:
{{"doc_type": "<category>", "confidence": <0.8-1.0>, "reasoning": "<key signal>"}}"""

CLASSIFIER_SYSTEM_PROMPT = "Document classifier. CRITICAL: Work Experience/Skills = resume. College/student ID card = id_document (NOT academic). Academic includes transcripts, textbooks, syllabus, and research papers. Medical receipt with charges = financial_document. Return JSON only."


# ============================================================================
# EXTRACTOR AGENT PROMPTS
# ============================================================================

EXTRACTOR_PROMPT = """You are extracting structured data from a {doc_type} document.

STEP 1 — Study this correctly extracted example for a {doc_type}:
{few_shot_examples}

STEP 2 — Extract at least 15 relevant fields from the document below that best identify and characterise it as a {doc_type}. Choose the most meaningful fields for this document type.

STEP 3 — Rules:
- Use snake_case field names (e.g. candidate_name, date_of_birth, company_name)
- Match field names from the example above as closely as possible
- dates → YYYY-MM-DD format
- money → float without $ or commas
- names → Title Case
- list fields (work_experience, education, skills, certifications, medications, courses, etc.) → ALWAYS return as array, use [] if none found
- work_experience: each entry MUST have: job_title, employer, start_date, end_date, responsibilities
- education: each entry MUST have: degree, institution, graduation_date, gpa
- Use null for missing scalar fields, [] for missing list fields

DOCUMENT:
{document_text}

Return ONLY valid JSON with at least 15 fields. No markdown, no explanation."""

EXTRACTOR_SYSTEM_PROMPT = "Document extraction engine. Return JSON only with exact schema field names. Dates=YYYY-MM-DD, money=float, missing=null."


# ============================================================================
# VALIDATOR AGENT PROMPTS
# ============================================================================

VALIDATOR_PROMPT = """Validate extracted {doc_type} fields.

SCHEMA FIELDS: {schema_fields}
REQUIRED: {priority_fields}
EXTRACTED: {extracted_fields}

RULES:
- If required fields contain any real value → is_valid = true
- NEVER flag type mismatches: lists, arrays, objects, strings are equally valid
- Field name aliases are fine: "experience" = work_experience, "employment_history" = work_experience
- Accept any date, number, or currency format
- If 40%+ schema fields have real values → is_valid = true
- Only return errors for fields that are COMPLETELY absent (null / empty string / missing key)
- If you have doubts, default to is_valid = true

Return ONLY this JSON (no extra text):
{{
  "is_valid": true,
  "errors": [],
  "warnings": [],
  "status": "valid"
}}"""

VALIDATOR_SYSTEM_PROMPT = "Document validator. Return ONLY valid JSON with is_valid, errors, warnings, status. No code, no explanations."


# ============================================================================
# SELF-REPAIR NODE PROMPTS
# ============================================================================

SELF_REPAIR_PROMPT = """Fix validation errors in extracted data fields.

**Extracted Fields:**
{extracted_fields}

**Errors to Fix:**
{errors}

**Reference Text:**
{text_excerpt}

**Task:** Correct the errors and return valid JSON with fixed fields.

**Output Format:** Return only the corrected JSON object."""

SELF_REPAIR_RE_EXTRACTION_PROMPT = """Re-extract fields from this {doc_type} document. Current accuracy: {current_accuracy}% — target 90-100%.

SCHEMA FIELDS (extract all):
{schema_fields}

PREVIOUS EXTRACTION (keep non-null values, fill in the rest):
{extracted_fields}

MISSING FIELDS (focus here):
{missing_fields_list}

ISSUES:
{validation_errors}

DOCUMENT:
{document_text}

Return ONLY JSON with all schema fields (keep existing + fill missing). Missing=null. No markdown."""

SELF_REPAIR_SYSTEM_PROMPT = "Extract and repair data fields. Return valid JSON only."


# ============================================================================
# REDACTOR AGENT PROMPTS
# ============================================================================

REDACTOR_PII_DETECTION_PROMPT = """Identify ALL genuine personal identifiable information (PII) in the following text. Be THOROUGH but PRECISE.

**⚠️ CRITICAL: DO NOT detect domains/URLs ending with .com, .net, .org, .edu, .gov, .io, .ai, etc. These are NOT names!**
**⚠️ DO NOT detect text containing dots (.) unless it's a valid PII format (email with @, or proper date with separators)**
**⚠️ DO NOT detect concatenated dates like "march152024" or "april142024" - these are NOT real dates!**

**Text:**
{text}

**PII Types to Detect - Be THOROUGH but PRECISE:**

1. EMAIL: Complete email addresses
   - MUST have @ and domain
   - DO NOT detect domain parts alone ("acme.com") or username fragments ("john.doe")

2. PHONE: Phone numbers
   - MUST have 10+ digits
   - With formatting: "+1-555-1234", "(415) 555-1234"
   - Without formatting: "4155551234", "9876543210"
   - Various formatting: spaces, dashes, parentheses

3. SSN: National identification numbers
   - US SSN: 9 digits ("123-45-6789" or "123456789")
   - India Aadhaar: 12 digits ("1234 5678 9012")
   - Other formats: 9-12 digits with or without formatting
   - Detect partially redacted: "XXX-XX-1234", "XXXX XXXX 1234"

4. CREDIT_CARD: Credit/debit card numbers
   - 13-19 digits with optional spaces/dashes
   - Look for "Card ending in", "**** **** **** 1234"

5. BANK_ACCOUNT: Bank account numbers
   - 8-20 digits
   - DO NOT detect: Routing numbers, IFSC codes, SWIFT codes (bank identifiers)

6. TAX_ID: Tax identification numbers
   - India PAN: AAAAA9999A ("ABCDE1234F")
   - India GSTIN: 15 alphanumeric ("22AAAAA0000A1Z5")
   - Other tax IDs: Various formats

7. NAME: Full person names
   - MUST detect BOTH multi-word ("John Smith") AND single-word concatenated names ("davidpark", "robertthompson", "lisarodriguez")
   - Look for names even if concatenated without spaces: "sarahwilliams", "michaelchen"
   - Include titles if attached: "dr.sarahwilliams", "mr.johnsmith"
   - DO NOT detect: Company names, domains (.com, .net, .org), software/tech terms (github, linkedin), job titles

8. ADDRESS: COMPLETE street addresses
   - MUST include: Street/House number + Street name + City
   - Examples: "123 Main St, San Jose, CA 95110" or "Plot 45, MG Road, Bangalore 560001"
   - DO NOT detect: City names alone, postal codes alone

9. DATE_OF_BIRTH: Actual birthdates of people
   - MUST be in context of a person (near "DOB", "Birth", etc.)
   - DO NOT detect: Invoice dates, billing periods, random dates

10. MEDICAL_ID: Medical record numbers, health insurance numbers, patient IDs

**CRITICAL EXCLUSIONS (DO NOT detect):**
❌ Durations: "30 days", "6 years", "60 hrs", "monthly", "annually"
❌ City names alone (without full address)
❌ Month/year: "May 2019", "March 2024"
❌ Concatenated dates WITHOUT separators: "march152024", "april142024", "february282024"
❌ Domains/URLs: "acme.com", "company.net", "github.com", "linkedin.com", "gmail.com", "stanford.edu"
❌ Domain fragments: "acmecorp.com", "cloudtech.com", "pge.com", "mariosnyc.com"
❌ Email fragments: "john.doe" (without @)
❌ Tech terms: "Docker", "Python", "Kubernetes", "synaptic", "deeplearning"
❌ Latin phrases: "summa cum laude", "magna cum laude"
❌ Job titles: "Senior Engineer", "Manager"
❌ Company names
❌ Postal codes alone
❌ Bank identifiers: routing numbers, SWIFT codes, sort codes
❌ Invoice/document numbers: "INV-2024-001", "PO-12345"
❌ Alphanumeric codes: "wa1234567", "xx1234567", "r01", "i10"

**Response Format:**
{{
  "pii_detections": [
    {{
      "field_name": "email_address",
      "pii_type": "EMAIL",
      "original_text": "john@example.com",
      "redacted_text": "[EMAIL_REDACTED]",
      "confidence": 0.95
    }}
  ]
}}

CRITICAL: Return ONLY the JSON object. Start with {{ and end with }}. No explanations.
Be THOROUGH: Catch all real PII, even if formatting is imperfect. Better to over-detect (we'll filter) than miss actual PII."""

REDACTOR_SYSTEM_PROMPT = "You are a precise PII detection system. Detect ALL genuine personal information that could identify real individuals. Be thorough but precise - avoid detecting domains (.com, .org), URLs, company names, or non-PII text. Return ONLY valid JSON with no explanations."
