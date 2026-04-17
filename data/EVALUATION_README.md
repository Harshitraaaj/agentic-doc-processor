# Evaluation Dataset Documentation

## Overview

This dataset contains **10 carefully curated documents** (5 financial + 5 resumes) with complete ground truth annotations for evaluating the document processing pipeline.

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Total Documents | 10 |
| Financial Documents | 5 (invoices, receipts, bills) |
| Resumes | 5 (tech, academic, security roles) |
| Avg Document Length | ~2000 characters |
| **Total PII Instances** | **89** (across all docs) |
| Total Key Fields | ~100 |

## Why Only 2 Document Types?

✅ **Financial Documents** - Most common business use case (invoices, receipts, bills)  
✅ **Resumes** - Critical for HR/recruitment, high PII content

These 2 types let us validate core capabilities while minimizing LLM credit usage.

## Dataset Structure

### Columns

1. **document_id** - Unique identifier (FIN001-FIN005, RES001-RES005)
2. **document_text** - ~2000 char document text with realistic data
3. **ground_truth_class** - Expected classification (`financial_document`, `resume`)
4. **ground_truth_fields** - JSON object with expected extracted fields
5. **ground_truth_pii** - JSON array with PII annotations (type, value, position)
6. **notes** - Description of document variant

### PII Coverage

Each document contains multiple PII types:
- **NAME** - Person and company names
- **EMAIL** - Email addresses
- **PHONE** - Phone numbers (various formats)
- **ADDRESS** - Physical addresses
- **SSN** - Social Security Numbers
- **DATE_OF_BIRTH** - Birth dates
- **CREDIT_CARD** - Account/transaction numbers
- **MEDICAL_ID** - Medical record numbers, insurance IDs

## Ground Truth PII Format

```json
[
  {
    "type": "EMAIL",
    "value": "john.doe@example.com",
    "position": [123, 145]
  },
  {
    "type": "SSN",
    "value": "123-45-6789",
    "position": [450, 461]
  }
]
```

## Document Variants

### Financial Documents
- **FIN001** - Standard B2B invoice with tax calculation
- **FIN002** - Restaurant receipt with tip and credit card
- **FIN003** - Utility bill with account numbers
- **FIN004** - International invoice (zero-rated export)
- **FIN005** - Medical billing statement (complex)
- **FIN006** - Rental receipt with lease details

### Resumes
- **RES001** - Senior data scientist (tech industry)
- **RES002** - Academic PhD researcher (publications)
- **RES003** - Junior frontend developer
- **RES004** - Cybersecurity engineer (clearance)

## Expected Metrics (Baseline)

Based on current system:

| Metric | Expected | Target | Status |
|--------|----------|--------|--------|
| Classification Accuracy | ~85-95% | - | Baseline |
| Extraction Accuracy | **~75-85%** | **≥90%** | ⚠️ Needs improvement |
| PII Recall | **~85-90%** | **≥95%** | ⚠️ Needs improvement |
| PII Precision | **~80-85%** | **≥90%** | ⚠️ Needs improvement |
| Success Rate | ~90-95% | ≥90% | ✅ Likely passing |
| p95 Latency | ~2000ms | ≤4000ms | ✅ Passing |

## Running Evaluation

```bash
# Activate environment
.venv\Scripts\activate

# Run evaluation
python evaluate_metrics.py
```

This will:
1. Process all 10 documents through the pipeline
2. Compare results against ground truth
3. Calculate all target metrics
4. Save detailed results to `evaluation_results.json`
5. Print pass/fail for each target

## How PII Recall/Precision Works

### Per-Document Measurement
Each document is evaluated individually:
- Count how many PII were correctly detected (True Positives)
- Count missed PII (False Negatives)  
- Count false detections (False Positives)

### Aggregate Measurement
Then we sum across ALL documents:

**Recall = Total TP / (Total TP + Total FN)**  
*"What % of all PII did we find?"*

**Precision = Total TP / (Total TP + Total FP)**  
*"What % of our detections were correct?"*

Example:
```
Doc 1: Found 8/10 PII, 1 false positive
Doc 2: Found 9/9 PII, 0 false positives
Doc 3: Found 7/10 PII, 2 false positives

Aggregate:
- Total TP: 8 + 9 + 7 = 24
- Total FN: 2 + 0 + 3 = 5
- Total FP: 1 + 0 + 2 = 3

Recall = 24 / (24 + 5) = 82.8%
Precision = 24 / (24 + 3) = 88.9%
```

This gives us a **single recall/precision score** for the entire model across all document types.

## Extending the Dataset

To add more documents:

1. **Add row to CSV** with same column structure
2. **Use format**:
   - `document_id`: `{TYPE}{XXX}` (e.g., CON001 for contracts)
   - `document_text`: ~2000 chars realistic text
   - `ground_truth_class`: One of DocumentType enum values
   - `ground_truth_fields`: JSON with expected extractions
   - `ground_truth_pii`: JSON array with PII annotations

3. **PII annotation tips**:
   - Position = [start_index, end_index] in document_text
   - Be thorough - mark ALL PII instances
   - Include variations (formatted vs unformatted phone numbers)

## Common Issues & Solutions

### Issue: Low Extraction Accuracy
**Solutions:**
- Add few-shot examples to extraction prompts
- Strengthen field validation rules
- Lower LLM temperature (0.1-0.3)

### Issue: Low PII Recall (Missing PII)
**Solutions:**
- Add custom Presidio recognizers
- Lower confidence threshold
- Use context-aware detection

### Issue: Low PII Precision (False Positives)
**Solutions:**
- Raise confidence threshold
- Add PII allowlist (company names, generic terms)
- Post-detection validation

### Issue: Documents Failing
**Solutions:**
- Check document format/encoding
- Enhance error recovery in agents
- Add preprocessing validation

## Credit Usage Estimate

**Per document (estimated):**
- Classification: ~200 tokens (~$0.0001)
- Extraction: ~1500 tokens (~$0.0008)
- Validation: ~800 tokens (~$0.0004)
- Redaction: ~600 tokens (~$0.0003)

**Total per doc:** ~$0.0016  
**10 documents:** ~$0.016 (negligible)

**Running evaluation multiple times is affordable!**

## Next Steps

1. **Baseline:** Run evaluation to get current metrics
2. **Identify gaps:** See which metrics are below targets
3. **Implement fixes:** Apply improvements (few-shot, validation rules, etc.)
4. **Re-evaluate:** Run again to measure improvement
5. **Iterate:** Repeat until all targets met

## Questions?

See [evaluate_metrics.py](../evaluate_metrics.py) for the full evaluation implementation.
