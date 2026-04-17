"""
Knowledge look-up service for validation support.

Provides:
- Storage: SQLite (persistent knowledge entries)
- Vector search: FAISS semantic retrieval
- Schema payload: Pydantic-generated JSON Schema + optional file-backed custom schemas
"""
import json
import sqlite3
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from schemas.document_schemas import (
    DocumentType,
    FinancialDocumentFields,
    ResumeFields,
    JobOfferFields,
    MedicalRecordFields,
    IdDocumentFields,
    AcademicFields,
    GeneralDocumentFields,
)
from utils.config import settings
from utils.faiss_manager import FAISSIndex
from utils.logger import logger


DEFAULT_REQUIRED_FIELDS = {
    "financial_document": ["total_amount", "issuer_name", "document_date", "document_number", "recipient_name"],
    "resume": ["candidate_name", "email", "phone", "work_experience", "education"],
    "job_offer": ["candidate_name", "position_title", "salary", "start_date", "company_name"],
    "medical_record": ["patient_name", "date_of_birth", "diagnosis", "physician_name", "visit_date"],
    "id_document": ["full_name", "document_number", "date_of_birth", "issue_date", "expiration_date"],
    "academic": ["student_name", "institution_name", "degree_program", "graduation_date", "gpa"],
    "unknown": ["title", "date", "author"],
}


def _model_json_schema(model_cls) -> Dict[str, Any]:
    if hasattr(model_cls, "model_json_schema"):
        return model_cls.model_json_schema()
    return model_cls.schema()


class KnowledgeLookup:
    """Knowledge registry with SQLite persistence + FAISS retrieval."""

    def __init__(self):
        self.knowledge_dir = settings.DATA_DIR / "knowledge"
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.knowledge_dir / "knowledge_lookup.db"
        self.custom_schema_file = self.knowledge_dir / "custom_schemas.json"

        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._init_db()

        self.index = FAISSIndex(index_name="knowledge_lookup")
        self._bootstrap_knowledge()
        self._rebuild_index()

        logger.info("KnowledgeLookup initialized")

    def _init_db(self) -> None:
        cursor = self.conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS knowledge_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_type TEXT NOT NULL UNIQUE,
                source TEXT NOT NULL,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.conn.commit()

    def _upsert_entry(
        self,
        doc_type: str,
        source: str,
        title: str,
        content: str,
        metadata: Dict[str, Any],
    ) -> None:
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO knowledge_entries (doc_type, source, title, content, metadata_json)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(doc_type) DO UPDATE SET
                source=excluded.source,
                title=excluded.title,
                content=excluded.content,
                metadata_json=excluded.metadata_json,
                updated_at=CURRENT_TIMESTAMP
            """,
            (doc_type, source, title, content, json.dumps(metadata, ensure_ascii=False)),
        )
        self.conn.commit()

    def _bootstrap_knowledge(self) -> None:
        """Seed builtin Pydantic schemas and optional custom file schemas."""
        builtin_map = {
            DocumentType.FINANCIAL_DOCUMENT.value: FinancialDocumentFields,
            DocumentType.RESUME.value: ResumeFields,
            DocumentType.JOB_OFFER.value: JobOfferFields,
            DocumentType.MEDICAL_RECORD.value: MedicalRecordFields,
            DocumentType.ID_DOCUMENT.value: IdDocumentFields,
            DocumentType.ACADEMIC.value: AcademicFields,
            DocumentType.UNKNOWN.value: GeneralDocumentFields,
        }

        for doc_type, model_cls in builtin_map.items():
            schema = _model_json_schema(model_cls)
            schema_fields = list(schema.get("properties", {}).keys())
            required_fields = DEFAULT_REQUIRED_FIELDS.get(doc_type, schema.get("required", []))
            content = (
                f"Document type: {doc_type}\n"
                f"Schema fields: {', '.join(schema_fields)}\n"
                f"Priority/required fields: {', '.join(required_fields)}"
            )
            metadata = {
                "doc_type": doc_type,
                "schema_fields": schema_fields,
                "required_fields": required_fields,
                "json_schema": schema,
            }
            self._upsert_entry(
                doc_type=doc_type,
                source="pydantic",
                title=f"{doc_type} schema profile",
                content=content,
                metadata=metadata,
            )

        # Optional file-backed custom schemas
        if not self.custom_schema_file.exists():
            self.custom_schema_file.write_text("[]", encoding="utf-8")

        try:
            custom_entries = json.loads(self.custom_schema_file.read_text(encoding="utf-8"))
            if isinstance(custom_entries, list):
                for entry in custom_entries:
                    doc_type = str(entry.get("doc_type", "")).strip().lower()
                    if not doc_type:
                        continue
                    json_schema = entry.get("json_schema", {}) if isinstance(entry.get("json_schema", {}), dict) else {}
                    schema_fields = list(json_schema.get("properties", {}).keys())
                    required_fields = entry.get("required_fields", [])
                    if not isinstance(required_fields, list):
                        required_fields = []
                    notes = str(entry.get("notes", "")).strip()

                    content = (
                        f"Custom document type: {doc_type}\n"
                        f"Schema fields: {', '.join(schema_fields)}\n"
                        f"Required fields: {', '.join(required_fields)}\n"
                        f"Notes: {notes}"
                    )
                    metadata = {
                        "doc_type": doc_type,
                        "schema_fields": schema_fields,
                        "required_fields": required_fields,
                        "json_schema": json_schema,
                        "notes": notes,
                    }
                    self._upsert_entry(
                        doc_type=doc_type,
                        source="file_json_schema",
                        title=f"{doc_type} custom schema profile",
                        content=content,
                        metadata=metadata,
                    )
        except Exception as e:
            logger.warning(f"KnowledgeLookup: Failed reading custom schemas: {e}")

    def _read_custom_schema_entries(self) -> List[Dict[str, Any]]:
        if not self.custom_schema_file.exists():
            return []
        try:
            data = json.loads(self.custom_schema_file.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except Exception as e:
            logger.warning(f"KnowledgeLookup: Failed to read custom schema file: {e}")
            return []

    def _write_custom_schema_entries(self, entries: List[Dict[str, Any]]) -> None:
        self.custom_schema_file.write_text(
            json.dumps(entries, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _optionalize_property_schema(self, schema_fragment: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(schema_fragment, dict):
            return {"type": ["string", "null"]}

        fragment = dict(schema_fragment)
        prop_type = fragment.get("type")
        if isinstance(prop_type, str):
            if prop_type != "null":
                fragment["type"] = [prop_type, "null"]
        elif isinstance(prop_type, list):
            normalized = []
            for item in prop_type:
                if isinstance(item, str) and item not in normalized:
                    normalized.append(item)
            if "null" not in normalized:
                normalized.append("null")
            fragment["type"] = normalized
        else:
            fragment["type"] = ["string", "null"]

        return fragment

    def _merge_custom_json_schema(
        self,
        existing_schema: Dict[str, Any],
        incoming_schema: Dict[str, Any],
        explicit_required_fields: List[str],
    ) -> Dict[str, Any]:
        base_schema = existing_schema if isinstance(existing_schema, dict) else {}
        next_schema = incoming_schema if isinstance(incoming_schema, dict) else {}

        existing_properties = base_schema.get("properties", {})
        incoming_properties = next_schema.get("properties", {})
        if not isinstance(existing_properties, dict):
            existing_properties = {}
        if not isinstance(incoming_properties, dict):
            incoming_properties = {}

        merged_properties = dict(existing_properties)
        for field_name, field_schema in incoming_properties.items():
            if not isinstance(field_name, str) or not field_name.strip():
                continue
            clean_name = field_name.strip()
            if clean_name in merged_properties:
                merged_properties[clean_name] = field_schema if isinstance(field_schema, dict) else merged_properties[clean_name]
                continue

            if clean_name in explicit_required_fields:
                merged_properties[clean_name] = field_schema if isinstance(field_schema, dict) else {"type": "string"}
            else:
                merged_properties[clean_name] = self._optionalize_property_schema(
                    field_schema if isinstance(field_schema, dict) else {"type": "string"}
                )

        for required_field in explicit_required_fields:
            if required_field not in merged_properties:
                merged_properties[required_field] = {"type": "string"}

        existing_required = base_schema.get("required", [])
        if not isinstance(existing_required, list):
            existing_required = []

        merged_required = []
        for field_name in list(existing_required) + list(explicit_required_fields):
            if (
                isinstance(field_name, str)
                and field_name in merged_properties
                and field_name not in merged_required
            ):
                merged_required.append(field_name)

        merged_schema = {
            "type": "object",
            "properties": merged_properties,
            "required": merged_required,
        }

        for key in ("title", "description", "additionalProperties"):
            if key in next_schema:
                merged_schema[key] = next_schema[key]
            elif key in base_schema:
                merged_schema[key] = base_schema[key]

        return merged_schema

    def _infer_json_schema_for_value(self, value: Any) -> Dict[str, Any]:
        if value is None:
            return {"type": ["string", "null"]}
        if isinstance(value, bool):
            return {"type": ["boolean", "null"]}
        if isinstance(value, int) and not isinstance(value, bool):
            return {"type": ["integer", "null"]}
        if isinstance(value, float):
            return {"type": ["number", "null"]}
        if isinstance(value, dict):
            return {"type": ["object", "null"]}
        if isinstance(value, list):
            return {"type": ["array", "null"]}
        return {"type": ["string", "null"]}

    def register_runtime_observed_fields(
        self,
        doc_type: str,
        observed_fields: Dict[str, Any],
        custom_doc_type: Optional[str] = None,
        notes: str = "",
    ) -> Dict[str, Any]:
        """
        Merge observed fields into SQLite-backed profile only.

        This method intentionally does NOT write to custom_schemas.json.
        """
        target = str(custom_doc_type or doc_type or "unknown").strip().lower().replace(" ", "_")
        if not target:
            target = "unknown"

        cleaned_fields: Dict[str, Any] = {}
        if isinstance(observed_fields, dict):
            for key, value in observed_fields.items():
                field_name = str(key).strip()
                if field_name:
                    cleaned_fields[field_name] = value

        if not cleaned_fields:
            return {
                "doc_type": target,
                "required_fields": [],
                "json_schema": {},
                "notes": "",
                "source": "runtime_observed",
            }

        row = self._fetch_entry(target)
        existing_source = "runtime_observed"
        existing_content = ""
        existing_metadata: Dict[str, Any] = {}

        if row is None and target != "unknown":
            row = self._fetch_entry("unknown")

        if row is not None:
            _, _, existing_source, existing_content, existing_metadata = row

        existing_schema = existing_metadata.get("json_schema", {}) if isinstance(existing_metadata, dict) else {}
        if not isinstance(existing_schema, dict):
            existing_schema = {}

        existing_required = existing_metadata.get("required_fields", []) if isinstance(existing_metadata, dict) else []
        if not isinstance(existing_required, list):
            existing_required = []
        existing_required = [str(field).strip() for field in existing_required if str(field).strip()]

        incoming_schema = {
            "type": "object",
            "properties": {
                field_name: self._infer_json_schema_for_value(field_value)
                for field_name, field_value in cleaned_fields.items()
            },
            "required": [],
        }

        merged_schema = self._merge_custom_json_schema(
            existing_schema=existing_schema,
            incoming_schema=incoming_schema,
            explicit_required_fields=existing_required,
        )

        merged_required = merged_schema.get("required", []) if isinstance(merged_schema.get("required", []), list) else []
        merged_fields = list(merged_schema.get("properties", {}).keys()) if isinstance(merged_schema.get("properties", {}), dict) else []

        merged_notes = str(notes or "").strip() or str(existing_metadata.get("notes", "") if isinstance(existing_metadata, dict) else "").strip()
        content = (
            f"Runtime profile for: {target}\n"
            f"Schema fields: {', '.join(merged_fields)}\n"
            f"Required fields: {', '.join(merged_required)}\n"
            f"Observed in latest run: {', '.join(cleaned_fields.keys())}"
        )

        metadata = {
            "doc_type": target,
            "schema_fields": merged_fields,
            "required_fields": merged_required,
            "json_schema": merged_schema,
            "notes": merged_notes,
            "runtime_observed_fields": list(cleaned_fields.keys()),
        }

        if existing_source.startswith("fallback_"):
            existing_source = existing_source.replace("fallback_", "", 1)
        if existing_source == "none":
            existing_source = "runtime_observed"

        source_label = f"runtime_augmented_{existing_source}"
        self._upsert_entry(
            doc_type=target,
            source=source_label,
            title=f"{target} runtime schema profile",
            content=content if content.strip() else existing_content,
            metadata=metadata,
        )
        self._rebuild_index()

        return {
            "doc_type": target,
            "required_fields": merged_required,
            "json_schema": merged_schema,
            "notes": merged_notes,
            "source": source_label,
        }

    def refresh(self) -> None:
        """Re-read schema sources and rebuild the vector index."""
        self._bootstrap_knowledge()
        self._rebuild_index()
        logger.info("KnowledgeLookup refreshed")

    def register_custom_schema(
        self,
        doc_type: str,
        required_fields: List[str],
        json_schema: Dict[str, Any],
        notes: str = "",
    ) -> Dict[str, Any]:
        """Add or update a custom schema, merging new fields as optional by default."""
        normalized_doc_type = str(doc_type).strip().lower().replace(" ", "_")
        cleaned_required = [str(field).strip() for field in required_fields if str(field).strip()]

        entries = self._read_custom_schema_entries()
        existing_entry = None
        replaced = False
        for index, entry in enumerate(entries):
            if str(entry.get("doc_type", "")).strip().lower() == normalized_doc_type:
                existing_entry = entry
                replaced = True
                break

        existing_schema = {}
        existing_required = []
        existing_notes = ""
        if isinstance(existing_entry, dict):
            maybe_schema = existing_entry.get("json_schema", {})
            existing_schema = maybe_schema if isinstance(maybe_schema, dict) else {}
            maybe_required = existing_entry.get("required_fields", [])
            if isinstance(maybe_required, list):
                existing_required = [
                    str(field).strip() for field in maybe_required if str(field).strip()
                ]
            existing_notes = str(existing_entry.get("notes", "") or "").strip()

        merged_required = []
        for field_name in existing_required + cleaned_required:
            if field_name and field_name not in merged_required:
                merged_required.append(field_name)

        merged_schema = self._merge_custom_json_schema(
            existing_schema=existing_schema,
            incoming_schema=json_schema if isinstance(json_schema, dict) else {},
            explicit_required_fields=merged_required,
        )

        payload = {
            "doc_type": normalized_doc_type,
            "required_fields": merged_schema.get("required", []),
            "json_schema": merged_schema,
            "notes": str(notes or "").strip() or existing_notes,
        }

        if replaced:
            for index, entry in enumerate(entries):
                if str(entry.get("doc_type", "")).strip().lower() == normalized_doc_type:
                    entries[index] = payload
                    break
        else:
            entries.append(payload)

        self._write_custom_schema_entries(entries)
        self.refresh()
        return payload

    def list_profiles(self) -> List[Dict[str, Any]]:
        """List all schema profiles currently available in the knowledge registry."""
        profiles = []
        for _, doc_type, source, content, metadata in self._all_entries():
            profiles.append(
                {
                    "doc_type": doc_type,
                    "source": source,
                    "schema_fields": metadata.get("schema_fields", []),
                    "required_fields": metadata.get("required_fields", []),
                    "notes": metadata.get("notes", content),
                }
            )
        return profiles

    def _fetch_entry(self, doc_type: str) -> Optional[Tuple[int, str, str, str, Dict[str, Any]]]:
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT id, doc_type, source, title, content, metadata_json FROM knowledge_entries WHERE doc_type = ?",
            (doc_type,),
        )
        row = cursor.fetchone()
        if not row:
            return None
        metadata = {}
        try:
            metadata = json.loads(row[5]) if row[5] else {}
        except Exception:
            metadata = {}
        return row[0], row[1], row[2], row[4], metadata

    def _all_entries(self) -> List[Tuple[int, str, str, str, Dict[str, Any]]]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, doc_type, source, title, content, metadata_json FROM knowledge_entries ORDER BY id")
        rows = cursor.fetchall()
        results = []
        for row in rows:
            try:
                metadata = json.loads(row[5]) if row[5] else {}
            except Exception:
                metadata = {}
            results.append((row[0], row[1], row[2], row[4], metadata))
        return results

    def _rebuild_index(self) -> None:
        entries = self._all_entries()
        self.index.clear()
        if not entries:
            return

        texts = [entry[3] for entry in entries]
        metadata = [
            {
                "entry_id": entry[0],
                "doc_type": entry[1],
                "source": entry[2],
                "schema_fields": entry[4].get("schema_fields", []),
                "required_fields": entry[4].get("required_fields", []),
            }
            for entry in entries
        ]
        self.index.add_documents(texts, metadata)
        self.index.save()

    def search_knowledge(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        return self.index.search(query, k=k)

    def get_validation_profile(
        self,
        doc_type: str,
        custom_doc_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Return schema profile for validator.

        Resolution order:
          1) custom_doc_type exact match (file-backed custom schemas)
          2) doc_type exact match (builtin pydantic schemas)
          3) unknown fallback profile
        """
        target = (custom_doc_type or doc_type or "unknown").strip().lower()
        base_doc_type = str(doc_type or "unknown").strip().lower()

        resolution = "exact_target"
        row = self._fetch_entry(target)
        if row is None and target != base_doc_type:
            row = self._fetch_entry(base_doc_type)
            if row is not None:
                resolution = "fallback_base_doc_type"
        if row is None:
            row = self._fetch_entry("unknown")
            if row is not None:
                resolution = "fallback_unknown"

        if row is None:
            return {
                "doc_type": target,
                "schema_fields": [],
                "required_fields": [],
                "json_schema": {},
                "knowledge_notes": "",
                "semantic_hits": [],
                "source": "none",
            }

        _, resolved_doc_type, source, content, metadata = row

        semantic_hits = self.search_knowledge(
            query=f"validation schema for {target}",
            k=3,
        )

        notes = metadata.get("notes", "")
        if not notes:
            notes = content

        source_label = source
        if resolution == "fallback_base_doc_type":
            source_label = f"fallback_{source}_base_doc_type"
        elif resolution == "fallback_unknown":
            source_label = f"fallback_{source}_unknown"

        return {
            "doc_type": resolved_doc_type,
            "schema_fields": metadata.get("schema_fields", []),
            "required_fields": metadata.get("required_fields", []),
            "json_schema": metadata.get("json_schema", {}),
            "knowledge_notes": notes,
            "semantic_hits": semantic_hits,
            "source": source_label,
            "resolution": resolution,
            "target_doc_type": target,
        }


_knowledge_lookup: Optional[KnowledgeLookup] = None


def get_knowledge_lookup() -> KnowledgeLookup:
    global _knowledge_lookup
    if _knowledge_lookup is None:
        _knowledge_lookup = KnowledgeLookup()
    return _knowledge_lookup
