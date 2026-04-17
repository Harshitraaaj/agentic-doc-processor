"""
FastAPI Server for Document Processing API
"""
import asyncio
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import boto3

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from utils.config import settings
from graph.workflow import hitl_workflow
from agents.supervisor_agent import SupervisorAgent
from utils.logger import logger
from utils.graph_visualizer import graph_visualizer
from utils.knowledge_lookup import get_knowledge_lookup
from utils.observability import get_observer
from utils.service_registry import ServiceRegistry
from schemas.document_schemas import ProcessingResult


supervisor_agent = SupervisorAgent()
# Wire the HITLWorkflow shim back to the supervisor so any legacy callers
# that still go through hitl_workflow still reach the supervisor's graph.
hitl_workflow.set_supervisor(supervisor_agent)
observer = get_observer()


# Request/Response models
class ProcessRequest(BaseModel):
    """Request model for document processing"""
    file_path: str = Field(..., description="Local file path to process")
    thread_id: Optional[str] = Field(None, description="Optional thread ID for checkpointing")


class ProcessResponse(BaseModel):
    """Response model for document processing"""
    success: bool
    message: str
    doc_type: Optional[str] = None
    confidence: Optional[float] = None
    reasoning: Optional[str] = None
    extracted_fields: Optional[Dict[str, Any]] = None
    validation: Optional[Dict[str, Any]] = None
    redaction: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None
    errors: Optional[list[str]] = None
    knowledge_lookup_used: Optional[bool] = None
    schema_source: Optional[str] = None
    classification_path: Optional[str] = None  # normal_llm_classification | hitl_approved_classification | hitl_corrected_classification | hitl_custom_classification
    classification_predicted_doc_type: Optional[str] = None
    knowledge_requested_doc_type: Optional[str] = None
    knowledge_resolved_doc_type: Optional[str] = None
    supervisor_policy: Optional[Dict[str, Any]] = None
    processing_time: float
    timestamp: str
    trace_log: Optional[list[Dict[str, Any]]] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    timestamp: str
    llm_available: bool


class KnowledgeSchemaRequest(BaseModel):
    """Create or update a custom schema profile."""
    doc_type: str
    required_fields: List[str] = Field(default_factory=list)
    json_schema: Dict[str, Any] = Field(default_factory=dict)
    notes: Optional[str] = None


class KnowledgeSchemaResponse(BaseModel):
    """Knowledge schema API response."""
    success: bool
    message: str
    schema: Optional[Dict[str, Any]] = None
    profiles: Optional[List[Dict[str, Any]]] = None


# Create FastAPI app
app = FastAPI(
    title="Agentic Document Processor",
    description="Production-grade document processing with LangGraph + Amazon Bedrock",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def monitoring_middleware(request: Request, call_next):
    """Emit per-request LangSmith traces (no Prometheus/local metrics)."""
    start = time.perf_counter()
    obs_run_id = None
    use_langsmith_monitoring = (
        settings.LANGSMITH_ENABLED
        and settings.LANGSMITH_REQUEST_TRACES
        and observer.is_active
    )
    if use_langsmith_monitoring:
        obs_run_id = observer.start_run(
            name="monitoring.http_request",
            inputs={
                "method": request.method,
                "path": request.url.path,
                "query": str(request.url.query or ""),
            },
            tags=["monitoring", "http", "middleware"],
            metadata={"component": "fastapi_middleware"},
        )

    status_code = 500
    error_text = None
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    except Exception as e:
        status_code = 500
        error_text = str(e)
        raise
    finally:
        latency_ms = (time.perf_counter() - start) * 1000
        if use_langsmith_monitoring:
            observer.end_run(
                obs_run_id,
                outputs={
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": status_code,
                    "latency_ms": round(latency_ms, 3),
                },
                error=error_text,
            )


@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("FastAPI server starting up")

    # Compile orchestrated workflows via SupervisorAgent
    try:
        supervisor_agent.compile_workflows()
        logger.info("SupervisorAgent compiled all workflows successfully")
    except Exception as e:
        logger.error(f"Failed to compile workflows via SupervisorAgent: {e}")

    if settings.STARTUP_WARMUP_ENABLED:
        # Warm up Presidio NLP engine (spaCy loads lazily on first .analyze() call — 2-3s)
        # Running a dummy call here means zero extra latency on the first real document.
        try:
            from agents.redactor_agent import redactor_agent
            if redactor_agent.analyzer:
                redactor_agent.analyzer.analyze(text="warmup", language="en")
                logger.info("Presidio NLP engine warmed up successfully")
        except Exception as e:
            logger.warning(f"Presidio warm-up failed (non-fatal): {e}")

        # Warm up FAISS + embedding path so model/state loads at startup,
        # not on first validator/knowledge lookup call.
        try:
            get_knowledge_lookup()
            logger.info("Knowledge lookup warmed up successfully")
        except Exception as e:
            logger.warning(f"Knowledge lookup warm-up failed (non-fatal): {e}")
    else:
        logger.info("Startup warm-up disabled by configuration")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("FastAPI server shutting down")


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "service": "Agentic Document Processor",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    from utils.llm_client import llm_client
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.utcnow().isoformat(),
        llm_available=llm_client.is_available()
    )


@app.get("/monitoring/health", response_model=Dict[str, Any])
async def monitoring_health() -> Dict[str, Any]:
    """Return runtime status of LangSmith observability."""
    return {
        "langsmith_request_traces": settings.LANGSMITH_REQUEST_TRACES,
        "langsmith_enabled": settings.LANGSMITH_ENABLED,
        "langsmith_active": observer.is_active,
        "langsmith_project": settings.LANGSMITH_PROJECT,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/cloud/health", response_model=Dict[str, Any])
async def cloud_health() -> Dict[str, Any]:
    """Return cloud integration health for S3, Secrets Manager, and observability."""
    region = settings.AWS_REGION
    bucket = settings.AWS_S3_BUCKET
    groq_secret_name = settings.AWS_GROQ_SECRET_NAME
    langsmith_secret_name = settings.AWS_LANGSMITH_SECRET_NAME

    s3_ok = False
    s3_error = None
    uploads_objects = 0

    try:
        s3 = boto3.client("s3", region_name=region)
        s3.list_objects_v2(Bucket=bucket, Prefix="uploads/", MaxKeys=1)
        s3_ok = True

        uploads_resp = s3.list_objects_v2(Bucket=bucket, Prefix="uploads/", MaxKeys=1000)
        uploads_objects = uploads_resp.get("KeyCount", 0)
    except Exception as e:
        s3_error = str(e)

    groq_secret = settings._fetch_aws_secret_dict(region=region, secret_name=groq_secret_name)
    langsmith_secret = settings._fetch_aws_secret_dict(region=region, secret_name=langsmith_secret_name)

    groq_has_primary = bool(
        groq_secret.get("api_key_primary")
        or groq_secret.get("GROQ_API_KEY")
        or groq_secret.get("api_key")
    )
    groq_has_secondary = bool(
        groq_secret.get("api_key_secondary")
        or groq_secret.get("GROQ_API_KEY_B")
    )
    groq_has_tertiary = bool(
        groq_secret.get("api_key_tertiary")
        or groq_secret.get("GROQ_API_KEY_C")
    )
    langsmith_has_key = bool(
        langsmith_secret.get("api_key")
        or langsmith_secret.get("LANGSMITH_API_KEY")
    )

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "region": region,
        "storage_provider": settings.STACK_STORAGE_PROVIDER,
        "ocr_provider": settings.STACK_OCR_PROVIDER,
        "s3": {
            "bucket": bucket,
            "ok": s3_ok,
            "uploads_object_count": uploads_objects,
            "error": s3_error,
        },
        "secrets": {
            "groq_secret_name": groq_secret_name,
            "groq_fetched": bool(groq_secret),
            "groq_has_primary": groq_has_primary,
            "groq_has_secondary": groq_has_secondary,
            "groq_has_tertiary": groq_has_tertiary,
            "langsmith_secret_name": langsmith_secret_name,
            "langsmith_fetched": bool(langsmith_secret),
            "langsmith_has_api_key": langsmith_has_key,
        },
        "langsmith": {
            "enabled": settings.LANGSMITH_ENABLED,
            "request_traces": settings.LANGSMITH_REQUEST_TRACES,
            "active_client": observer.is_active,
            "project": settings.LANGSMITH_PROJECT,
        },
    }


@app.post("/monitoring/langsmith/test", response_model=Dict[str, Any])
async def monitoring_langsmith_test() -> Dict[str, Any]:
    """Force-create a monitoring trace in LangSmith for verification."""
    run_id = observer.start_run(
        name="monitoring.manual_test",
        inputs={"source": "api", "purpose": "langsmith_verification"},
        tags=["monitoring", "langsmith", "manual_test"],
        metadata={"endpoint": "/monitoring/langsmith/test"},
    )
    observer.end_run(run_id, outputs={"ok": True, "created_at": datetime.utcnow().isoformat()})
    return {
        "ok": True,
        "langsmith_active": observer.is_active,
        "langsmith_project": settings.LANGSMITH_PROJECT,
        "run_id": run_id,
    }


@app.post("/process", response_model=ProcessResponse)
async def process_document(request: ProcessRequest):
    """
    Process a document through the agentic pipeline
    
    Args:
        request: ProcessRequest with file_path
    
    Returns:
        ProcessResponse with results
    """
    start_time = datetime.utcnow()
    obs_run_id = observer.start_run(
        name="api.process",
        inputs={"file_path": request.file_path, "thread_id": request.thread_id},
        tags=["api", "process", "standard"],
        metadata={"endpoint": "/process"},
    )
    
    try:
        logger.info(f"API: Processing document: {request.file_path}")
        
        # Validate file exists - handle both absolute and relative paths
        file_path = Path(request.file_path)
        if not file_path.is_absolute():
            # Resolve relative paths against project root
            file_path = settings.PROJECT_ROOT / file_path
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")
        
        # Process document
        thread_id = request.thread_id or f"api_{datetime.utcnow().timestamp()}"
        
        # Run workflow in executor to avoid blocking
        loop = asyncio.get_event_loop()
        final_state = await loop.run_in_executor(
            None,
            supervisor_agent.process_document,
            str(file_path),
            thread_id
        )
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Extract detailed results from final_state
        classification_result = final_state.get("classification_result")
        extraction_result = final_state.get("extraction_result")
        validation_result = final_state.get("validation_result")
        redaction_result = final_state.get("redaction_result")
        custom_doc_type = final_state.get("custom_doc_type")
        hitl_required = bool(final_state.get("hitl_required", False))
        classify_hitl_resolution = final_state.get("hitl_resolution") if final_state.get("hitl_type") == "classify" else None
        classification_path = (
            "hitl_custom_classification"
            if custom_doc_type
            else "hitl_corrected_classification"
            if hitl_required and classify_hitl_resolution == "corrected"
            else "hitl_approved_classification"
            if hitl_required
            else "normal_llm_classification"
        )
        predicted_doc_type = (
            classification_result.doc_type.value
            if classification_result and getattr(classification_result, "doc_type", None)
            else "unknown"
        )
        resolved_doc_type = (
            custom_doc_type
            if custom_doc_type
            else final_state.get("doc_type").value if final_state.get("doc_type") else None
        )
        serialized_trace = [
            log.model_dump() if hasattr(log, 'model_dump') else log.dict() if hasattr(log, 'dict') else log
            for log in final_state.get("trace_log", [])
        ]
        schema_source = None
        knowledge_requested_doc_type = custom_doc_type or (final_state.get("doc_type").value if final_state.get("doc_type") else None)
        knowledge_resolved_doc_type = None
        knowledge_lookup_used = False
        for log in reversed(serialized_trace):
            if (log.get("agent_name") == "ValidatorAgent"):
                context = log.get("context_data") or {}
                schema_source = context.get("knowledge_source")
                knowledge_requested_doc_type = context.get("knowledge_requested_doc_type") or knowledge_requested_doc_type
                knowledge_resolved_doc_type = context.get("knowledge_resolved_doc_type")
                knowledge_lookup_used = bool(schema_source)
                break
        
        # Build response
        response = ProcessResponse(
            success=final_state.get("success", False),
            message="Document processed successfully" if final_state.get("success") else "Processing completed with errors",
            doc_type=resolved_doc_type,
            confidence=classification_result.confidence if classification_result else None,
            reasoning=classification_result.reasoning if classification_result else None,
            extracted_fields=final_state.get("extracted_fields"),
            validation={
                "status": validation_result.status.value if validation_result else "unknown",
                "is_valid": validation_result.is_valid if validation_result else False,
                "errors": validation_result.errors if validation_result else [],
                "warnings": validation_result.warnings if validation_result else []
            } if validation_result else None,
            redaction={
                "pii_count": redaction_result.pii_count if redaction_result else 0,
                "precision": redaction_result.precision if redaction_result else 0,
                "recall": redaction_result.recall if redaction_result else 0,
                "pii_detections": [
                    {
                        "pii_type": det.pii_type.value if hasattr(det.pii_type, 'value') else str(det.pii_type),
                        "original_text": det.original_text,
                        "redacted_text": det.redacted_text,
                        "detection_source": det.detection_source if hasattr(det, 'detection_source') else "unknown"
                    }
                    for det in (redaction_result.pii_detections if hasattr(redaction_result, 'pii_detections') else redaction_result.detected_pii if hasattr(redaction_result, 'detected_pii') else [])
                ] if redaction_result else [],
                "redacted_text": redaction_result.redacted_text if redaction_result else None
            } if redaction_result else None,
            metrics=final_state.get("metrics"),
            errors=final_state.get("errors", []),
            knowledge_lookup_used=knowledge_lookup_used,
            schema_source=schema_source,
            classification_path=classification_path,
            classification_predicted_doc_type=predicted_doc_type,
            knowledge_requested_doc_type=knowledge_requested_doc_type,
            knowledge_resolved_doc_type=knowledge_resolved_doc_type,
            supervisor_policy=_extract_supervisor_policy(final_state),
            processing_time=processing_time,
            timestamp=datetime.utcnow().isoformat(),
            trace_log=serialized_trace
        )
        
        logger.info(
            f"API: Document processing complete",
            file=request.file_path,
            success=response.success,
            processing_time=processing_time
        )
        observer.end_run(
            obs_run_id,
            outputs={
                "status": "complete",
                "success": response.success,
                "doc_type": response.doc_type,
                "classification_path": response.classification_path,
                "processing_time": response.processing_time,
            },
        )
        
        return response
    
    except HTTPException:
        observer.end_run(obs_run_id, error="http_exception")
        raise
    except Exception as e:
        logger.error(f"API: Processing failed: {e}", exc_info=True)
        observer.end_run(obs_run_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload-and-process")
async def upload_and_process(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Upload a file and process it
    
    Args:
        file: Uploaded file
        background_tasks: Background tasks
    
    Returns:
        ProcessResponse
    """
    try:
        logger.info(f"API: Uploading file: {file.filename}")
        storage = ServiceRegistry.get_storage()
        
        # Save uploaded file temporarily
        temp_dir = settings.DATA_DIR / "uploads"
        temp_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_file_path = temp_dir / f"{timestamp}_{file.filename}"
        
        # Write file
        contents = await file.read()
        with open(temp_file_path, 'wb') as f:
            f.write(contents)

        # Persist uploaded bytes using configured storage backend (S3/local).
        # Processing still uses local temp path to minimize workflow changes.
        storage_key = f"uploads/{timestamp}_{file.filename}"
        storage_uri = storage.put_file(
            key=storage_key,
            data=contents,
            content_type=file.content_type,
        )
        
        logger.info(
            "API: File saved",
            temp_file_path=str(temp_file_path),
            storage_uri=storage_uri,
            storage_provider=settings.STACK_STORAGE_PROVIDER,
        )
        
        # Process document
        request = ProcessRequest(file_path=str(temp_file_path))
        response = await process_document(request)
        
        # Optionally clean up file in background
        # if background_tasks:
        #     background_tasks.add_task(temp_file_path.unlink, missing_ok=True)
        
        return response
    
    except Exception as e:
        logger.error(f"API: Upload and process failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/visualize/graph")
async def visualize_graph(format: str = "mermaid", mode: str = "supervisor"):
    """
    Get workflow graph visualization
    
    Args:
        format: Output format - "mermaid" or "json"
    
    Returns:
        Mermaid diagram or graph JSON
    """
    try:
        mode_normalized = (mode or "supervisor").strip().lower()
        target_workflow = supervisor_agent if mode_normalized == "supervisor" else workflow

        if format == "mermaid":
            mermaid_diagram = graph_visualizer.generate_mermaid_diagram(target_workflow)
            return {
                "format": "mermaid",
                "mode": mode_normalized,
                "diagram": mermaid_diagram
            }
        elif format == "json":
            # Return graph structure as JSON
            if mode_normalized == "supervisor":
                supervisor_agent.compile_workflows()
                graph = supervisor_agent._compiled.get_graph()
            else:
                if workflow.compiled_graph is None:
                    workflow.compile()
                graph = workflow.compiled_graph.get_graph()

            return {
                "format": "json",
                "mode": mode_normalized,
                "nodes": [node for node in graph.nodes],
                "edges": [(edge[0], edge[1]) for edge in graph.edges]
            }
        else:
            raise HTTPException(status_code=400, detail="Format must be 'mermaid' or 'json'")
    
    except Exception as e:
        logger.error(f"API: Graph visualization failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/visualize/trace")
async def visualize_execution_trace(trace_log: List[Dict[str, Any]], mode: str = "supervisor"):
    """
    Generate execution trace visualization from trace log (sequence diagram)
    
    Args:
        trace_log: List of ResponsibleAILog entries
    
    Returns:
        Mermaid sequence diagram showing actual execution flow
    """
    try:
        sequence_diagram = graph_visualizer.visualize_execution_trace(trace_log, mode=mode)
        
        # Extract execution path
        execution_path = graph_visualizer.extract_execution_path(trace_log, mode=mode)
        
        return {
            "format": "mermaid",
            "mode": mode,
            "diagram": sequence_diagram,
            "execution_path": execution_path,
            "total_steps": len(trace_log)
        }
    
    except Exception as e:
        logger.error(f"API: Trace visualization failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/visualize/execution-path")
async def visualize_execution_path(trace_log: List[Dict[str, Any]], mode: str = "supervisor"):
    """
    Generate execution path diagram showing the actual path taken through the workflow
    
    Args:
        trace_log: List of ResponsibleAILog entries from document processing
    
    Returns:
        Mermaid diagram with highlighted execution path (classify -> extract -> validate -> etc)
    """
    try:
        # Generate path diagram with highlighted nodes
        path_diagram = graph_visualizer.generate_execution_path_diagram(trace_log, mode=mode)
        
        # Extract execution path for reference
        execution_path = graph_visualizer.extract_execution_path(trace_log, mode=mode)
        
        # Count repair attempts
        repair_count = execution_path.count("repair")
        
        return {
            "format": "mermaid",
            "mode": mode,
            "diagram": path_diagram,
            "execution_path": execution_path,
            "repair_attempts": repair_count,
            "total_steps": len(trace_log)
        }
    
    except Exception as e:
        logger.error(f"API: Execution path visualization failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
        if background_tasks:
            background_tasks.add_task(temp_file_path.unlink, missing_ok=True)
        
        return response
    
    except Exception as e:
        logger.error(f"API: Upload and process failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/workflow/diagram")
async def get_workflow_diagram(mode: str = "supervisor"):
    """
    Get Mermaid diagram of the workflow
    
    Returns:
        Mermaid diagram string
    """
    try:
        from utils.graph_visualizer import graph_visualizer

        mode_normalized = (mode or "supervisor").strip().lower()
        target_workflow = supervisor_agent if mode_normalized == "supervisor" else workflow
        mermaid_diagram = graph_visualizer.generate_mermaid_diagram(target_workflow)
        return {
            "diagram": mermaid_diagram,
            "mode": mode_normalized,
            "format": "mermaid",
            "visualizer_url": "https://mermaid.live/edit"
        }
    except Exception as e:
        logger.error(f"Failed to get workflow diagram: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics/summary")
async def get_metrics_summary():
    """
    Get summary of recent processing metrics
    
    Returns:
        Metrics summary
    """
    try:
        # Read recent report files
        import json
        
        reports_dir = settings.REPORTS_DIR
        report_files = sorted(
            reports_dir.glob("metrics_report_*.json"),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )[:10]  # Last 10 reports
        
        summaries = []
        for report_file in report_files:
            with open(report_file, 'r') as f:
                report_data = json.load(f)
                summaries.append({
                    "document": report_data.get("document"),
                    "timestamp": report_data.get("timestamp"),
                    "doc_type": report_data.get("doc_type"),
                    "extraction_accuracy": report_data.get("metrics", {}).get("extraction_accuracy"),
                    "workflow_success": report_data.get("metrics", {}).get("workflow_success")
                })
        
        return {"recent_reports": summaries}
    
    except Exception as e:
        logger.error(f"Failed to get metrics summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/knowledge/schemas", response_model=KnowledgeSchemaResponse)
async def list_knowledge_schemas():
    """List all currently registered schema profiles."""
    try:
        knowledge_lookup = get_knowledge_lookup()
        return KnowledgeSchemaResponse(
            success=True,
            message="Knowledge schema profiles retrieved successfully",
            profiles=knowledge_lookup.list_profiles(),
        )
    except Exception as e:
        logger.error(f"Failed to list knowledge schemas: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/knowledge/schemas", response_model=KnowledgeSchemaResponse)
async def register_knowledge_schema(request: KnowledgeSchemaRequest):
    """Register or update a custom schema profile and rebuild retrieval state."""
    try:
        knowledge_lookup = get_knowledge_lookup()
        schema = knowledge_lookup.register_custom_schema(
            doc_type=request.doc_type,
            required_fields=request.required_fields,
            json_schema=request.json_schema,
            notes=request.notes or "",
        )
        return KnowledgeSchemaResponse(
            success=True,
            message=f"Custom schema '{schema['doc_type']}' registered successfully",
            schema=schema,
        )
    except Exception as e:
        logger.error(f"Failed to register knowledge schema: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/knowledge/refresh", response_model=KnowledgeSchemaResponse)
async def refresh_knowledge_schemas():
    """Force reload of schema storage and FAISS index."""
    try:
        knowledge_lookup = get_knowledge_lookup()
        knowledge_lookup.refresh()
        return KnowledgeSchemaResponse(
            success=True,
            message="Knowledge lookup refreshed successfully",
        )
    except Exception as e:
        logger.error(f"Failed to refresh knowledge lookup: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# HITL Agentic Endpoints  (LangGraph interrupt() based)
# ─────────────────────────────────────────────────────────────────────────────

class StartProcessingRequest(BaseModel):
    """Start a new HITL-aware processing run."""
    file_path: str
    thread_id: Optional[str] = None


class ResumeRequest(BaseModel):
    """Human decision sent to resume a suspended graph."""
    resolution: str                             # "approved" | "corrected" | "rejected"
    doc_type_override: Optional[str] = None     # classify checkpoint only
    corrections: Optional[Dict[str, Any]] = None  # extract checkpoint only


class AutoProcessRequest(BaseModel):
    """Supervisor auto policy request for selecting standard vs HITL path."""
    file_path: str
    thread_id: Optional[str] = None
    preferred_mode: Optional[str] = "auto"  # auto | standard | hitl


class ProcessingStatusResponse(BaseModel):
    """
    Unified response for both /process/start and /thread/{id}/resume.

    • status == "interrupted": graph is suspended; interrupt_data contains the
      hitl_type and fields needed to render the review UI.
    • status == "complete":    graph finished; result contains the full ProcessResponse.
    """
    thread_id: str
    status: str                                    # "interrupted" | "complete"
    interrupt_data: Optional[Dict[str, Any]] = None
    result: Optional[Dict[str, Any]] = None        # ProcessResponse dict when complete
    supervisor_policy: Optional[Dict[str, Any]] = None


def _extract_supervisor_policy(state: Dict[str, Any]) -> Dict[str, Any]:
    """Extract normalized supervisor policy metadata from workflow state."""
    if not state:
        return {}
    return {
        "mode": state.get("supervisor_mode"),
        "classification_confidence": state.get("supervisor_classification_confidence"),
        "classification_decision": state.get("supervisor_classification_decision"),
        "classification_reason": state.get("supervisor_classification_reason"),
        "validation_decision": state.get("supervisor_validation_decision"),
        "validation_reason": state.get("supervisor_validation_reason"),
    }


def _serialize_trace(trace_log: list) -> List[Dict[str, Any]]:
    """Convert ResponsibleAILog objects → plain dicts."""
    out = []
    for log in trace_log:
        if hasattr(log, "model_dump"):
            out.append(log.model_dump())
        elif hasattr(log, "dict"):
            out.append(log.dict())
        elif isinstance(log, dict):
            out.append(log)
    return out


def _persist_source_file_to_storage(file_path: Path, *, category: str = "uploads") -> Optional[str]:
    """Persist a source document to configured storage backend (S3/local)."""
    try:
        storage = ServiceRegistry.get_storage()
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        storage_prefix = (category or "uploads").strip("/")
        if not storage_prefix:
            storage_prefix = "uploads"
        storage_key = f"{storage_prefix}/{timestamp}_{file_path.name}"
        content = file_path.read_bytes()
        return storage.put_file(key=storage_key, data=content)
    except Exception as exc:
        logger.warning(
            "API: Failed to persist source file to storage",
            file_path=str(file_path),
            storage_provider=settings.STACK_STORAGE_PROVIDER,
            error=str(exc),
        )
        if settings.STACK_STORAGE_PROVIDER.lower().strip() in {"s3", "aws_s3"}:
            raise
        return None


def _build_process_response_from_state(
    final_state: Dict[str, Any],
    start_time: datetime,
) -> Dict[str, Any]:
    """Build a ProcessResponse-shaped dict from a completed HITLWorkflow final_state."""
    processing_time = (datetime.utcnow() - start_time).total_seconds()
    cr  = final_state.get("classification_result")
    vr  = final_state.get("validation_result")
    rr  = final_state.get("redaction_result")

    def _pii_list(redaction_result):
        source = (
            getattr(redaction_result, "pii_detections", None)
            or getattr(redaction_result, "detected_pii", None)
            or []
        )
        return [
            {
                "pii_type":        det.pii_type.value if hasattr(det.pii_type, "value") else str(det.pii_type),
                "original_text":   det.original_text,
                "redacted_text":   det.redacted_text,
                "detection_source": getattr(det, "detection_source", "unknown"),
            }
            for det in source
        ]

    doc_type_val = final_state.get("doc_type")
    custom_doc_type = final_state.get("custom_doc_type")
    hitl_required = bool(final_state.get("hitl_required", False))
    classify_hitl_resolution = final_state.get("hitl_resolution") if final_state.get("hitl_type") == "classify" else None
    classification_path = (
        "hitl_custom_classification"
        if custom_doc_type
        else "hitl_corrected_classification"
        if hitl_required and classify_hitl_resolution == "corrected"
        else "hitl_approved_classification"
        if hitl_required
        else "normal_llm_classification"
    )
    predicted_doc_type = (
        cr.doc_type.value
        if cr and getattr(cr, "doc_type", None)
        else "unknown"
    )
    resolved_doc_type = (
        custom_doc_type
        if custom_doc_type
        else doc_type_val.value if doc_type_val and hasattr(doc_type_val, "value") else str(doc_type_val or "unknown")
    )
    serialized_trace = _serialize_trace(final_state.get("trace_log", []))
    schema_source = None
    knowledge_requested_doc_type = custom_doc_type or (doc_type_val.value if doc_type_val and hasattr(doc_type_val, "value") else str(doc_type_val or "unknown"))
    knowledge_resolved_doc_type = None
    knowledge_lookup_used = False
    for log in reversed(serialized_trace):
        if log.get("agent_name") == "ValidatorAgent":
            context = log.get("context_data") or {}
            schema_source = context.get("knowledge_source")
            knowledge_requested_doc_type = context.get("knowledge_requested_doc_type") or knowledge_requested_doc_type
            knowledge_resolved_doc_type = context.get("knowledge_resolved_doc_type")
            knowledge_lookup_used = bool(schema_source)
            break

    return {
        "success":          final_state.get("success", False),
        "message":          (
            "Document processed successfully with human review"
            if final_state.get("success")
            else "Processing completed with issues"
        ),
        "doc_type":         resolved_doc_type,
        "confidence":       float(cr.confidence) if cr else None,
        "reasoning":        cr.reasoning if cr else None,
        "extracted_fields": final_state.get("extracted_fields"),
        "validation": {
            "status":   vr.status.value if vr else "unknown",
            "is_valid": vr.is_valid if vr else False,
            "errors":   vr.errors if vr else [],
            "warnings": vr.warnings if vr else [],
        } if vr else None,
        "redaction": {
            "pii_count":      rr.pii_count if rr else 0,
            "precision":      rr.precision if rr else 0,
            "recall":         rr.recall if rr else 0,
            "pii_detections": _pii_list(rr) if rr else [],
            "redacted_text":  rr.redacted_text if rr else None,
        } if rr else None,
        "metrics":          final_state.get("metrics"),
        "errors":           final_state.get("errors", []),
        "knowledge_lookup_used": knowledge_lookup_used,
        "schema_source": schema_source,
        "classification_path": classification_path,
        "classification_predicted_doc_type": predicted_doc_type,
        "knowledge_requested_doc_type": knowledge_requested_doc_type,
        "knowledge_resolved_doc_type": knowledge_resolved_doc_type,
        "supervisor_policy": _extract_supervisor_policy(final_state),
        "processing_time":  processing_time,
        "timestamp":        datetime.utcnow().isoformat(),
        "trace_log":        serialized_trace,
    }


@app.post("/process/start", response_model=ProcessingStatusResponse)
async def start_document_processing(request: StartProcessingRequest):
    """
    Agentic HITL endpoint — start a new document processing run.

    Runs the LangGraph HITLWorkflow until the first interrupt() fires
    (always at the classification review node), then returns for human input.

    The caller receives {thread_id, status:"interrupted", interrupt_data}
    and renders the appropriate review UI.  When the human submits their
    decision, POST /thread/{thread_id}/resume to continue.
    """
    start_time = datetime.utcnow()
    obs_run_id = observer.start_run(
        name="api.process_start",
        inputs={"file_path": request.file_path, "thread_id": request.thread_id},
        tags=["api", "process", "hitl", "start"],
        metadata={"endpoint": "/process/start"},
    )

    file_path = Path(request.file_path)
    if not file_path.is_absolute():
        file_path = settings.PROJECT_ROOT / file_path
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")

        storage_uri = _persist_source_file_to_storage(file_path, category="uploads")
        if storage_uri:
            logger.info("API: Source file persisted", storage_uri=storage_uri)

    storage_uri = _persist_source_file_to_storage(file_path, category="uploads")
    if storage_uri:
        logger.info("API: HITL source file persisted", storage_uri=storage_uri)

    thread_id = request.thread_id or f"hitl_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"

    try:
        loop = asyncio.get_event_loop()
        outcome = await loop.run_in_executor(
            None, supervisor_agent.start_processing, str(file_path), thread_id
        )
    except ValueError as e:
        observer.end_run(obs_run_id, error=str(e))
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"API /process/start failed: {e}", exc_info=True)
        observer.end_run(obs_run_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

    if outcome["status"] == "interrupted":
        observer.end_run(
            obs_run_id,
            outputs={
                "status": "interrupted",
                "thread_id": thread_id,
                "hitl_type": (outcome.get("interrupt_data") or {}).get("hitl_type"),
            },
        )
        return ProcessingStatusResponse(
            thread_id=thread_id,
            status="interrupted",
            interrupt_data=outcome["interrupt_data"],
            supervisor_policy=_extract_supervisor_policy(outcome.get("state", {})),
        )
    else:
        result = _build_process_response_from_state(outcome["final_state"], start_time)
        observer.end_run(
            obs_run_id,
            outputs={
                "status": "complete",
                "thread_id": thread_id,
                "success": result.get("success", False),
                "doc_type": result.get("doc_type"),
            },
        )
        return ProcessingStatusResponse(
            thread_id=thread_id,
            status="complete",
            result=result,
            supervisor_policy=result.get("supervisor_policy"),
        )


@app.post("/thread/{thread_id}/resume", response_model=ProcessingStatusResponse)
async def resume_document_processing(thread_id: str, request: ResumeRequest):
    """
    Agentic HITL endpoint — resume a suspended graph with human input.

    The graph was suspended at an interrupt() call.  This endpoint passes
    the human decision back via LangGraph Command(resume=...), which makes
    the interrupt() inside the node evaluate to the human_input dict and the
    graph continues from that exact point.

    If the graph hits another interrupt (extraction review), returns
    {status:"interrupted", interrupt_data}.  If it completes, returns
    {status:"complete", result}.
    """
    start_time = datetime.utcnow()
    obs_run_id = observer.start_run(
        name="api.process_resume",
        inputs={
            "thread_id": thread_id,
            "resolution": request.resolution,
            "doc_type_override": request.doc_type_override,
            "has_corrections": bool(request.corrections),
        },
        tags=["api", "process", "hitl", "resume"],
        metadata={"endpoint": "/thread/{thread_id}/resume"},
    )

    human_input = {
        "resolution":        request.resolution,
        "doc_type_override": request.doc_type_override,
        "corrections":       request.corrections,
    }

    try:
        loop = asyncio.get_event_loop()
        outcome = await loop.run_in_executor(
            None, supervisor_agent.resume_processing, thread_id, human_input
        )
    except Exception as e:
        logger.error(f"API /thread/{thread_id}/resume failed: {e}", exc_info=True)
        observer.end_run(obs_run_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

    if outcome["status"] == "interrupted":
        observer.end_run(
            obs_run_id,
            outputs={
                "status": "interrupted",
                "thread_id": thread_id,
                "hitl_type": (outcome.get("interrupt_data") or {}).get("hitl_type"),
            },
        )
        return ProcessingStatusResponse(
            thread_id=thread_id,
            status="interrupted",
            interrupt_data=outcome["interrupt_data"],
            supervisor_policy=_extract_supervisor_policy(outcome.get("state", {})),
        )
    else:
        result = _build_process_response_from_state(outcome["final_state"], start_time)
        observer.end_run(
            obs_run_id,
            outputs={
                "status": "complete",
                "thread_id": thread_id,
                "success": result.get("success", False),
                "doc_type": result.get("doc_type"),
            },
        )
        return ProcessingStatusResponse(
            thread_id=thread_id,
            status="complete",
            result=result,
            supervisor_policy=result.get("supervisor_policy"),
        )


@app.post("/process/auto", response_model=ProcessingStatusResponse)
async def auto_process_document(request: AutoProcessRequest):
    """
    Supervisor auto policy endpoint.

    Policy:
      - `preferred_mode=standard` => run /process-style standard workflow
      - `preferred_mode=hitl`     => run HITL workflow start
      - `preferred_mode=auto`     => route to HITL workflow with internal
                                     confidence/validation supervisor gates
    """
    start_time = datetime.utcnow()
    obs_run_id = observer.start_run(
        name="api.process_auto",
        inputs={
            "file_path": request.file_path,
            "thread_id": request.thread_id,
            "preferred_mode": request.preferred_mode,
        },
        tags=["api", "process", "auto"],
        metadata={"endpoint": "/process/auto"},
    )

    file_path = Path(request.file_path)
    if not file_path.is_absolute():
        file_path = settings.PROJECT_ROOT / file_path
    if not file_path.exists():
        observer.end_run(obs_run_id, error=f"file_not_found:{request.file_path}")
        raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")

    storage_uri = _persist_source_file_to_storage(file_path, category="uploads")
    if storage_uri:
        logger.info("API: Auto source file persisted", storage_uri=storage_uri)

    thread_id = request.thread_id or f"auto_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"
    preferred_mode = (request.preferred_mode or "auto").strip().lower()

    if preferred_mode not in {"auto", "standard", "hitl"}:
        observer.end_run(obs_run_id, error=f"invalid_preferred_mode:{preferred_mode}")
        raise HTTPException(status_code=400, detail="preferred_mode must be one of: auto, standard, hitl")

    loop = asyncio.get_event_loop()

    # Standard mode: always complete (no interrupt)
    if preferred_mode == "standard":
        try:
            final_state = await loop.run_in_executor(
                None,
                supervisor_agent.process_document,
                str(file_path),
                thread_id,
            )
        except Exception as e:
            logger.error(f"API /process/auto (standard) failed: {e}", exc_info=True)
            observer.end_run(obs_run_id, error=str(e))
            raise HTTPException(status_code=500, detail=str(e))

        final_state["supervisor_mode"] = "standard"
        final_state["supervisor_classification_decision"] = "standard_pipeline"
        final_state["supervisor_classification_reason"] = "preferred_mode=standard"
        result = _build_process_response_from_state(final_state, start_time)
        observer.end_run(
            obs_run_id,
            outputs={
                "status": "complete",
                "mode": "standard",
                "thread_id": thread_id,
                "success": result.get("success", False),
                "doc_type": result.get("doc_type"),
            },
        )
        return ProcessingStatusResponse(
            thread_id=thread_id,
            status="complete",
            result=result,
            supervisor_policy=result.get("supervisor_policy"),
        )

    # HITL/Auto mode: run HITL workflow start, may interrupt based on policy
    try:
        outcome = await loop.run_in_executor(
            None,
            supervisor_agent.start_processing,
            str(file_path),
            thread_id,
        )
    except Exception as e:
        logger.error(f"API /process/auto ({preferred_mode}) failed: {e}", exc_info=True)
        observer.end_run(obs_run_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

    # If caller requested explicit hitl mode, annotate policy reason
    if outcome.get("state") is not None and preferred_mode == "hitl":
        outcome["state"]["supervisor_mode"] = "forced_hitl"
        outcome["state"]["supervisor_classification_reason"] = "preferred_mode=hitl"

    if outcome["status"] == "interrupted":
        observer.end_run(
            obs_run_id,
            outputs={
                "status": "interrupted",
                "mode": preferred_mode,
                "thread_id": thread_id,
                "hitl_type": (outcome.get("interrupt_data") or {}).get("hitl_type"),
            },
        )
        return ProcessingStatusResponse(
            thread_id=thread_id,
            status="interrupted",
            interrupt_data=outcome["interrupt_data"],
            supervisor_policy=_extract_supervisor_policy(outcome.get("state", {})),
        )

    final_state = outcome["final_state"]
    if preferred_mode == "hitl":
        final_state["supervisor_mode"] = "forced_hitl"
        final_state["supervisor_classification_reason"] = "preferred_mode=hitl"

    result = _build_process_response_from_state(final_state, start_time)
    observer.end_run(
        obs_run_id,
        outputs={
            "status": "complete",
            "mode": preferred_mode,
            "thread_id": thread_id,
            "success": result.get("success", False),
            "doc_type": result.get("doc_type"),
        },
    )
    return ProcessingStatusResponse(
        thread_id=thread_id,
        status="complete",
        result=result,
        supervisor_policy=result.get("supervisor_policy"),
    )


if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting FastAPI server on {settings.API_HOST}:{settings.API_PORT}")
    
    uvicorn.run(
        "api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD,
        log_level=settings.LOG_LEVEL.lower()
    )
