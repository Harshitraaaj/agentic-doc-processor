"""
Compatibility config loader placed under `utils`.
Reads `config.ini` and `.env` from project root and exposes `Env` and `settings`.
This replaces the previous top-level `env.py` module.
"""
import os
import json
from pathlib import Path
from configparser import ConfigParser
from typing import Any


class Env:
    def __init__(self):
        # project root is two levels up from this file
        self.PROJECT_ROOT = Path(__file__).parent.parent
        self.config = ConfigParser()
        config_file = self.PROJECT_ROOT / "config.ini"
        if not config_file.exists():
            raise FileNotFoundError(f"config.ini not found at {config_file}")
        self.config.read(config_file)

        env_file = self.PROJECT_ROOT / ".env"
        if env_file.exists():
            self._load_env_file(env_file)

        skip_hf_from_config = self.config.getboolean('runtime', 'skip_hf', fallback=False)
        if skip_hf_from_config and not os.environ.get("SKIP_HF"):
            os.environ["SKIP_HF"] = "true"

        self._load_aws_secrets()

    def _load_env_file(self, env_file: Path) -> None:
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

    def _fetch_aws_secret_dict(self, *, region: str, secret_name: str) -> dict[str, Any]:
        if not secret_name:
            return {}
        try:
            import boto3
            client = boto3.client("secretsmanager", region_name=region)
            response = client.get_secret_value(SecretId=secret_name)
            secret_text = response.get("SecretString", "")
            if not secret_text:
                return {}
            data = json.loads(secret_text)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _load_aws_secrets(self) -> None:
        region = os.environ.get("AWS_REGION") or self.config.get('aws', 'region', fallback='us-east-1')
        groq_secret_name = self.config.get('aws', 'groq_secret_name', fallback='').strip()
        langsmith_secret_name = self.config.get('aws', 'langsmith_secret_name', fallback='').strip()

        if groq_secret_name:
            groq_secret = self._fetch_aws_secret_dict(region=region, secret_name=groq_secret_name)
            primary = (
                groq_secret.get("api_key_primary")
                or groq_secret.get("GROQ_API_KEY")
                or groq_secret.get("api_key")
            )
            secondary = groq_secret.get("api_key_secondary") or groq_secret.get("GROQ_API_KEY_B")
            tertiary = groq_secret.get("api_key_tertiary") or groq_secret.get("GROQ_API_KEY_C")

            if primary:
                os.environ["GROQ_API_KEY"] = str(primary)
            if secondary:
                os.environ["GROQ_API_KEY_B"] = str(secondary)
            if tertiary:
                os.environ["GROQ_API_KEY_C"] = str(tertiary)

        if langsmith_secret_name:
            langsmith_secret = self._fetch_aws_secret_dict(region=region, secret_name=langsmith_secret_name)
            api_key = langsmith_secret.get("api_key") or langsmith_secret.get("LANGSMITH_API_KEY")
            if api_key:
                os.environ["LANGSMITH_API_KEY"] = str(api_key)

    def get(self, section: str, key: str, fallback: Any = None) -> Any:
        value = self.config.get(section, key, fallback=fallback)
        if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
            env_var = value[2:-1]
            return os.environ.get(env_var, fallback)
        return value

    def getint(self, section: str, key: str, fallback: Any = None) -> int:
        return self.config.getint(section, key, fallback=fallback)

    def getfloat(self, section: str, key: str, fallback: Any = None) -> float:
        return self.config.getfloat(section, key, fallback=fallback)

    def getboolean(self, section: str, key: str, fallback: Any = None) -> bool:
        return self.config.getboolean(section, key, fallback=fallback)

    # Path properties
    @property
    def DATA_DIR(self) -> Path:
        return self.PROJECT_ROOT / self.get('paths', 'data_dir', 'data')

    @property
    def SAMPLES_DIR(self) -> Path:
        return self.PROJECT_ROOT / self.get('paths', 'samples_dir', 'data/samples')

    @property
    def REPORTS_DIR(self) -> Path:
        return self.PROJECT_ROOT / self.get('paths', 'reports_dir', 'reports')

    @property
    def LOGS_DIR(self) -> Path:
        return self.PROJECT_ROOT / self.get('paths', 'logs_dir', 'logs')

    @property
    def CACHE_DIR(self) -> Path:
        return self.PROJECT_ROOT / self.get('paths', 'cache_dir', 'data/cache')

    # Logging
    @property
    def LOG_LEVEL(self) -> str:
        return os.environ.get('LOG_LEVEL', 'INFO')

    # Groq LLM
    @property
    def GROQ_API_KEY(self) -> str:
        return self.get('groq', 'api_key', '')

    @property
    def GROQ_API_KEY_B(self) -> str:
        return self.get('groq', 'api_key_b', '')

    @property
    def GROQ_API_KEY_C(self) -> str:
        return self.get('groq', 'api_key_c', '')

    @property
    def GROQ_MODEL(self) -> str:
        return self.get('groq', 'model', 'llama-3.1-8b-instant')

    @property
    def GROQ_MAX_TOKENS(self) -> int:
        return self.getint('groq', 'max_tokens', 800)

    @property
    def GROQ_TEMPERATURE(self) -> float:
        return self.getfloat('groq', 'temperature', 0.0)

    @property
    def GROQ_TIMEOUT(self) -> int:
        return self.getint('groq', 'timeout', 8)

    # Bedrock
    @property
    def AWS_REGION(self) -> str:
        return self.get('bedrock', 'region', 'us-east-1')

    @property
    def AWS_ACCESS_KEY_ID(self) -> str:
        return self.get('bedrock', 'access_key_id', '')

    @property
    def AWS_SECRET_ACCESS_KEY(self) -> str:
        return self.get('bedrock', 'secret_access_key', '')

    @property
    def BEDROCK_MODEL_ID(self) -> str:
        return self.get('bedrock', 'model_id', 'us.anthropic.claude-3-5-haiku-20241022-v1:0')

    @property
    def BEDROCK_MAX_TOKENS(self) -> int:
        return self.getint('bedrock', 'max_tokens', 400)

    @property
    def BEDROCK_TEMPERATURE(self) -> float:
        return self.getfloat('bedrock', 'temperature', 0.0)

    @property
    def BEDROCK_TIMEOUT(self) -> int:
        return self.getint('bedrock', 'timeout', 30)

    # HuggingFace
    @property
    def HF_API_KEY(self) -> str:
        return self.get('huggingface', 'api_key', '')

    @property
    def HF_TIMEOUT(self) -> int:
        return self.getint('huggingface', 'timeout', 30)

    @property
    def HF_MODEL(self) -> str:
        return self.get('huggingface', 'model', '')

    @property
    def HF_MAX_TOKENS(self) -> int:
        return self.getint('huggingface', 'max_tokens', 512)

    @property
    def HF_TEMPERATURE(self) -> float:
        return self.getfloat('huggingface', 'temperature', 0.0)

    # FAISS
    @property
    def FAISS_INDEX_DIR(self) -> Path:
        return self.PROJECT_ROOT / 'data' / 'faiss_indexes'

    # OCR
    @property
    def TESSERACT_CMD(self) -> str:
        return self.get('ocr', 'tesseract_cmd', 'tesseract')

    @property
    def OCR_LANGUAGES(self) -> str:
        return self.get('ocr', 'languages', 'eng')

    # Ollama
    @property
    def OLLAMA_BASE_URL(self) -> str:
        return self.get('ollama', 'base_url', 'http://localhost:11434')

    @property
    def OLLAMA_MODEL(self) -> str:
        return self.get('ollama', 'model', 'llama3.1')

    @property
    def OLLAMA_TIMEOUT(self) -> int:
        return self.getint('ollama', 'timeout', 60)

    # Document processing
    @property
    def MAX_CHUNK_SIZE(self) -> int:
        return self.getint('document', 'max_chunk_size', 3000)

    @property
    def CHUNK_OVERLAP(self) -> int:
        return self.getint('document', 'chunk_overlap', 200)

    # Retry
    @property
    def MAX_RETRIES(self) -> int:
        return self.getint('retry', 'max_retries', 3)

    @property
    def RETRY_MIN_WAIT(self) -> int:
        return self.getint('retry', 'min_wait', 2)

    @property
    def RETRY_MAX_WAIT(self) -> int:
        return self.getint('retry', 'max_wait', 10)

    @property
    def RETRY_MULTIPLIER(self) -> float:
        return self.getfloat('retry', 'multiplier', 2.0)

    # LLM Response Cache
    @property
    def LLM_CACHE_ENABLED(self) -> bool:
        return self.getboolean('llm_cache', 'enabled', True)

    @property
    def LLM_CACHE_TTL_SECONDS(self) -> int:
        return self.getint('llm_cache', 'ttl_seconds', 86400)

    @property
    def LLM_CACHE_MAX_ENTRIES(self) -> int:
        return self.getint('llm_cache', 'max_entries', 5000)

    # Stack profile (local-first with cloud-ready switches)
    @property
    def STACK_PROFILE(self) -> str:
        return self.get('stack', 'profile', 'local')

    @property
    def STACK_LLM_PROVIDER(self) -> str:
        return self.get('stack', 'llm_provider', 'local_llama')

    @property
    def STACK_OCR_PROVIDER(self) -> str:
        return self.get('stack', 'ocr_provider', 'tesseract')

    @property
    def STACK_VECTOR_PROVIDER(self) -> str:
        return self.get('stack', 'vector_provider', 'faiss')

    @property
    def STACK_STORAGE_PROVIDER(self) -> str:
        return self.get('stack', 'storage_provider', 'local_fs')

    @property
    def STACK_CACHE_PROVIDER(self) -> str:
        return self.get('stack', 'cache_provider', 'local')

    @property
    def STACK_OBSERVABILITY_PROVIDER(self) -> str:
        return self.get('stack', 'observability_provider', 'local_logs')

    @property
    def AWS_S3_BUCKET(self) -> str:
        return self.get('aws', 's3_bucket', '')

    @property
    def AWS_REDIS_ENDPOINT(self) -> str:
        return self.get('aws', 'redis_endpoint', '')

    @property
    def AWS_OPENSEARCH_ENDPOINT(self) -> str:
        return self.get('aws', 'opensearch_endpoint', '')

    @property
    def AWS_TEXTRACT_ENABLED(self) -> bool:
        return self.getboolean('aws', 'textract_enabled', False)

    @property
    def AWS_GROQ_SECRET_NAME(self) -> str:
        return self.get('aws', 'groq_secret_name', '')

    @property
    def AWS_LANGSMITH_SECRET_NAME(self) -> str:
        return self.get('aws', 'langsmith_secret_name', '')

    # LangSmith Observability
    @property
    def LANGSMITH_ENABLED(self) -> bool:
        return self.getboolean('langsmith', 'enabled', False)

    @property
    def LANGSMITH_API_KEY(self) -> str:
        return self.get('langsmith', 'api_key', '')

    @property
    def LANGSMITH_PROJECT(self) -> str:
        return self.get('langsmith', 'project', 'agentic-doc-processor')

    @property
    def LANGSMITH_ENDPOINT(self) -> str:
        return self.get('langsmith', 'endpoint', 'https://api.smith.langchain.com')

    @property
    def LANGSMITH_REQUEST_TRACES(self) -> bool:
        return self.getboolean('langsmith', 'request_traces', True)

    @property
    def BEDROCK_ONLY_MODE(self) -> bool:
        return os.environ.get('BEDROCK_ONLY_MODE', 'false').strip().lower() in {"1", "true", "yes", "on"}

    @property
    def BEDROCK_ONLY_PROVIDER(self) -> str:
        value = os.environ.get('BEDROCK_ONLY_PROVIDER', 'nova').strip().lower()
        return value if value in {"nova", "claude"} else "nova"

    @property
    def LOW_LATENCY_MODE(self) -> bool:
        return os.environ.get('LOW_LATENCY_MODE', 'false').strip().lower() in {"1", "true", "yes", "on"}

    @property
    def SKIP_HF(self) -> bool:
        env_val = os.environ.get('SKIP_HF')
        if env_val is not None:
            return env_val.strip().lower() in {"1", "true", "yes", "on"}
        return self.getboolean('runtime', 'skip_hf', False)

    @property
    def STARTUP_WARMUP_ENABLED(self) -> bool:
        env_val = os.environ.get('STARTUP_WARMUP_ENABLED')
        if env_val is not None:
            return env_val.strip().lower() in {"1", "true", "yes", "on"}
        return self.getboolean('runtime', 'startup_warmup_enabled', True)

    # Metrics
    @property
    def MIN_EXTRACTION_ACCURACY(self) -> float:
        return self.getfloat('metrics', 'min_extraction_accuracy', 0.90)

    @property
    def MIN_PII_RECALL(self) -> float:
        return self.getfloat('metrics', 'min_pii_recall', 0.95)

    @property
    def MIN_PII_PRECISION(self) -> float:
        return self.getfloat('metrics', 'min_pii_precision', 0.90)

    @property
    def MIN_WORKFLOW_SUCCESS_RATE(self) -> float:
        return self.getfloat('metrics', 'min_workflow_success_rate', 0.90)

    @property
    def MAX_P95_LATENCY_MS(self) -> float:
        return self.getfloat('metrics', 'max_p95_latency_ms', 4000.0)

    # API
    @property
    def API_HOST(self) -> str:
        return self.get('api', 'host', '0.0.0.0')

    @property
    def API_PORT(self) -> int:
        return self.getint('api', 'port', 8000)

    @property
    def API_RELOAD(self) -> bool:
        return self.getboolean('api', 'reload', False)

    # Local Llama (Transformers)
    @property
    def LOCAL_MODEL_NAME(self) -> str:
        return self.get('local_llama', 'model_name', '')

    @property
    def LLAMA_CONTEXT_LENGTH(self) -> int:
        return self.getint('local_llama', 'context_length', 512)

    @property
    def LLAMA_TEMPERATURE(self) -> float:
        return self.getfloat('local_llama', 'temperature', 0.0)

    @property
    def LLAMA_MAX_TOKENS(self) -> int:
        return self.getint('local_llama', 'max_tokens', 64)

    @property
    def USE_GPU_INFERENCE(self) -> bool:
        return self.getboolean('local_llama', 'use_gpu', False)

    @property
    def LLAMA_MODEL_PATH(self) -> str:
        return self.get('local_llama', 'model_path', '')

    @property
    def LLAMA_N_GPU_LAYERS(self) -> int:
        return self.getint('local_llama', 'n_gpu_layers', 0)


# Backwards-compatible module-level instance
try:
    settings = Env()
except Exception:
    settings = None

# Export Env class for code that instantiates it directly
__all__ = ["Env", "settings"]
