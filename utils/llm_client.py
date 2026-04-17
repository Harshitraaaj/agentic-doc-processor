"""
LLM Client with configured fallback order and Groq key support.

Fallback chain (configured per user request):
    Groq (primary) -> Bedrock Claude -> HuggingFace -> Local Llama (GGUF/Transformers)

Nova Lite is intentionally not used by default to respect user preference.
"""
import json
import time
import sqlite3
import hashlib
import threading
import re
from typing import Optional, Dict, Any, List
from enum import Enum

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from utils.config import settings
from utils.logger import logger
from utils.retry_decorator import with_retry


class LLMProvider(str, Enum):
    """LLM Provider types"""
    GROQ = "groq"
    BEDROCK_CLAUDE = "bedrock_claude"
    BEDROCK_NOVA = "bedrock_nova"   # Amazon Nova Lite: 4s, same quality as Claude Haiku
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"               # Local Ollama server (llama3.1 or any pulled model)
    LOCAL_LLAMA = "local_llama"     # Raw transformers / GGUF fallback


class LLMClient:
    """
    Unified LLM client with automatic fallback (Groq → Bedrock Claude → HF → local Llama)
    """

    def __init__(self):
        self.stack_profile = settings.STACK_PROFILE.strip().lower()
        self.stack_provider = settings.STACK_LLM_PROVIDER.strip().lower()
        self.skip_hf = settings.SKIP_HF
        self.cache_enabled = settings.LLM_CACHE_ENABLED
        self.cache_ttl_seconds = settings.LLM_CACHE_TTL_SECONDS
        self.cache_max_entries = settings.LLM_CACHE_MAX_ENTRIES
        self._cache_conn = None
        self._cache_service = None
        self._cache_lock = threading.Lock()
        self.groq_client = None
        self.groq_client_b = None   # Backup Groq client (GROQ_API_KEY_B)
        self.groq_client_c = None   # Tertiary Groq client (GROQ_API_KEY_C)
        self.bedrock_client = None
        self.huggingface_client = None
        self.ollama_client = None   # Stores base URL string when Ollama is reachable
        self.llama_client = None
        self.llama_tokenizer = None
        self.llama_device = None
        self.llama_model_name = None
        self._groq_rr_lock = threading.Lock()
        self._groq_next_key = 1
        self._groq_cooldown_until = {1: 0.0, 2: 0.0, 3: 0.0}
        self._groq_provider_cooldown_until = 0.0
        self._initialize_groq()
        self._initialize_groq_b()
        self._initialize_groq_c()

        if self.stack_profile == "cloud":
            self._initialize_bedrock()
            logger.info(
                "Cloud profile configured; using Groq + Bedrock only "
                "(HuggingFace/Ollama/Local Llama disabled)"
            )
        elif self.stack_provider != LLMProvider.GROQ.value:
            self._initialize_bedrock()
            if not self.skip_hf:
                self._initialize_huggingface()
            else:
                logger.info("SKIP_HF enabled: skipping HuggingFace client initialization")
            self._initialize_ollama()
            self._initialize_llama()
        else:
            logger.info(
                "Local profile + Groq primary configured; "
                "enabling Bedrock/HuggingFace/Ollama/Local Llama as fallbacks"
            )
            self._initialize_bedrock()
            if not self.skip_hf:
                self._initialize_huggingface()
            else:
                logger.info("SKIP_HF enabled: skipping HuggingFace client initialization")
            self._initialize_ollama()
            self._initialize_llama()

        self._initialize_cache()

    def _initialize_cache(self) -> None:
        """Initialize LLM response cache (Redis in cloud profile, SQLite in local profile)."""
        if not self.cache_enabled:
            logger.info("LLM response cache disabled")
            return

        if self.stack_profile == "cloud":
            try:
                from utils.service_registry import ServiceRegistry

                self._cache_service = ServiceRegistry.get_cache()
                logger.info(
                    "LLM response cache initialized via cache service "
                    f"({type(self._cache_service).__name__}, ttl={self.cache_ttl_seconds}s)"
                )
                return
            except Exception as e:
                logger.warning(f"Failed to initialize cloud cache service, falling back to SQLite: {e}")
                self._cache_service = None

        try:
            settings.CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_db = settings.CACHE_DIR / "llm_response_cache.db"
            self._cache_conn = sqlite3.connect(str(cache_db), check_same_thread=False)
            cur = self._cache_conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_cache (
                    cache_key TEXT PRIMARY KEY,
                    response_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    last_accessed REAL NOT NULL
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_llm_cache_last_accessed ON llm_cache(last_accessed)"
            )
            self._cache_conn.commit()
            logger.info(
                f"LLM response cache initialized at {cache_db} "
                f"(ttl={self.cache_ttl_seconds}s, max_entries={self.cache_max_entries})"
            )
        except Exception as e:
            logger.warning(f"Failed to initialize LLM cache: {e}")
            self._cache_conn = None

    def _build_cache_key(
        self,
        prompt: str,
        system_prompt: Optional[str],
        max_tokens: Optional[int],
        temperature: Optional[float],
        force_provider: Optional[LLMProvider],
        groq_model: Optional[str],
        groq_key: int,
    ) -> str:
        payload = {
            "prompt": prompt,
            "system_prompt": system_prompt or "",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "force_provider": force_provider.value if force_provider else None,
            "groq_model": groq_model,
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Best-effort 429 / rate-limit detector across provider SDK exception types."""
        text = str(error).lower()
        return (
            "429" in text
            or "rate limit" in text
            or "too many requests" in text
            or "ratelimit" in text
        )

    def _next_groq_start_key(self, available_keys: List[int]) -> int:
        """Round-robin key selector across all currently available Groq keys."""
        if not available_keys:
            return 1

        with self._groq_rr_lock:
            candidate = self._groq_next_key
            if candidate not in available_keys:
                candidate = sorted(available_keys)[0]
            self._groq_next_key = 1 if candidate >= 3 else candidate + 1
            return candidate

    def _groq_attempt_plan(self, requested_key: int = 1) -> List[tuple[str, Any, int]]:
        """Return ordered [(label, client, key_num), ...] for this request."""
        available = {
            1: self.groq_client,
            2: self.groq_client_b,
            3: self.groq_client_c,
        }
        available_keys = [k for k, c in available.items() if c is not None]
        if not available_keys:
            return []
        if len(available_keys) == 1:
            key_num = available_keys[0]
            return [(f"key-{key_num}", available[key_num], key_num)]

        # If caller explicitly asks for non-default key (2/3), honor that first.
        if requested_key in available_keys and requested_key != 1:
            ordered_keys = [requested_key] + [k for k in sorted(available_keys) if k != requested_key]
        else:
            start_key = self._next_groq_start_key(available_keys)
            sorted_keys = sorted(available_keys)
            start_index = sorted_keys.index(start_key)
            ordered_keys = sorted_keys[start_index:] + sorted_keys[:start_index]

        return [(f"key-{k}", available[k], k) for k in ordered_keys]

    def _extract_retry_after_seconds(self, error: Exception) -> Optional[float]:
        """Parse provider error text for retry-after hints like 750ms, 6.2s, 20m20.8s."""
        text = str(error).lower()

        complex_match = re.search(r"please try again in\s*(\d+)m(\d+(?:\.\d+)?)s", text)
        if complex_match:
            minutes = float(complex_match.group(1))
            seconds = float(complex_match.group(2))
            return minutes * 60.0 + seconds

        sec_match = re.search(r"please try again in\s*(\d+(?:\.\d+)?)s", text)
        if sec_match:
            return float(sec_match.group(1))

        ms_match = re.search(r"please try again in\s*(\d+(?:\.\d+)?)ms", text)
        if ms_match:
            return float(ms_match.group(1)) / 1000.0

        return None

    def _set_groq_key_cooldown(self, key_num: int, error: Exception) -> None:
        """Mark a Groq key unavailable until the provider-suggested retry time."""
        retry_after = self._extract_retry_after_seconds(error)
        if retry_after is None:
            retry_after = 2.0
        self._groq_cooldown_until[key_num] = time.time() + max(0.2, retry_after)

    def _set_groq_provider_cooldown_from_keys(self) -> None:
        """If all available keys are cooling down, set provider cooldown until earliest key recovery."""
        now = time.time()
        clients = {
            1: self.groq_client,
            2: self.groq_client_b,
            3: self.groq_client_c,
        }
        active_keys = [k for k, client in clients.items() if client is not None]
        if not active_keys:
            return

        cooldowns = [self._groq_cooldown_until.get(k, 0.0) for k in active_keys]
        if all(c > now for c in cooldowns):
            self._groq_provider_cooldown_until = min(cooldowns)

    def _invoke_groq_with_fallback(
        self,
        prompt: str,
        system_prompt: Optional[str],
        max_tokens: Optional[int],
        temperature: Optional[float],
        groq_model: Optional[str],
        requested_key: int,
        errors: List[str],
    ) -> Optional[Dict[str, Any]]:
        """Try Groq keys in order for this request; return first success."""
        plan = self._groq_attempt_plan(requested_key=requested_key)
        if not plan:
            return None

        now = time.time()
        active_plan = []
        for key_label, key_client, key_num in plan:
            cooldown_until = self._groq_cooldown_until.get(key_num, 0.0)
            if now < cooldown_until:
                wait_left = cooldown_until - now
                logger.info(f"Skipping Groq ({key_label}) due to cooldown ({wait_left:.2f}s left)")
                errors.append(f"Groq-{key_label}: cooldown {wait_left:.2f}s")
                continue
            active_plan.append((key_label, key_client, key_num))

        if not active_plan:
            self._set_groq_provider_cooldown_from_keys()
            return None

        for key_label, key_client, key_num in active_plan:
            try:
                logger.info(f"Using Groq ({key_label}) model={groq_model or settings.GROQ_MODEL}")
                return self._invoke_groq(
                    prompt,
                    system_prompt,
                    max_tokens,
                    temperature,
                    groq_model=groq_model,
                    client=key_client,
                )
            except Exception as groq_error:
                if self._is_rate_limit_error(groq_error):
                    self._set_groq_key_cooldown(key_num, groq_error)
                    self._set_groq_provider_cooldown_from_keys()
                    logger.warning(f"Groq ({key_label}) rate-limited: {groq_error}")
                else:
                    logger.warning(f"Groq ({key_label}) failed: {groq_error}")
                errors.append(f"Groq-{key_label}: {groq_error}")

        return None

    def _cache_get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        if not self.cache_enabled:
            return None

        if self._cache_service is not None:
            try:
                payload = self._cache_service.get(cache_key)
                if not payload:
                    return None
                cached = json.loads(payload)
                cached["cache_hit"] = True
                cached["latency"] = 0.0
                return cached
            except Exception as e:
                logger.warning(f"Cloud cache get failed, treating as miss: {e}")
                return None

        if self._cache_conn is None:
            return None

        now = time.time()
        with self._cache_lock:
            cur = self._cache_conn.cursor()
            cur.execute(
                "SELECT response_json, created_at FROM llm_cache WHERE cache_key = ?",
                (cache_key,),
            )
            row = cur.fetchone()
            if not row:
                return None

            response_json, created_at = row
            if (now - float(created_at)) > self.cache_ttl_seconds:
                cur.execute("DELETE FROM llm_cache WHERE cache_key = ?", (cache_key,))
                self._cache_conn.commit()
                return None

            cur.execute(
                "UPDATE llm_cache SET last_accessed = ? WHERE cache_key = ?",
                (now, cache_key),
            )
            self._cache_conn.commit()

        try:
            cached = json.loads(response_json)
            cached["cache_hit"] = True
            cached["latency"] = 0.0
            return cached
        except Exception:
            return None

    def _cache_set(self, cache_key: str, response: Dict[str, Any]) -> None:
        if not self.cache_enabled:
            return

        cache_payload = {
            "content": response.get("content", ""),
            "provider": response.get("provider"),
            "model": response.get("model"),
            "tokens": response.get("tokens", {}),
            "system_prompt": response.get("system_prompt"),
            "user_prompt": response.get("user_prompt"),
        }

        if self._cache_service is not None:
            try:
                self._cache_service.set(
                    cache_key,
                    json.dumps(cache_payload, ensure_ascii=False),
                    ttl_seconds=self.cache_ttl_seconds,
                )
                return
            except Exception as e:
                logger.warning(f"Cloud cache set failed, skipping cache write: {e}")
                return

        if self._cache_conn is None:
            return

        now = time.time()
        with self._cache_lock:
            cur = self._cache_conn.cursor()
            cur.execute(
                """
                INSERT INTO llm_cache(cache_key, response_json, created_at, last_accessed)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    response_json=excluded.response_json,
                    created_at=excluded.created_at,
                    last_accessed=excluded.last_accessed
                """,
                (cache_key, json.dumps(cache_payload, ensure_ascii=False), now, now),
            )

            cur.execute("SELECT COUNT(*) FROM llm_cache")
            count = int(cur.fetchone()[0])
            if count > self.cache_max_entries:
                to_remove = count - self.cache_max_entries
                cur.execute(
                    "DELETE FROM llm_cache WHERE cache_key IN ("
                    "SELECT cache_key FROM llm_cache ORDER BY last_accessed ASC LIMIT ?"
                    ")",
                    (to_remove,),
                )

            self._cache_conn.commit()
    
    def _initialize_groq(self) -> None:
        """Initialize Groq API client (primary key)"""
        if not settings.GROQ_API_KEY:
            logger.info("GROQ_API_KEY not configured, skipping Groq initialization")
            self.groq_client = None
            return
            
        try:
            from groq import Groq

            # Keep provider-level retries at 0 so our explicit fallback logic can
            # immediately switch key/provider on 429s.
            try:
                self.groq_client = Groq(
                    api_key=settings.GROQ_API_KEY,
                    timeout=settings.GROQ_TIMEOUT,
                    max_retries=0,
                )
            except TypeError:
                self.groq_client = Groq(
                    api_key=settings.GROQ_API_KEY,
                    timeout=settings.GROQ_TIMEOUT,
                )
            
            logger.info(f"Groq API client (primary) initialized (model: {settings.GROQ_MODEL})")
        except ImportError:
            logger.warning("groq package not installed. Install with: pip install groq")
            self.groq_client = None
        except Exception as e:
            logger.warning(f"Failed to initialize Groq client: {e}")
            self.groq_client = None

    def _initialize_groq_b(self) -> None:
        """Initialize backup Groq API client (GROQ_API_KEY_B).

        Used as a second Groq attempt when the primary key hits a rate-limit
        or request error, keeping Groq-class latency before falling back to
        the much slower Bedrock (90 s timeout).
        """
        if not getattr(settings, 'GROQ_API_KEY_B', None):
            logger.info("GROQ_API_KEY_B not configured — backup Groq key unavailable")
            self.groq_client_b = None
            return

        try:
            from groq import Groq

            try:
                self.groq_client_b = Groq(
                    api_key=settings.GROQ_API_KEY_B,
                    timeout=settings.GROQ_TIMEOUT,
                    max_retries=0,
                )
            except TypeError:
                self.groq_client_b = Groq(
                    api_key=settings.GROQ_API_KEY_B,
                    timeout=settings.GROQ_TIMEOUT,
                )
            logger.info(f"Groq API client (backup) initialized (model: {settings.GROQ_MODEL})")
        except Exception as e:
            logger.warning(f"Failed to initialize backup Groq client: {e}")
            self.groq_client_b = None

    def _initialize_groq_c(self) -> None:
        """Initialize tertiary Groq API client (GROQ_API_KEY_C)."""
        if not getattr(settings, 'GROQ_API_KEY_C', None):
            logger.info("GROQ_API_KEY_C not configured — tertiary Groq key unavailable")
            self.groq_client_c = None
            return

        try:
            from groq import Groq

            try:
                self.groq_client_c = Groq(
                    api_key=settings.GROQ_API_KEY_C,
                    timeout=settings.GROQ_TIMEOUT,
                    max_retries=0,
                )
            except TypeError:
                self.groq_client_c = Groq(
                    api_key=settings.GROQ_API_KEY_C,
                    timeout=settings.GROQ_TIMEOUT,
                )
            logger.info(f"Groq API client (tertiary) initialized (model: {settings.GROQ_MODEL})")
        except Exception as e:
            logger.warning(f"Failed to initialize tertiary Groq client: {e}")
            self.groq_client_c = None
    
    def _initialize_bedrock(self) -> None:
        """Initialize AWS Bedrock client"""
        # Skip initialization if credentials not configured
        if not settings.AWS_ACCESS_KEY_ID or not settings.AWS_SECRET_ACCESS_KEY:
            logger.info("AWS credentials not configured, skipping Bedrock initialization")
            self.bedrock_client = None
            return
            
        try:
            bedrock_timeout = 10 if settings.BEDROCK_ONLY_MODE else settings.BEDROCK_TIMEOUT
            boto_config = Config(
                region_name=settings.AWS_REGION,
                connect_timeout=bedrock_timeout,
                read_timeout=bedrock_timeout,
                retries={'max_attempts': 1 if settings.BEDROCK_ONLY_MODE else settings.MAX_RETRIES}
            )
            
            self.bedrock_client = boto3.client(
                service_name='bedrock-runtime',
                region_name=settings.AWS_REGION,
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                config=boto_config
            )
            
            logger.info("Bedrock Claude client initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize Bedrock client: {e}")
            self.bedrock_client = None
    
    def _initialize_huggingface(self) -> None:
        """Initialize HuggingFace Inference API client"""
        try:
            if settings.HF_API_KEY:
                from huggingface_hub import InferenceClient
                
                self.huggingface_client = InferenceClient(
                    token=settings.HF_API_KEY,
                    timeout=settings.HF_TIMEOUT
                )
                logger.info("HuggingFace Inference API client initialized successfully")
            else:
                logger.warning("HF_API_KEY not configured, HuggingFace fallback unavailable")
                self.huggingface_client = None
        except ImportError:
            logger.warning("huggingface_hub not installed, HuggingFace fallback unavailable")
            self.huggingface_client = None
        except Exception as e:
            logger.warning(f"Failed to initialize HuggingFace client: {type(e).__name__}: {e}", exc_info=True)
            self.huggingface_client = None
    
    def _initialize_ollama(self) -> None:
        """Check if Ollama server is running and mark available if so."""
        try:
            import requests
            base_url = getattr(settings, 'OLLAMA_BASE_URL', 'http://localhost:11434')
            resp = requests.get(f"{base_url}/api/tags", timeout=3)
            if resp.status_code == 200:
                self.ollama_client = base_url
                model = getattr(settings, 'OLLAMA_MODEL', 'llama3.1')
                logger.info(f"Ollama server reachable at {base_url}, model='{model}'")
            else:
                self.ollama_client = None
                logger.info("Ollama server returned non-200, skipping")
        except Exception:
            self.ollama_client = None
            logger.info("Ollama server not reachable (start with: ollama serve)")

    def _initialize_llama(self) -> None:
        """Initialize local LLM client using GGUF (llama-cpp-python) or Transformers."""
        # Check if GGUF model path is configured
        if hasattr(settings, 'LLAMA_MODEL_PATH') and settings.LLAMA_MODEL_PATH:
            self._initialize_llama_gguf()
        else:
            self._initialize_llama_transformers()
    
    def _initialize_llama_gguf(self) -> None:
        """Initialize local LLM using llama-cpp-python with GGUF file"""
        try:
            from llama_cpp import Llama
            import os
            
            model_path = settings.LLAMA_MODEL_PATH
            if not os.path.exists(model_path):
                logger.warning(f"GGUF model file not found: {model_path}")
                self.llama_client = None
                return
            
            n_gpu_layers = getattr(settings, 'LLAMA_N_GPU_LAYERS', 0)
            n_ctx = settings.LLAMA_CONTEXT_LENGTH
            n_batch = getattr(settings, 'LLAMA_BATCH_SIZE', 256)  # Use configured batch size
            
            logger.info(f"Initializing Llama with GGUF file: {model_path} (GPU layers: {n_gpu_layers}, ctx: {n_ctx}, batch: {n_batch})...")
            
            self.llama_client = Llama(
                model_path=model_path,
                n_gpu_layers=n_gpu_layers,
                n_ctx=n_ctx,
                n_batch=n_batch,
                verbose=False
            )
            
            self.llama_model_name = f"local-llama-gguf"
            self.llama_tokenizer = None  # GGUF uses internal tokenizer
            logger.info(f"Local LLM (GGUF) initialized successfully with {n_gpu_layers} GPU layers, context: {n_ctx}")
            
        except ImportError:
            logger.warning("llama-cpp-python not installed. Install with: pip install llama-cpp-python")
            self.llama_client = None
        except Exception as e:
            logger.warning(f"Failed to initialize GGUF LLM: {type(e).__name__}: {e}", exc_info=True)
            self.llama_client = None
    
    def _initialize_llama_transformers(self) -> None:
        """Initialize local LLM client using Transformers with GPU support"""
        # Skip if no model name configured (GGUF will be used instead)
        if not settings.LOCAL_MODEL_NAME:
            logger.info("LOCAL_MODEL_NAME not configured, skipping Transformers initialization")
            self.llama_client = None
            return
        
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from huggingface_hub import login
            
            # Authenticate with HuggingFace if API key is available (for gated models)
            if settings.HF_API_KEY:
                try:
                    login(token=settings.HF_API_KEY)
                    logger.info("HuggingFace authentication successful")
                except Exception as e:
                    logger.warning(f"HuggingFace login failed: {e}")
            
            # Check if GPU is available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            gpu_info = f"GPU ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "CPU"
            
            # Use model from config
            model_name = settings.LOCAL_MODEL_NAME
            
            logger.info(f"Initializing local LLM '{model_name}' on {gpu_info}...")
            
            # Load tokenizer (with token for gated models)
            self.llama_tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                token=settings.HF_API_KEY if settings.HF_API_KEY else None
            )
            
            # Set pad token if not exists
            if self.llama_tokenizer.pad_token is None:
                self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
            
            # Configure for efficient inference
            if torch.cuda.is_available() and settings.USE_GPU_INFERENCE:
                try:
                    # Try 4-bit quantization for GPU (saves memory)
                    from transformers import BitsAndBytesConfig
                    
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                    
                    self.llama_client = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        quantization_config=quantization_config,
                        device_map="auto",
                        trust_remote_code=True,
                        torch_dtype=torch.float16,
                        token=settings.HF_API_KEY if settings.HF_API_KEY else None
                    )
                    logger.info("Loaded model with 4-bit quantization")
                except Exception as e:
                    logger.warning(f"4-bit quantization failed, loading in FP16: {e}")
                    # Fallback to FP16 without quantization
                    self.llama_client = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        device_map="auto",
                        trust_remote_code=True,
                        torch_dtype=torch.float16,
                        token=settings.HF_API_KEY if settings.HF_API_KEY else None
                    )
            else:
                # CPU inference with FP32
                logger.info("Loading model for CPU inference...")
                self.llama_client = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="cpu",
                    trust_remote_code=True,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    token=settings.HF_API_KEY if settings.HF_API_KEY else None
                )
            
            self.llama_device = device
            self.llama_model_name = model_name
            logger.info(f"Local LLM initialized successfully on {gpu_info}")
            
        except ImportError as e:
            logger.warning(f"Required packages not installed for local LLM: {e}")
            self.llama_client = None
            self.llama_tokenizer = None
        except Exception as e:
            logger.warning(f"Failed to initialize LLM client: {type(e).__name__}: {e}", exc_info=True)
            self.llama_client = None
            self.llama_tokenizer = None
    
    @with_retry()
    def _invoke_groq(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = None,
        temperature: float = None,
        groq_model: Optional[str] = None,
        client=None   # Pass specific Groq client instance (key 1 or key 2)
    ) -> Dict[str, Any]:
        """
        Invoke Groq API with retry logic
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            client: Override which Groq client (key) to use
        
        Returns:
            Response dict
        """
        groq_client = client or self.groq_client
        if not groq_client:
            raise RuntimeError("Groq client not initialized")
        
        max_tokens = max_tokens or settings.GROQ_MAX_TOKENS
        temperature = temperature if temperature is not None else settings.GROQ_TEMPERATURE
        model = groq_model or settings.GROQ_MODEL
        
        start_time = time.time()
        
        try:
            messages = []
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            messages.append({
                "role": "user",
                "content": prompt
            })
            
            response = groq_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False
            )
            
            content = response.choices[0].message.content
            latency = time.time() - start_time
            
            # Extract token counts (match structure with other providers)
            tokens_in = 0
            tokens_out = 0
            if hasattr(response, 'usage') and response.usage:
                tokens_in = response.usage.prompt_tokens or 0
                tokens_out = response.usage.completion_tokens or 0
            
            return {
                "content": content,
                "provider": "groq",
                "model": model,
                "latency": latency,
                "tokens": {
                    "input": tokens_in,
                    "output": tokens_out
                },
                "system_prompt": system_prompt,
                "user_prompt": prompt
            }
            
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            raise
    
    @with_retry(max_attempts=1, min_wait=1, max_wait=1, multiplier=1)
    def _invoke_bedrock_claude(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = None,
        temperature: float = None
    ) -> Dict[str, Any]:
        """
        Invoke Bedrock Claude with retry logic
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        
        Returns:
            Response dict with content and metadata
        
        Raises:
            Exception if Bedrock invocation fails
        """
        if not self.bedrock_client:
            raise RuntimeError("Bedrock client not initialized")
        
        max_tokens = max_tokens or settings.BEDROCK_MAX_TOKENS
        if settings.BEDROCK_ONLY_MODE:
            max_tokens = min(max_tokens, 450)
        temperature = temperature if temperature is not None else settings.BEDROCK_TEMPERATURE
        
        # Construct Claude 3 message format
        messages = [{"role": "user", "content": prompt}]
        
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages
        }
        
        if system_prompt:
            request_body["system"] = system_prompt
        
        start_time = time.time()
        
        try:
            response = self.bedrock_client.invoke_model(
                modelId=settings.BEDROCK_MODEL_ID,  # Configurable via .env / config.py
                body=json.dumps(request_body)
            )
            
            response_body = json.loads(response['body'].read())
            latency = time.time() - start_time
            
            content = response_body['content'][0]['text']
            
            return {
                "content": content,
                "provider": LLMProvider.BEDROCK_CLAUDE,
                "model": settings.BEDROCK_MODEL_ID,
                "latency": latency,
                "tokens": {
                    "input": response_body.get('usage', {}).get('input_tokens', 0),
                    "output": response_body.get('usage', {}).get('output_tokens', 0)
                },
                "system_prompt": system_prompt,
                "user_prompt": prompt
            }
        
        except ClientError as e:
            error_code = e.response['Error']['Code']
            logger.error(f"Bedrock API error: {error_code}", error=str(e))
            raise
        except Exception as e:
            logger.error(f"Unexpected Bedrock error: {e}")
            raise

    NOVA_LITE_MODEL_ID = "us.amazon.nova-lite-v1:0"

    @with_retry(max_attempts=1, min_wait=1, max_wait=1, multiplier=1)
    def _invoke_nova(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = None,
        temperature: float = None
    ) -> Dict[str, Any]:
        """
        Invoke Amazon Nova Lite via Bedrock — 2x faster than Claude Haiku, same quality.
        Uses Converse-style message format.
        """
        if not self.bedrock_client:
            raise RuntimeError("Bedrock client not initialized")

        max_tokens = max_tokens or settings.BEDROCK_MAX_TOKENS
        # Hard cap for low latency under Bedrock-only mode.
        if settings.BEDROCK_ONLY_MODE:
            max_tokens = min(max_tokens, 700)
        temperature = temperature if temperature is not None else settings.BEDROCK_TEMPERATURE

        request_body: Dict[str, Any] = {
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {"maxTokens": max_tokens, "temperature": temperature}
        }
        if system_prompt:
            request_body["system"] = [{"text": system_prompt}]

        start_time = time.time()
        try:
            response = self.bedrock_client.invoke_model(
                modelId=self.NOVA_LITE_MODEL_ID,
                body=json.dumps(request_body)
            )
            response_body = json.loads(response["body"].read())
            latency = time.time() - start_time

            content_list = response_body.get("output", {}).get("message", {}).get("content", [])
            content = content_list[0].get("text", "") if content_list else ""
            usage = response_body.get("usage", {})

            return {
                "content": content,
                "provider": LLMProvider.BEDROCK_NOVA,
                "model": self.NOVA_LITE_MODEL_ID,
                "latency": latency,
                "tokens": {
                    "input": usage.get("inputTokens", 0),
                    "output": usage.get("outputTokens", 0)
                },
                "system_prompt": system_prompt,
                "user_prompt": prompt
            }
        except Exception as e:
            logger.error(f"Nova Lite error: {e}")
            raise

    def _invoke_llama(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = None,
        temperature: float = None
    ) -> Dict[str, Any]:
        """
        Invoke local LLM (routes to GGUF or Transformers based on model type)
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        
        Returns:
            Response dict with content and metadata
        
        Raises:
            RuntimeError if LLM client not initialized
        """
        if not self.llama_client:
            raise RuntimeError("Local LLM client not initialized")
        
        # Check if using GGUF (llama-cpp-python) or Transformers
        if hasattr(self.llama_client, '__call__'):  # GGUF Llama object
            return self._invoke_llama_gguf(prompt, system_prompt, max_tokens, temperature)
        else:  # Transformers model
            return self._invoke_llama_transformers(prompt, system_prompt, max_tokens, temperature)
    
    def _invoke_llama_gguf(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = None,
        temperature: float = None
    ) -> Dict[str, Any]:
        """Invoke GGUF model using llama-cpp-python - SPEED OPTIMIZED"""
        max_tokens = max_tokens or getattr(settings, 'LLAMA_MAX_TOKENS', 64)
        temperature = temperature if temperature is not None else settings.LLAMA_TEMPERATURE
        
        # Balanced truncation for speed while preserving content
        prompt_max = 800  # Enough for document excerpt (was 200, too aggressive)
        if len(prompt) > prompt_max:
            # Keep beginning and end for context
            half = prompt_max // 2
            prompt = prompt[:half] + "\n...[truncated]...\n" + prompt[-half:]
        
        # Build Llama 3.1 prompt format
        if system_prompt:
            system_max = 100  # Increased from 50 for clarity
            if len(system_prompt) > system_max:
                system_prompt = system_prompt[:system_max]
            full_prompt = f"<|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        else:
            full_prompt = f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        
        start_time = time.time()
        
        try:
            response = self.llama_client(
                full_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["<|eot_id|>", "<|end_of_text|>"],
                echo=False
            )
            
            latency = time.time() - start_time
            content = response['choices'][0]['text'].strip()
            
            return {
                "content": content,
                "provider": LLMProvider.LOCAL_LLAMA,
                "model": self.llama_model_name,
                "latency": latency,
                "tokens": {
                    "input": response['usage']['prompt_tokens'],
                    "output": response['usage']['completion_tokens']
                },
                "system_prompt": system_prompt,
                "user_prompt": prompt
            }
        except Exception as e:
            logger.error(f"GGUF LLM invocation error: {e}", exc_info=True)
            raise
    
    def _invoke_llama_transformers(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = None,
        temperature: float = None
    ) -> Dict[str, Any]:
        """
        Invoke local LLM using Transformers
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        
        Returns:
            Response dict with content and metadata
        
        Raises:
            RuntimeError if LLM client not initialized
        """
        if not self.llama_client or not self.llama_tokenizer:
            raise RuntimeError("Local LLM client not initialized")
        
        import torch
        
        # Aggressively optimize for speed
        max_tokens = max_tokens or getattr(settings, 'LLAMA_MAX_TOKENS', 128)
        temperature = temperature if temperature is not None else settings.LLAMA_TEMPERATURE
        
        # Aggressively truncate for much faster processing
        prompt_max = 400  # Reduced from 800
        if len(prompt) > prompt_max:
            prompt = prompt[:prompt_max] + "..."
        
        # Minimal prompt format for speed
        if system_prompt:
            # Very short system prompt
            system_max = 100  # Reduced from 200
            if len(system_prompt) > system_max:
                system_prompt = system_prompt[:system_max]
            input_text = f"{system_prompt}\n\n{prompt}"
        else:
            input_text = prompt
        
        start_time = time.time()
        
        try:
            # Tokenize
            inputs = self.llama_tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=settings.LLAMA_CONTEXT_LENGTH
            )
            
            # Move to device
            device = next(self.llama_client.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate with maximum speed optimizations
            with torch.no_grad():
                outputs = self.llama_client.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,  # Greedy decoding
                    num_beams=1,  # No beam search
                    pad_token_id=self.llama_tokenizer.pad_token_id,
                    eos_token_id=self.llama_tokenizer.eos_token_id,
                    use_cache=True,  # KV cache
                    early_stopping=True,  # Stop as soon as possible
                    repetition_penalty=1.0  # No penalty computation
                )
            
            # Decode only the generated tokens (skip input)
            generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            content = self.llama_tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            latency = time.time() - start_time
            
            return {
                "content": content.strip(),
                "provider": LLMProvider.LOCAL_LLAMA,
                "model": getattr(self, 'llama_model_name', 'local-llm'),
                "latency": latency,
                "tokens": {
                    "input": inputs['input_ids'].shape[1],
                    "output": len(generated_tokens)
                },
                "system_prompt": system_prompt,
                "user_prompt": prompt
            }
        
        except Exception as e:
            logger.error(f"Local LLM invocation error: {e}", exc_info=True)
            raise
    
    def _invoke_huggingface(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = None,
        temperature: float = None
    ) -> Dict[str, Any]:
        """
        Invoke HuggingFace Inference API
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        
        Returns:
            Response dict with content and metadata
        
        Raises:
            RuntimeError if HuggingFace client not initialized
        """
        if not self.huggingface_client:
            raise RuntimeError("HuggingFace client not initialized")
        
        max_tokens = max_tokens or settings.HF_MAX_TOKENS
        temperature = temperature if temperature is not None else settings.HF_TEMPERATURE
        
        # Format messages for HuggingFace chat completion
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        start_time = time.time()
        
        try:
            # Use chat completion API for instruction-tuned models
            response = self.huggingface_client.chat_completion(
                messages=messages,
                model=settings.HF_MODEL,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            latency = time.time() - start_time
            
            content = response.choices[0].message.content
            
            # Extract token counts if available
            tokens_in = 0
            tokens_out = 0
            if hasattr(response, 'usage') and response.usage:
                tokens_in = response.usage.prompt_tokens or 0
                tokens_out = response.usage.completion_tokens or 0
            
            return {
                "content": content,
                "provider": LLMProvider.HUGGINGFACE,
                "model": settings.HF_MODEL,
                "latency": latency,
                "tokens": {
                    "input": tokens_in,
                    "output": tokens_out
                },
                "system_prompt": system_prompt,
                "user_prompt": prompt
            }
        
        except Exception as e:
            logger.error(f"HuggingFace invocation error: {e}")
            raise
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = None,
        temperature: float = None,
        force_provider: Optional[LLMProvider] = None,
        groq_model: Optional[str] = None,
        groq_key: int = 1  # 1 = primary key (classifier/validator), 2 = secondary key (extractor/repair/redactor)
    ) -> Dict[str, Any]:
        """
        Generate completion with automatic fallback
        
        Fallback chain: Groq → Bedrock → HuggingFace → Llama (local)
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            force_provider: Force specific provider (skip fallback)
        
        Returns:
            Response dict with content and metadata
        
        Raises:
            RuntimeError if all providers fail
        """
        effective_force_provider = force_provider
        stack_provider = settings.STACK_LLM_PROVIDER.strip().lower()

        cloud_profile = settings.STACK_PROFILE.strip().lower() == "cloud"

        if settings.BEDROCK_ONLY_MODE and force_provider is None and stack_provider != LLMProvider.GROQ.value:
            if settings.BEDROCK_ONLY_PROVIDER == "claude":
                effective_force_provider = LLMProvider.BEDROCK_CLAUDE
            else:
                effective_force_provider = LLMProvider.BEDROCK_NOVA

        cache_key = self._build_cache_key(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            force_provider=effective_force_provider,
            groq_model=groq_model,
            groq_key=groq_key,
        )
        cached_response = self._cache_get(cache_key)
        if cached_response is not None:
            logger.info("LLM cache hit")
            return cached_response

        # Try forced provider if specified
        if effective_force_provider == LLMProvider.GROQ:
            logger.info("Using forced provider: Groq")
            response = self._invoke_groq_with_fallback(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                groq_model=groq_model,
                requested_key=groq_key,
                errors=[],
            )
            if response is None:
                raise RuntimeError("Forced Groq provider failed on all configured keys")
            self._cache_set(cache_key, response)
            return response
        elif effective_force_provider == LLMProvider.BEDROCK_NOVA:
            logger.info("Using forced provider: Nova Lite")
            try:
                response = self._invoke_nova(prompt, system_prompt, max_tokens, temperature)
                self._cache_set(cache_key, response)
                return response
            except Exception as nova_error:
                if settings.BEDROCK_ONLY_MODE and self.bedrock_client:
                    logger.warning(f"Nova Lite failed in bedrock-only mode, trying Claude: {nova_error}")
                    response = self._invoke_bedrock_claude(prompt, system_prompt, max_tokens, temperature)
                    self._cache_set(cache_key, response)
                    return response
                raise
        elif effective_force_provider == LLMProvider.BEDROCK_CLAUDE:
            logger.info("Using forced provider: Bedrock Claude")
            try:
                response = self._invoke_bedrock_claude(prompt, system_prompt, max_tokens, temperature)
                self._cache_set(cache_key, response)
                return response
            except Exception as claude_error:
                if settings.BEDROCK_ONLY_MODE and self.bedrock_client:
                    logger.warning(f"Bedrock Claude failed in bedrock-only mode, trying Nova Lite: {claude_error}")
                    response = self._invoke_nova(prompt, system_prompt, max_tokens, temperature)
                    self._cache_set(cache_key, response)
                    return response
                raise
        elif effective_force_provider == LLMProvider.HUGGINGFACE:
            if self.skip_hf:
                raise RuntimeError("HuggingFace provider is disabled (SKIP_HF=true)")
            logger.info("Using forced provider: HuggingFace")
            response = self._invoke_huggingface(prompt, system_prompt, max_tokens, temperature)
            self._cache_set(cache_key, response)
            return response
        elif effective_force_provider == LLMProvider.LOCAL_LLAMA:
            logger.info("Using forced provider: Local Llama")
            response = self._invoke_llama(prompt, system_prompt, max_tokens, temperature)
            self._cache_set(cache_key, response)
            return response
        
        errors = []
        now = time.time()
        groq_provider_in_cooldown = now < self._groq_provider_cooldown_until
        if groq_provider_in_cooldown:
            wait_left = self._groq_provider_cooldown_until - now
            logger.info(f"Skipping Groq provider due to cooldown ({wait_left:.2f}s left)")
            errors.append(f"Groq-provider: cooldown {wait_left:.2f}s")

        # ROUTING STRATEGY:
        # - If a groq_model is provided and a Groq client is available, prefer Groq (fast)
        if (not groq_provider_in_cooldown) and groq_model and (self.groq_client or self.groq_client_b or self.groq_client_c):
            response = self._invoke_groq_with_fallback(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                groq_model=groq_model,
                requested_key=groq_key,
                errors=errors,
            )
            if response is not None:
                self._cache_set(cache_key, response)
                return response

        prefer_nova = (max_tokens or 0) >= 1000

        # Bedrock fallback (Nova first for large generations to reduce latency)
        if self.bedrock_client:
            bedrock_attempts = [
                ("Nova Lite", self._invoke_nova),
                ("Bedrock Claude", self._invoke_bedrock_claude),
            ] if prefer_nova else [
                ("Bedrock Claude", self._invoke_bedrock_claude),
                ("Nova Lite", self._invoke_nova),
            ]

            for bedrock_label, bedrock_call in bedrock_attempts:
                try:
                    logger.info(f"Using {bedrock_label} fallback")
                    response = bedrock_call(prompt, system_prompt, max_tokens, temperature)
                    self._cache_set(cache_key, response)
                    return response
                except Exception as bedrock_error:
                    logger.warning(f"{bedrock_label} failed: {bedrock_error}")
                    errors.append(f"{bedrock_label}: {bedrock_error}")
        else:
            logger.debug("Bedrock client not available, skipping")
        
        if cloud_profile:
            logger.error("Cloud profile provider set exhausted (Groq + Bedrock only)")
            raise RuntimeError(f"Cloud providers failed (Groq/Bedrock): {'; '.join(errors)}")

        # Fallback to HuggingFace
        if self.skip_hf:
            logger.info("SKIP_HF enabled: skipping HuggingFace fallback")
        elif self.huggingface_client:
            try:
                logger.info("Falling back to HuggingFace Inference API")
                response = self._invoke_huggingface(prompt, system_prompt, max_tokens, temperature)
                self._cache_set(cache_key, response)
                return response
            
            except Exception as hf_error:
                logger.warning(f"HuggingFace fallback failed: {hf_error}", exc_info=True)
                errors.append(f"HuggingFace: {hf_error}")
        else:
            logger.info("HuggingFace client not available, skipping")
        
        # Fallback to local Llama (GGUF / Transformers)
        if self.llama_client:
            try:
                logger.info("Falling back to local Llama (GGUF/Transformers)")
                response = self._invoke_llama(prompt, system_prompt, max_tokens, temperature)
                self._cache_set(cache_key, response)
                return response
            except Exception as llama_error:
                logger.error(f"Local Llama fallback also failed: {llama_error}")
                errors.append(f"LocalLlama: {llama_error}")

        # If we reach here, all providers failed
        logger.error("All LLM providers exhausted")
        raise RuntimeError(f"All configured LLM providers failed: {'; '.join(errors)}")
    
    def is_available(self, provider: Optional[LLMProvider] = None) -> bool:
        """
        Check if LLM provider is available
        
        Args:
            provider: Specific provider to check, or None for any
        
        Returns:
            True if provider is available
        """
        if provider == LLMProvider.GROQ:
            return (
                self.groq_client is not None
                or self.groq_client_b is not None
                or self.groq_client_c is not None
            )
        elif provider == LLMProvider.BEDROCK_CLAUDE:
            return self.bedrock_client is not None
        elif provider == LLMProvider.HUGGINGFACE:
            return self.huggingface_client is not None
        elif provider == LLMProvider.OLLAMA:
            return self.ollama_client is not None
        elif provider == LLMProvider.LOCAL_LLAMA:
            return self.llama_client is not None
        else:
            return (
                self.groq_client is not None
                or self.groq_client_b is not None
                or self.groq_client_c is not None
                or
                self.bedrock_client is not None
                or self.huggingface_client is not None
                or self.ollama_client is not None
                or self.llama_client is not None
            )
    
# Global LLM client instance
llm_client = LLMClient()
