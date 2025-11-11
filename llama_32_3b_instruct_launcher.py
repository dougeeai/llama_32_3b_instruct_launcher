# %% [0.0] Launcher Script Info
# Script metadata and documentation for version tracking
# Llama 3.2 3B Instruct Transformers Launcher
# Description: Optimized launcher for Llama-3.2-3B-Instruct safetensors models
# Author: dougeeai
# Created: 2025-11-09
# Last Updated: 2025-11-11
# Optimized for: Python 3.13 + CUDA 13.0 + Transformers

# %% [0.1] Model Card & Summary
# Quick reference for model capabilities and requirements
# MODEL: Llama-3.2-3B-Instruct
# Architecture: Llama 3.2 (3.21B parameters)
# Format: Safetensors (native HuggingFace format)
# Context: 128K max
# Best For: Instruction following, chat, code assistance
# GPU Memory: Varies by precision (FP16: ~6-7GB, 8-bit: ~3-4GB, 4-bit: ~2-3GB)

# %% [1.0] Core Imports
# Essential Python libraries required for Transformers operation
import os
import sys
import json
import warnings
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass, field
import torch
import torch.nn as nn

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Transformers imports
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextStreamer,
    GenerationConfig,
    StoppingCriteria,
    StoppingCriteriaList
)

# %% [1.1] Utility Imports
# Supporting libraries for hardware monitoring and performance metrics
import time
import psutil
import platform
from datetime import datetime
import gc

# GPU monitoring
try:
    import pynvml
    pynvml.nvmlInit()
    NVIDIA_GPU_AVAILABLE = True
except:
    NVIDIA_GPU_AVAILABLE = False

# Optional: bitsandbytes for quantization
try:
    import bitsandbytes as bnb
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False
    print("INFO: bitsandbytes not installed. Quantization options disabled.")

# Optional: flash attention for performance
try:
    import flash_attn
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    print("INFO: Flash Attention not available (optional performance enhancement)")

# %% [2.0] Base Directory Configuration
# Set base AI directory - all paths will be relative to this location

# Set your base directory (change this for your system)
BASE_DIR = r"E:\ai"  # <-- CHANGE THIS to your AI folder location

# Directory structure (automatically created from BASE_DIR)
MODELS_DIR = os.path.join(BASE_DIR, "models")
HF_DOWNLOADS_DIR = os.path.join(MODELS_DIR, "huggingface_downloads")  # For HF downloads
HF_CACHE_DIR = os.path.join(HF_DOWNLOADS_DIR, "cache")  # HF cache

# Create directories if they don't exist
for dir_path in [MODELS_DIR, HF_DOWNLOADS_DIR, HF_CACHE_DIR]:
    Path(dir_path).mkdir(parents=True, exist_ok=True)

# %% [2.1] Model Source Configuration
# Configure where to load the model from - local folder or HuggingFace download

# Choose model source: "local" or "huggingface"
MODEL_SOURCE = "local"  # Options: "local" or "huggingface"

# Model identification
MODEL_NAME = "llama_32_3b_instruct"  # Folder name for this model

# Local model configuration (for manually downloaded models)
# Local models go directly in: BASE_DIR/models/MODEL_NAME/
LOCAL_MODEL_PATH = os.path.join(MODELS_DIR, MODEL_NAME)

# HuggingFace configuration
HF_REPO_ID = "meta-llama/Llama-3.2-3B-Instruct"
# HuggingFace models will be saved to: models/huggingface_downloads/MODEL_NAME/
HF_MODEL_DIR = os.path.join(HF_DOWNLOADS_DIR, MODEL_NAME)

# Resolve actual model path based on source
if MODEL_SOURCE == "huggingface":
    try:
        from huggingface_hub import snapshot_download
        print(f"Downloading model from HuggingFace: {HF_REPO_ID}")
        print(f"Download location: {HF_MODEL_DIR}")
        print(f"Cache location: {HF_CACHE_DIR}")
        
        MODEL_PATH = snapshot_download(
            repo_id=HF_REPO_ID,
            cache_dir=HF_CACHE_DIR,
            local_dir=HF_MODEL_DIR,
            local_dir_use_symlinks=False
        )
        print(f"Model downloaded to: {MODEL_PATH}")
    except ImportError:
        print("ERROR: huggingface-hub not installed. Run: pip install huggingface-hub")
        print("Falling back to local path...")
        MODEL_PATH = LOCAL_MODEL_PATH
    except Exception as e:
        print(f"ERROR downloading from HuggingFace: {e}")
        print("Falling back to local path...")
        MODEL_PATH = LOCAL_MODEL_PATH
else:
    MODEL_PATH = LOCAL_MODEL_PATH
    if not Path(MODEL_PATH).exists():
        print(f"WARNING: Local model not found at: {MODEL_PATH}")
        print(f"Expected location: {LOCAL_MODEL_PATH}")
        print(f"To download from HuggingFace, set MODEL_SOURCE = 'huggingface'")

# %% [2.2] User Configuration - All Settings
# Central location for all user-modifiable model and generation settings

# Precision Configuration (Choose one)
# Options: "fp16" (full precision), "8bit", "4bit"
PRECISION_MODE = "fp16"  # Default to full precision

# Hardware Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_MAP = "auto"      # Auto device mapping for model parallelism
MAX_MEMORY = None        # Set to {"0": "20GB"} to limit GPU memory

# Loading Configuration
LOW_CPU_MEM_USAGE = True  # Reduce CPU memory during loading
TRUST_REMOTE_CODE = False # Whether to trust remote code (not needed for Llama)

# Performance Configuration
USE_FLASH_ATTENTION = True      # Use Flash Attention 2 if available (set False to disable)
USE_BETTER_TRANSFORMER = False  # Use BetterTransformer optimization
TORCH_COMPILE = False           # Use torch.compile() (requires PyTorch 2.0+)

# Quantization Settings (used when PRECISION_MODE is "8bit" or "4bit")
LOAD_IN_8BIT = False      # Set automatically based on PRECISION_MODE
LOAD_IN_4BIT = False      # Set automatically based on PRECISION_MODE
BNB_4BIT_COMPUTE_DTYPE = "float16"  # Compute dtype for 4-bit
BNB_4BIT_QUANT_TYPE = "nf4"         # Quantization type: "nf4" or "fp4"
BNB_4BIT_USE_DOUBLE_QUANT = True    # Double quantization for 4-bit

# Generation Configuration
TEMPERATURE = 0.7        # Randomness (0.0 = deterministic, 2.0 = very random)
TOP_P = 0.9             # Nucleus sampling threshold
TOP_K = 40              # Top-k sampling
REPETITION_PENALTY = 1.1 # Penalty for repetition
MAX_NEW_TOKENS = 2048   # Maximum tokens to generate
DO_SAMPLE = True        # Use sampling (False = greedy decoding)

# Memory Configuration
CLEAR_CACHE_FREQ = 5    # Clear GPU cache every N generations
USE_CACHE = True        # Use KV cache during generation

# Chat Configuration
SYSTEM_MESSAGE = "You are a helpful AI assistant."
ADD_GENERATION_PROMPT = True  # Add the generation prompt for chat

# Debug Configuration
VERBOSE = False          # Show detailed loading info

# Generation Presets (alternative to manual settings above)
GENERATION_PRESETS = {
    "precise": {
        "temperature": 0.1,
        "top_p": 0.95,
        "top_k": 40,
        "repetition_penalty": 1.1,
        "do_sample": True
    },
    "balanced": {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "repetition_penalty": 1.1,
        "do_sample": True
    },
    "creative": {
        "temperature": 1.2,
        "top_p": 0.95,
        "top_k": 100,
        "repetition_penalty": 1.0,
        "do_sample": True
    },
    "deterministic": {
        "temperature": 0.0,
        "do_sample": False,
        "repetition_penalty": 1.1
    }
}

# Select a preset (None = use manual settings above)
USE_PRESET = None  # Options: None, "precise", "balanced", "creative", "deterministic"

# %% [2.3] Model Configuration Dataclass
# Structured container for passing configuration to model loader

@dataclass
class ModelConfig:
    """Configuration container - populated from user settings above"""
    # Model paths
    model_path: str
    
    # Precision settings
    precision_mode: str = PRECISION_MODE
    device: str = DEVICE
    device_map: str = DEVICE_MAP
    max_memory: Optional[Dict] = MAX_MEMORY
    
    # Loading settings
    low_cpu_mem_usage: bool = LOW_CPU_MEM_USAGE
    trust_remote_code: bool = TRUST_REMOTE_CODE
    use_flash_attention: bool = USE_FLASH_ATTENTION
    
    # Quantization settings
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    bnb_4bit_compute_dtype: str = BNB_4BIT_COMPUTE_DTYPE
    bnb_4bit_quant_type: str = BNB_4BIT_QUANT_TYPE
    bnb_4bit_use_double_quant: bool = BNB_4BIT_USE_DOUBLE_QUANT
    
    # Generation settings
    temperature: float = TEMPERATURE
    top_p: float = TOP_P
    top_k: int = TOP_K
    repetition_penalty: float = REPETITION_PENALTY
    max_new_tokens: int = MAX_NEW_TOKENS
    do_sample: bool = DO_SAMPLE
    use_cache: bool = USE_CACHE
    
    # Chat settings
    system_message: str = SYSTEM_MESSAGE
    add_generation_prompt: bool = ADD_GENERATION_PROMPT
    
    # Performance settings
    use_better_transformer: bool = USE_BETTER_TRANSFORMER
    torch_compile: bool = TORCH_COMPILE
    clear_cache_freq: int = CLEAR_CACHE_FREQ
    
    # Debug settings
    verbose: bool = VERBOSE
    
    def __post_init__(self):
        """Apply precision mode settings"""
        if self.precision_mode == "8bit":
            self.load_in_8bit = True
            self.load_in_4bit = False
        elif self.precision_mode == "4bit":
            self.load_in_8bit = False
            self.load_in_4bit = True
        else:  # fp16
            self.load_in_8bit = False
            self.load_in_4bit = False

# %% [3.0] Hardware Auto-Detection
# Automatically determine optimal settings based on available hardware

def get_optimal_settings() -> Dict[str, Any]:
    """Auto-configure based on available hardware"""
    settings = {}
    
    # CPU settings
    cpu_count = psutil.cpu_count()
    settings['torch_threads'] = min(8, cpu_count // 2)
    torch.set_num_threads(settings['torch_threads'])
    
    # Memory settings
    ram_gb = psutil.virtual_memory().total / (1024**3)
    settings['low_cpu_mem_usage'] = ram_gb < 32
    
    # GPU settings
    if torch.cuda.is_available():
        settings['device'] = 'cuda'
        settings['device_map'] = 'auto'
        
        if NVIDIA_GPU_AVAILABLE:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            vram_bytes = pynvml.nvmlDeviceGetMemoryInfo(handle).total
            vram_gb = vram_bytes / (1024**3)
            
            # Precision recommendations for 3B model
            if vram_gb < 6:
                settings['precision_mode'] = '4bit'
                print("INFO: Low VRAM detected, recommending 4-bit quantization")
            elif vram_gb < 12:
                settings['precision_mode'] = '8bit'
                print("INFO: Medium VRAM detected, recommending 8-bit quantization")
            else:
                settings['precision_mode'] = 'fp16'
                print("INFO: Sufficient VRAM for full precision")
    else:
        settings['device'] = 'cpu'
        settings['precision_mode'] = '8bit'  # CPU benefits from quantization
        print("INFO: No GPU detected, using CPU with quantization")
    
    return settings

# %% [3.1] Hardware Detection
# Gather detailed hardware information for optimization decisions

def detect_hardware() -> Dict[str, Any]:
    """Detect available hardware capabilities"""
    info = {
        "platform": platform.system(),
        "cpu_name": platform.processor(),
        "cpu_count": psutil.cpu_count(),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 1),
        "ram_available_gb": round(psutil.virtual_memory().available / (1024**3), 1),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_available": False,
        "gpu_name": None,
        "vram_total_gb": 0,
        "vram_available_gb": 0
    }
    
    if torch.cuda.is_available():
        info["gpu_available"] = True
        info["gpu_name"] = torch.cuda.get_device_name(0)
        
        if NVIDIA_GPU_AVAILABLE:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                vram_total = mem_info.total
                vram_free = mem_info.free
                
                info["vram_total_gb"] = round(vram_total / (1024**3), 1)
                info["vram_available_gb"] = round(vram_free / (1024**3), 1)
            except Exception as e:
                print(f"DEBUG: Error getting detailed GPU info: {e}")
    
    return info

# %% [3.2] Environment Validation
# Verify Python version and required packages before proceeding

def validate_environment() -> bool:
    """Validate Python and package environment"""
    valid = True
    
    # Check Python version
    py_version = sys.version_info
    if py_version.major < 3 or py_version.minor < 10:
        print(f"WARNING: Python {py_version.major}.{py_version.minor} detected. Python 3.10+ recommended.")
    
    # Check PyTorch
    print(f"OK: PyTorch version {torch.__version__}")
    if torch.cuda.is_available():
        print(f"OK: CUDA {torch.version.cuda} available")
    else:
        print("INFO: CUDA not available, will use CPU")
    
    # Check transformers
    try:
        import transformers
        print(f"OK: Transformers version {transformers.__version__}")
    except ImportError:
        print("ERROR: transformers not installed")
        valid = False
    
    # Check bitsandbytes (optional)
    if BITSANDBYTES_AVAILABLE:
        print("OK: bitsandbytes available for quantization")
    else:
        print("INFO: bitsandbytes not available (install for quantization support)")
    
    # Check flash attention (optional)
    if FLASH_ATTN_AVAILABLE:
        print(f"OK: Flash Attention 2 available")
        if USE_FLASH_ATTENTION:
            print("INFO: Flash Attention 2 ENABLED in config")
        else:
            print("INFO: Flash Attention 2 DISABLED in config (set USE_FLASH_ATTENTION=True to enable)")
    else:
        if USE_FLASH_ATTENTION:
            print("INFO: Flash Attention requested but not available (install flash-attn for performance boost)")
    
    return valid

# %% [4.0] Model Loader
# Class to handle Transformers model loading with optimal settings

class TransformersModelLoader:
    """Transformers model loader with quantization support"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
    
    def _get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """Create quantization config based on precision mode"""
        if not BITSANDBYTES_AVAILABLE:
            if self.config.precision_mode in ["8bit", "4bit"]:
                print("WARNING: bitsandbytes not available, falling back to fp16")
                self.config.precision_mode = "fp16"
                return None
        
        if self.config.load_in_8bit:
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_enable_fp32_cpu_offload=False
            )
        elif self.config.load_in_4bit:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16 if self.config.bnb_4bit_compute_dtype == "float16" else torch.bfloat16,
                bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=self.config.bnb_4bit_use_double_quant
            )
        return None
    
    def load(self) -> tuple:
        """Load the model and tokenizer with optimized settings"""
        
        print(f"Loading model from: {self.config.model_path}")
        print(f"Precision mode: {self.config.precision_mode}")
        print(f"Device: {self.config.device}")
        
        # Report Flash Attention status
        if self.config.use_flash_attention and FLASH_ATTN_AVAILABLE:
            print("Flash Attention 2: ENABLED")
        elif self.config.use_flash_attention and not FLASH_ATTN_AVAILABLE:
            print("Flash Attention 2: REQUESTED but NOT AVAILABLE")
        else:
            print("Flash Attention 2: DISABLED")
        
        start_time = time.time()
        
        # Load tokenizer
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            trust_remote_code=self.config.trust_remote_code,
            use_fast=True
        )
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Prepare model loading arguments
        model_kwargs = {
            "pretrained_model_name_or_path": self.config.model_path,
            "device_map": self.config.device_map,
            "trust_remote_code": self.config.trust_remote_code,
            "low_cpu_mem_usage": self.config.low_cpu_mem_usage,
        }
        
        # Add quantization config if needed
        quant_config = self._get_quantization_config()
        if quant_config:
            model_kwargs["quantization_config"] = quant_config
        else:
            # Use appropriate dtype for full precision
            if self.config.precision_mode == "fp16":
                model_kwargs["dtype"] = torch.float16
            else:
                model_kwargs["dtype"] = torch.bfloat16
        
        # Add flash attention if available and requested
        if self.config.use_flash_attention and FLASH_ATTN_AVAILABLE:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        
        # Set memory limits if specified
        if self.config.max_memory:
            model_kwargs["max_memory"] = self.config.max_memory
        
        # Load model
        print("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
        
        # Apply optimizations
        if self.config.use_better_transformer and hasattr(self.model, 'to_bettertransformer'):
            print("Applying BetterTransformer optimization...")
            self.model = self.model.to_bettertransformer()
        
        if self.config.torch_compile and hasattr(torch, 'compile'):
            print("Compiling model with torch.compile()...")
            self.model = torch.compile(self.model)
        
        # Set to eval mode
        self.model.eval()
        
        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f} seconds")
        
        # Print memory usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        
        return self.model, self.tokenizer

# %% [4.1] Model Validation
# Verify model files exist and are valid before attempting to load

def validate_model_files(path: str) -> bool:
    """Validate model files exist and are valid"""
    path = Path(path)
    
    if not path.exists():
        print(f"ERROR: Model directory not found: {path}")
        return False
    
    # Check for essential files
    required_files = {
        "config.json": "Model configuration",
        "tokenizer.json": "Tokenizer data",
        "model.safetensors.index.json": "Model index"
    }
    
    for file, description in required_files.items():
        file_path = path / file
        if not file_path.exists():
            print(f"ERROR: Missing {description}: {file}")
            return False
        else:
            print(f"OK: Found {file}")
    
    # Check for model shards
    model_files = list(path.glob("model-*.safetensors"))
    if not model_files:
        print("ERROR: No model safetensor files found")
        return False
    else:
        total_size = sum(f.stat().st_size for f in model_files) / (1024**3)
        print(f"OK: Found {len(model_files)} model shards, total size: {total_size:.2f}GB")
    
    print("OK: Model validation passed")
    return True

# %% [5.0] Model Initialization
# Create and configure model instance with optional preset support

def initialize_model(config: Optional[ModelConfig] = None) -> tuple:
    """Initialize model and tokenizer with config"""
    
    if config is None:
        # Apply preset if selected
        gen_settings = {}
        if USE_PRESET and USE_PRESET in GENERATION_PRESETS:
            gen_settings = GENERATION_PRESETS[USE_PRESET]
            print(f"Using generation preset: {USE_PRESET}")
        
        config = ModelConfig(
            model_path=MODEL_PATH,
            temperature=gen_settings.get('temperature', TEMPERATURE),
            top_p=gen_settings.get('top_p', TOP_P),
            top_k=gen_settings.get('top_k', TOP_K),
            repetition_penalty=gen_settings.get('repetition_penalty', REPETITION_PENALTY),
            do_sample=gen_settings.get('do_sample', DO_SAMPLE)
        )
    
    loader = TransformersModelLoader(config)
    return loader.load()

# %% [6.0] Inference Test
# Quick test to verify model works and measure performance

def test_inference(model, tokenizer, prompt: str = "Hello! How are you?") -> str:
    """Quick test to verify model works"""
    print("\n--- Running inference test ---")
    print(f"Prompt: {prompt}")
    
    # Prepare messages with proper chat template
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    
    # Apply chat template
    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    inputs = tokenizer(chat_text, return_tensors="pt", padding=True, truncation=True)
    if torch.cuda.is_available():
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
    
    start_time = time.time()
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    inference_time = time.time() - start_time
    
    # Decode only the generated portion (exclude input)
    generated = outputs[0][inputs['input_ids'].shape[1]:]
    result = tokenizer.decode(generated, skip_special_tokens=True)
    
    total_tokens = generated.shape[0]
    tokens_per_second = total_tokens / inference_time
    
    print(f"Response: {result}")
    print(f"Inference time: {inference_time:.2f}s")
    print(f"Tokens/second: {tokens_per_second:.1f}")
    
    return result

# %% [6.1] Terminal Chat Interface
# Interactive chat loop with conversation history and streaming responses

def chat_loop(model, tokenizer, config: ModelConfig):
    """Simple terminal chat interface with conversation history"""
    print("\n--- Chat Interface ---")
    print("Type 'quit' to exit, 'clear' to reset conversation")
    print("-" * 40)
    
    # Initialize conversation
    messages = [
        {"role": "system", "content": config.system_message}
    ]
    
    # Create text streamer for better UX
    streamer = TextStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )
    
    generation_count = 0
    
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() == 'quit':
                print("Goodbye!")
                break
            elif user_input.lower() == 'clear':
                messages = [{"role": "system", "content": config.system_message}]
                print("Conversation cleared")
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                continue
            elif not user_input:
                continue
            
            # Add user message
            messages.append({"role": "user", "content": user_input})
            
            # Apply chat template
            chat_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=config.add_generation_prompt
            )
            
            # Tokenize
            inputs = tokenizer(
                chat_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=config.max_new_tokens
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.to('cuda') for k, v in inputs.items()}
            
            # Generate response
            print("\nAssistant: ", end="", flush=True)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    top_k=config.top_k,
                    repetition_penalty=config.repetition_penalty,
                    do_sample=config.do_sample,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    streamer=streamer,
                    use_cache=config.use_cache
                )
            
            # Extract response
            generated = outputs[0][inputs['input_ids'].shape[1]:]
            response = tokenizer.decode(generated, skip_special_tokens=True)
            
            # Add assistant response to history
            messages.append({"role": "assistant", "content": response})
            
            # Manage conversation length
            if len(messages) > 20:
                # Keep system message and last 18 messages
                messages = [messages[0]] + messages[-18:]
            
            # Clear cache periodically
            generation_count += 1
            if generation_count % config.clear_cache_freq == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("\n[Cache cleared]", end="")
                
        except KeyboardInterrupt:
            print("\nInterrupted. Type 'quit' to exit.")
        except Exception as e:
            print(f"\nError: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

# %% [7.0] Optional Features
# Advanced generation capabilities and monitoring utilities

class KeywordStoppingCriteria(StoppingCriteria):
    """Stop generation when specific keywords are encountered"""
    def __init__(self, keywords, tokenizer):
        self.keywords = keywords
        self.tokenizer = tokenizer
    
    def __call__(self, input_ids, scores, **kwargs) -> bool:
        last_token_text = self.tokenizer.decode(input_ids[0][-1])
        return any(keyword in last_token_text for keyword in self.keywords)

def generate_with_constraints(model, tokenizer, prompt: str, 
                            stop_words: List[str] = None,
                            max_length: int = 500) -> str:
    """Generate with custom stopping criteria"""
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
    
    stopping_criteria = StoppingCriteriaList()
    if stop_words:
        stopping_criteria.append(KeywordStoppingCriteria(stop_words, tokenizer))
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            stopping_criteria=stopping_criteria,
            temperature=0.7,
            do_sample=True
        )
    
    generated = outputs[0][inputs['input_ids'].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)

# Memory usage monitoring
def print_memory_usage():
    """Print current memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    
    ram_used = psutil.virtual_memory().used / (1024**3)
    ram_percent = psutil.virtual_memory().percent
    print(f"RAM: {ram_used:.2f}GB used ({ram_percent:.1f}%)")

# %% [8.0] Main Entry Point
# Orchestrate the entire launch sequence from validation to chat interface

def main():
    """Main execution function"""
    
    print("=" * 50)
    print("Llama 3.2 3B Instruct Transformers Launcher")
    print("=" * 50)
    
    # Step 1: Validate environment
    if not validate_environment():
        print("ERROR: Environment validation failed")
        return 1
    
    # Step 2: Detect hardware
    hw_info = detect_hardware()
    print(f"\nCPU: {hw_info['cpu_name']}")
    print(f"Cores: {hw_info['cpu_count']}, RAM: {hw_info['ram_total_gb']}GB total, {hw_info['ram_available_gb']}GB available")
    print(f"PyTorch: {hw_info['torch_version']}")
    if hw_info['gpu_available']:
        print(f"GPU: {hw_info['gpu_name']}")
        print(f"CUDA: {hw_info['cuda_version']}")
        print(f"VRAM: {hw_info['vram_total_gb']}GB total, {hw_info['vram_available_gb']}GB available")
    
    # Step 3: Validate model files
    if not validate_model_files(MODEL_PATH):
        return 1
    
    # Step 4: Create configuration
    config = ModelConfig(model_path=MODEL_PATH)
    print(f"\nUsing precision mode: {config.precision_mode}")
    
    # Step 5: Load model
    print("\nInitializing model...")
    model, tokenizer = initialize_model(config)
    
    # Step 6: Run test
    test_inference(model, tokenizer)
    
    # Step 7: Print memory usage
    print("\nMemory usage after loading:")
    print_memory_usage()
    
    # Step 8: Start chat
    print("\nStarting chat interface...")
    chat_loop(model, tokenizer, config)
    
    # Cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
