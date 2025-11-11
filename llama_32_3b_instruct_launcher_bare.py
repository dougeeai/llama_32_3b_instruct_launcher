# %% [0.0] Launcher Script Info
# Minimal launcher script metadata
# Llama 3.2 3B Instruct Transformers Launcher - BARE BONES VERSION
# Description: Minimal launcher for Llama-3.2-3B-Instruct safetensors models
# Author: dougeeai
# Created: 2025-11-09
# Last Updated: 2025-11-11

# %% [0.1] Model Card & Summary
# Bare bones version - minimal code, no checks
# MODEL: Llama-3.2-3B-Instruct
# Architecture: Llama 3.2 (3.21B parameters)

# %% [1.0] Core Imports
# Essential imports only
import os
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

# %% [1.1] Utility Imports
# Bare Version: Utility imports skipped

# %% [2.0] Base Directory Configuration
# Set base directory for portability
BASE_DIR = r"E:\ai"  # <-- CHANGE THIS to your AI folder location
MODELS_DIR = os.path.join(BASE_DIR, "models")

# %% [2.1] Model Source Configuration
# Simplified model path configuration
MODEL_NAME = "llama_32_3b_instruct"
MODEL_PATH = os.path.join(MODELS_DIR, MODEL_NAME)

# %% [2.2] User Configuration - All Settings
# Core settings for model operation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEMPERATURE = 0.7
TOP_P = 0.9
TOP_K = 40
REPETITION_PENALTY = 1.1
MAX_NEW_TOKENS = 2048
SYSTEM_MESSAGE = "You are a helpful AI assistant."

# %% [2.3] Model Configuration Dataclass
# Bare Version: Dataclass skipped - using direct variables

# %% [3.0] Hardware Auto-Detection
# Bare Version: Auto-detection skipped

# %% [3.1] Hardware Detection
# Bare Version: Hardware detection skipped

# %% [3.2] Environment Validation
# Bare Version: Environment validation skipped

# %% [4.0] Model Loader
# Direct model and tokenizer loading
def load_model():
    """Load model and tokenizer - bare bones"""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    model.eval()
    return model, tokenizer

# %% [4.1] Model Validation
# Bare Version: Model validation skipped

# %% [5.0] Model Initialization
# Bare Version: Using direct load_model() instead

# %% [6.0] Inference Test
# Bare Version: Inference test skipped

# %% [6.1] Terminal Chat Interface
# Minimal chat loop with streaming
def chat_loop(model, tokenizer):
    """Minimal chat interface"""
    print("Chat Interface - Type 'quit' to exit")
    print("-" * 40)
    
    messages = [{"role": "system", "content": SYSTEM_MESSAGE}]
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == 'quit':
            break
        elif not user_input:
            continue
        
        messages.append({"role": "user", "content": user_input})
        
        # Apply chat template
        chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(chat_text, return_tensors="pt", padding=True, truncation=True)
        
        if DEVICE == "cuda":
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
        
        print("\nAssistant: ", end="", flush=True)
        
        # Generate with streaming
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                top_k=TOP_K,
                repetition_penalty=REPETITION_PENALTY,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                streamer=streamer
            )
        
        # Extract and store response
        generated = outputs[0][inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(generated, skip_special_tokens=True)
        messages.append({"role": "assistant", "content": response})
        
        # Keep history manageable
        if len(messages) > 20:
            messages = [messages[0]] + messages[-18:]

# %% [7.0] Optional Features
# Bare Version: Optional features skipped

# %% [8.0] Main Entry Point
# Simple main function - load and chat
def main():
    """Bare bones main - just load and chat"""
    print("Loading model...")
    model, tokenizer = load_model()
    print("Model loaded. Starting chat...")
    chat_loop(model, tokenizer)

if __name__ == "__main__":
    main()
