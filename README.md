# Llama 3.2 3B Instruct Launcher

Quick and dirty launcher scripts for running Llama-3.2-3B-Instruct locally with HuggingFace Transformers.

## Description

Simple Python scripts to load and chat with Llama 3.2 3B Instruct model. Includes a full version with hardware detection and optimizations, and a bare-bones version for minimal setup.

## Requirements

- Python 3.13
- PyTorch with CUDA support
- 6-7GB VRAM for FP16 (or 2-3GB for 4-bit quantization)
- Llama-3.2-3B-Instruct model files (safetensors format)

## Setup

1. Create conda environment:
```bash
conda env create -f environment.yml
conda activate llama_32_3b_instruct_launcher
```

2. Install PyTorch with CUDA:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130
```

3. **IMPORTANT: Update model path in the script**
```python
MODEL_PATH = r"E:\ai\models\llama_32_3b_instruct"  # Change to your model location
```

4. Run:
```bash
python llama_32_3b_instruct_launcher.py  # Full version
# or
python llama_32_3b_instruct_launcher_bare.py  # Minimal version
```

## Files

- `llama_32_3b_instruct_launcher.py` - Full launcher with hardware detection, quantization options, memory monitoring
- `llama_32_3b_instruct_launcher_bare.py` - Bare minimum code to load and chat
- `environment.yml` - Conda environment specification

## Usage

Type messages at the prompt. Commands:
- `quit` - Exit the chat
- `clear` - Reset conversation history (full version only)

## Notes

- Bare version skips all validation and optimization features
- Full version includes 8-bit and 4-bit quantization support if bitsandbytes is installed
- Default configuration uses FP16 precision requiring ~6-7GB VRAM
