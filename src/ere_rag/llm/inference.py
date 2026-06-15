"""
inference.py

LLM inference backend for the ERE pipeline.
Supports four backends: "ollama", "local", "small", "api"
"""

import os
import time
import torch

# ── Module-level state ──────────────────────────────────────────────────────
_backend      = None   # "ollama" | "local" | "small" | "api"
_tokenizer    = None
_model        = None
_device       = None
_api_client   = None
_api_model    = None
_ollama_model = None

# ── Device detection ────────────────────────────────────────────────────────
def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

# ── Loader ──────────────────────────────────────────────────────────────────
def load_model(
    backend:    str  = "ollama",
    model_name: str  = "llama3",
    device:     str  = "cpu",
    use_4bit:   bool = False,
) -> None:
    """Load the chosen backend into module-level globals."""
    global _backend, _tokenizer, _model, _device, _api_client, _api_model, _ollama_model

    _backend = backend
    _device = device

    if backend == "ollama":
        try:
            import ollama
            _ollama_model = model_name
            print(f"✓ Connected to local Ollama backend (model: {_ollama_model})")
            return
        except ImportError:
            print(" ollama package not installed. Run: pip install ollama")
            print("WARNING: ollama not available, falling back to 'small' backend — results will differ!")
            backend = "small"
            _backend = "small"

    if backend == "api":
        try:
            _load_api_backend(model_name)
            print(f"✓ Loaded Anthropic Claude API backend")
            return
        except ImportError as e:
            print(f" {e}")
            print("Falling back to 'small' backend...")
            backend = "small"
            _backend = "small"

    if backend in ("small", "local"):
        print(f"Loading {model_name}...")
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        resolved = _detect_device() if device == "auto" else device
        _device  = resolved

        _tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token

        if resolved == "cuda" and use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            _model = AutoModelForCausalLM.from_pretrained(
                model_name, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
            )
        elif resolved == "mps":
            _model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float16, trust_remote_code=True
            ).to("mps")
        else:
            _model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float32, trust_remote_code=True
            )

        _model.eval()
        print(f"  Loaded. Parameters: ~{sum(p.numel() for p in _model.parameters())/1e9:.1f}B")

def _load_api_backend(model_name: str):
    try:
        import anthropic
    except ImportError:
        raise ImportError("anthropic package not installed. Run: pip install anthropic")
    
    global _api_client, _api_model
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set.")
    
    _api_client = anthropic.Anthropic(api_key=api_key)
    _api_model = model_name

def is_loaded() -> bool:
    if _backend == "ollama":
        return _ollama_model is not None
    if _backend == "api":
        return _api_client is not None
    return _model is not None and _tokenizer is not None

# ── Inference ───────────────────────────────────────────────────────────────
def run_inference(
    prompt:         str,
    max_new_tokens: int   = 128,
    temperature:    float = 0.0,
) -> tuple[str, float]:
    if not is_loaded():
        raise RuntimeError("Backend not loaded. Call load_model() first.")

    if _backend == "ollama":
        return _run_ollama(prompt, temperature)
    elif _backend == "api":
        return _run_api(prompt, max_new_tokens, temperature)
    else:
        return _run_hf(prompt, max_new_tokens, temperature)

def _run_ollama(prompt: str, temperature: float) -> tuple[str, float]:
    import ollama
    start = time.time()
    response = ollama.chat(
        model=_ollama_model, 
        messages=[{'role': 'user', 'content': prompt}],
        options={'temperature': temperature}
    )
    elapsed = time.time() - start
    return response['message']['content'].strip(), elapsed

def _run_hf(prompt: str, max_new_tokens: int, temperature: float) -> tuple[str, float]:
    messages = [{"role": "user", "content": prompt}]
    formatted = _tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = _tokenizer(formatted, return_tensors="pt", truncation=True, max_length=2048)
    if _device in ("cuda", "mps"):
        inputs = {k: v.to(_device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]

    start = time.time()
    with torch.no_grad():
        if temperature == 0.0:
            outputs = _model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        else:
            outputs = _model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature)
    elapsed = time.time() - start

    new_tokens = outputs[0][prompt_len:]
    response   = _tokenizer.decode(new_tokens, skip_special_tokens=True)
    return response.strip(), elapsed

def _run_api(prompt: str, max_new_tokens: int, temperature: float) -> tuple[str, float]:
    start = time.time()
    message = _api_client.messages.create(
        model=_api_model,
        max_tokens=max_new_tokens,
        temperature=max(temperature, 0.0),
        messages=[{"role": "user", "content": prompt}],
    )
    elapsed  = time.time() - start
    return message.content[0].text.strip(), elapsed