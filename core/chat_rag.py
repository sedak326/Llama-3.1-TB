#!/usr/bin/env python3
"""
chat_rag.py — model/adapter defaults shared by chat_ui.py.

The full repo's version of this file is also a terminal CLI for querying the
RAG pipeline directly; trimmed here to just the constants chat_ui.py needs.

DEFAULT_ADAPTER points at the public LoRA adapter on the Hugging Face Hub
(https://huggingface.co/kvlkS/tb-research-assistant-lora) rather than the
cluster-local checkpoint path used internally, so this repo runs standalone.
"""

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_ADAPTER = "kvlkS/tb-research-assistant-lora"
