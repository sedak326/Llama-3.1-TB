#!/usr/bin/env python3
"""
rag_pipeline.py — Core TB RAG retrieve-then-generate pipeline.

Mirrors query_rag() in PLVis's proteome_explorer_ux2.py, adapted to retrieve
from the local Chroma TB corpus index (build_index.py) and generate with one
of this project's own Llama checkpoints instead of GPT-4o-mini — fully
self-hosted, no OpenAI dependency.

Importable as a module (load_vectorstore / load_generator / query_rag), or
run directly via chat_rag.py for a CLI.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    from langchain_chroma import Chroma
except ImportError:
    Chroma = None  # only needed for RAG mode; --no-rag (the deployed config) works without it

RAG_TOP_K = 8

SYSTEM_PROMPT = """\
You are an expert tuberculosis (TB) research assistant. You have access to a curated corpus \
of TB research literature (context chunks below). Each chunk starts with its source location \
(Document, Chapter, Section, Page) — cite it naturally using that source location when you use \
it (e.g. "According to document 10027986, page 3..."). Prioritise answering from the context. \
The retrieved text may itself contain inline numeric or bracketed citation markers from the \
original paper's own reference list (e.g. "(39)", "[12]") — you were NOT given that paper's \
reference list, so these numbers are meaningless to you and to the reader. Never reproduce a \
bare numeric/bracketed citation marker from the retrieved text; if you need to refer to the \
claim it was attached to, cite the document/section it came from instead, or drop the marker \
entirely. If the context chunks are not relevant to the question, answer from your general \
TB/biomedical knowledge but clearly note: "This is from general knowledge, not the specific \
literature." Never fabricate citations.

Format your answer for readability using Markdown: keep paragraphs short, use a bullet list \
("- item") when enumerating multiple options, steps, or causes, use **bold** for key terms you \
are defining or emphasising, and use ## or ### headers only for longer answers with multiple \
distinct parts. Do not force structure (headers/bullets) onto a short, single-point answer that \
doesn't need it."""

# No-retrieval variant: same assistant framing and formatting rules as SYSTEM_PROMPT
# above, minus the context/citation instructions, which don't apply without retrieval.
DIRECT_SYSTEM_PROMPT = """\
You are an expert tuberculosis (TB) research assistant. You help TB researchers with factual \
questions about genes, drugs, and mechanisms; interpreting experimental results; generating \
hypotheses; advising on target prioritization; and explaining dense literature. Answer from your \
own knowledge. Never fabricate citations or specific figures you are not confident about.

Format your answer for readability using Markdown: keep paragraphs short, use a bullet list \
("- item") when enumerating multiple options, steps, or causes, use **bold** for key terms you \
are defining or emphasising, and use ## or ### headers only for longer answers with multiple \
distinct parts. Do not force structure (headers/bullets) onto a short, single-point answer that \
doesn't need it."""

# SFT-3 was fine-tuned exclusively on substantive TB Q&A, so bare greetings are
# out-of-distribution for it: appending the per-turn format reminder (needed to
# keep real answers structured - see query_direct/query_rag below) after a
# near-empty message like "hi" makes it free-associate into a random TB topic
# instead of just greeting back. Caught these before they ever reach the model.
_CHITCHAT_PATTERN = re.compile(
    r"^(hi|hello|hey|hiya|yo|sup|good\s*(morning|afternoon|evening)|"
    r"thanks?(\s*you)?|ty|ok|okay|cool|nice|great|bye|goodbye|see\s*ya|"
    r"how\s*are\s*you|how'?s\s*it\s*going|what'?s\s*up)[\s.!?]*$",
    re.IGNORECASE,
)
_CHITCHAT_REPLY = (
    "Hi! I'm a TB research assistant — ask me to interpret experimental results, "
    "critique methodology, prioritize targets, or generate hypotheses, as well as "
    "answer factual questions about TB genes, drugs, and mechanisms from the literature."
)


def _chitchat_reply(question: str) -> Optional[str]:
    """Returns a canned reply if `question` is just a greeting/acknowledgement
    with no real content, else None."""
    return _CHITCHAT_REPLY if _CHITCHAT_PATTERN.match(question.strip()) else None


_vectorstore: Optional[Chroma] = None
_model = None
_tokenizer = None


def load_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        # Lazy import: build_index.py (and its Chroma/sentence-transformers deps)
        # isn't part of this showcase repo's core/ sample - only needed if you
        # wire up the RAG path yourself. --no-rag (the deployed config) never
        # calls this function.
        from build_index import LocalSentenceTransformerEmbeddings, CHROMA_PERSIST_DIR, COLLECTION_NAME, EMBEDDING_MODEL
        embeddings = LocalSentenceTransformerEmbeddings(EMBEDDING_MODEL)
        _vectorstore = Chroma(
            persist_directory=str(CHROMA_PERSIST_DIR),
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME,
        )
        count = _vectorstore._collection.count()
        print(f"Chroma vectorstore loaded — {count} chunks indexed")
    return _vectorstore


def load_generator(model_path: str, tokenizer_path: Optional[str] = None, adapter_path: Optional[str] = None):
    """Loads the generation model, mirroring evaluate_emcqa.py's loading pattern
    (4-bit NF4 + PEFT adapter when --adapter is given)."""
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    tok_path = tokenizer_path or model_path
    # truncation_side="left": if the prompt (context + question) ever exceeds
    # max_length, drop from the front (oldest retrieved context) rather than
    # the default right-truncation, which would cut off the question itself.
    _tokenizer = AutoTokenizer.from_pretrained(tok_path, padding_side="left", truncation_side="left")
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    if adapter_path:
        from peft import PeftModel
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        base = AutoModelForCausalLM.from_pretrained(
            model_path, quantization_config=bnb_config, device_map="auto",
        )
        _model = PeftModel.from_pretrained(base, adapter_path)
    else:
        _model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto",
        )
    _model.eval()
    return _model, _tokenizer


def retrieve(question: str, top_k: int = RAG_TOP_K) -> list:
    """Returns a list of langchain Document objects, most relevant first."""
    vs = load_vectorstore()
    return vs.similarity_search(question, k=top_k)


def retrieve_with_score(question: str, top_k: int = RAG_TOP_K) -> list:
    """Like retrieve(), but also returns each chunk's Chroma distance score
    (lower = closer match). Used for adaptive/confidence-gated retrieval:
    the top-1 score is a cheap signal for whether anything genuinely
    relevant was found, as opposed to the index just returning the
    least-bad match from a 489k-chunk corpus regardless of relevance."""
    vs = load_vectorstore()
    return vs.similarity_search_with_score(question, k=top_k)


def build_context(retrieved: list) -> str:
    if not retrieved:
        return ""
    return "\n\n---\n\n".join(d.page_content for d in retrieved)


def query_rag(question: str, model_path: str, tokenizer_path: Optional[str] = None,
              adapter_path: Optional[str] = None, top_k: int = RAG_TOP_K,
              max_new_tokens: int = 512) -> tuple[str, list]:
    """Retrieve relevant TB corpus chunks and generate a grounded answer.

    Returns (answer, retrieved_documents).
    """
    canned = _chitchat_reply(question)
    if canned is not None:
        return canned, []

    model, tokenizer = load_generator(model_path, tokenizer_path, adapter_path)
    retrieved = retrieve(question, top_k=top_k)
    context = build_context(retrieved)

    # Repeated right before generation starts (not just in the system prompt):
    # the SFT style otherwise dominates and the model falls back to one dense
    # paragraph regardless of the system prompt's formatting instructions.
    format_reminder = (
        "(Format this answer in Markdown: short paragraphs, a bullet list for "
        "multiple options/steps/causes, **bold** for key terms, ## headers only "
        "if it's a long multi-part answer.)"
    )
    user_content = (
        f"Context:\n{context}\n\nQuestion: {question}\n\n{format_reminder}\n\nAnswer:"
        if context else
        f"Question: {question}\n\n{format_reminder}\n\nAnswer:"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    answer = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
    return answer, retrieved


def query_direct(question: str, model_path: str, tokenizer_path: Optional[str] = None,
                  adapter_path: Optional[str] = None, max_new_tokens: int = 512) -> str:
    """Generate an answer with no retrieval step at all — just the fine-tuned
    model answering from its own parametric knowledge. Used for the SFT-3-only
    (no RAG) serving condition.

    Returns the answer string.
    """
    canned = _chitchat_reply(question)
    if canned is not None:
        return canned

    model, tokenizer = load_generator(model_path, tokenizer_path, adapter_path)

    format_reminder = (
        "(Format this answer in Markdown: short paragraphs, a bullet list for "
        "multiple options/steps/causes, **bold** for key terms, ## headers only "
        "if it's a long multi-part answer.)"
    )
    messages = [
        {"role": "system", "content": DIRECT_SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {question}\n\n{format_reminder}\n\nAnswer:"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    return tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
