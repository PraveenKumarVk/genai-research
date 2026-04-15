"""
Chunking strategies for long-document RAG study.
Paper: How Chunking, Embedding, and Reranking Interact in Long-Document RAG
Author: Praveen Kumar Varkala
"""

import nltk
from typing import List

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

from nltk.tokenize import sent_tokenize


def chunk_fixed(text: str, chunk_tokens: int = 256, overlap_tokens: int = 32) -> List[str]:
    """
    Fixed-size chunking with token overlap at boundaries.
    Splits on whitespace tokens; overlap prevents cutting evidence mid-sentence.
    """
    words = text.split()
    if not words:
        return []
    step = max(1, chunk_tokens - overlap_tokens)
    chunks = []
    for i in range(0, len(words), step):
        chunk = ' '.join(words[i:i + chunk_tokens])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def chunk_sentence(text: str, soft_limit: int = 300) -> List[str]:
    """
    Sentence-boundary chunking: accumulate sentences until soft token limit,
    then flush. Preserves sentence integrity at chunk boundaries.
    """
    sentences = sent_tokenize(text)
    chunks, current, current_len = [], [], 0
    for sent in sentences:
        sent_len = len(sent.split())
        if current_len + sent_len > soft_limit and current:
            chunks.append(' '.join(current))
            current, current_len = [], 0
        current.append(sent)
        current_len += sent_len
    if current:
        chunks.append(' '.join(current))
    return [c for c in chunks if c.strip()]


# Registry — add new strategies here
CHUNK_STRATEGIES = {
    'fixed_256': lambda t: chunk_fixed(t, chunk_tokens=256, overlap_tokens=32),
    'fixed_512': lambda t: chunk_fixed(t, chunk_tokens=512, overlap_tokens=64),
    'sent_300':  lambda t: chunk_sentence(t, soft_limit=300),
}
