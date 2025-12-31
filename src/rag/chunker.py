"""
Document Chunker - Splits documents into overlapping passages

Creates passage windows sized for LLM context with:
- Overlapping boundaries for context preservation
- Sentence-aware splitting
- Provenance tracking
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from .retrieval import RetrievedDocument


@dataclass
class Passage:
    """A chunk/passage from a document with provenance."""
    id: str
    document_id: str
    content: str
    start_char: int
    end_char: int
    
    # Provenance
    url: str
    title: str
    domain: str
    
    # Scoring
    relevance_score: float = 0.0
    position_in_doc: int = 0
    total_passages: int = 1


class DocumentChunker:
    """
    Splits documents into overlapping passages optimized for LLM context.
    
    Features:
    - Configurable chunk size and overlap
    - Sentence-aware boundaries
    - Paragraph-aware splitting
    - Preserves provenance metadata
    """
    
    def __init__(
        self,
        chunk_size: int = 500,  # Target tokens (~2000 chars)
        chunk_overlap: int = 100,  # Overlap tokens (~400 chars)
        min_chunk_size: int = 100,  # Minimum viable chunk
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        
        # Approximate chars per token (conservative estimate)
        self.chars_per_token = 4
    
    def chunk_document(self, document: RetrievedDocument) -> List[Passage]:
        """
        Split a document into overlapping passages.
        
        Returns list of Passage objects with provenance.
        """
        text = self._prepare_text(document)
        
        if not text:
            return []
        
        # Calculate char limits
        max_chars = self.chunk_size * self.chars_per_token
        overlap_chars = self.chunk_overlap * self.chars_per_token
        min_chars = self.min_chunk_size * self.chars_per_token
        
        # Split into paragraphs first
        paragraphs = self._split_paragraphs(text)
        
        # Create chunks from paragraphs
        chunks = self._create_chunks(paragraphs, max_chars, overlap_chars, min_chars)
        
        # Create passages with metadata
        passages = []
        for i, (content, start, end) in enumerate(chunks):
            passage = Passage(
                id=f"{document.id}_{i}",
                document_id=document.id,
                content=content.strip(),
                start_char=start,
                end_char=end,
                url=document.url,
                title=document.title,
                domain=document.domain,
                relevance_score=document.relevance_score,
                position_in_doc=i,
                total_passages=len(chunks)
            )
            passages.append(passage)
        
        # Update total passages count
        for p in passages:
            p.total_passages = len(passages)
        
        return passages
    
    def chunk_documents(self, documents: List[RetrievedDocument]) -> List[Passage]:
        """Chunk multiple documents and return all passages."""
        all_passages = []
        for doc in documents:
            passages = self.chunk_document(doc)
            all_passages.extend(passages)
        return all_passages
    
    def _prepare_text(self, document: RetrievedDocument) -> str:
        """Prepare document text for chunking."""
        # Combine title, snippet, and content
        parts = []
        
        if document.title:
            parts.append(f"# {document.title}")
        
        if document.content:
            parts.append(document.content)
        elif document.snippet:
            parts.append(document.snippet)
        
        text = "\n\n".join(parts)
        
        # Clean up the text
        text = self._clean_text(text)
        
        return text
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        text = re.sub(r'\t+', ' ', text)
        
        # Remove URLs (optional - keeps them for citations)
        # text = re.sub(r'https?://\S+', '[URL]', text)
        
        return text.strip()
    
    def _split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        # Split on double newlines
        paragraphs = re.split(r'\n\n+', text)
        
        # Filter empty paragraphs
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        return paragraphs
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting (handles common cases)
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _create_chunks(
        self,
        paragraphs: List[str],
        max_chars: int,
        overlap_chars: int,
        min_chars: int
    ) -> List[tuple]:
        """
        Create overlapping chunks from paragraphs.
        
        Returns list of (content, start_char, end_char) tuples.
        """
        if not paragraphs:
            return []
        
        # Join paragraphs back for char tracking
        full_text = "\n\n".join(paragraphs)
        
        # If text is small enough, return as single chunk
        if len(full_text) <= max_chars:
            return [(full_text, 0, len(full_text))]
        
        chunks = []
        current_chunk = []
        current_length = 0
        current_start = 0
        char_position = 0
        
        for para in paragraphs:
            para_len = len(para) + 2  # +2 for \n\n
            
            # If single paragraph is too long, split by sentences
            if para_len > max_chars:
                # Flush current chunk first
                if current_chunk:
                    chunk_text = "\n\n".join(current_chunk)
                    chunks.append((chunk_text, current_start, char_position))
                    current_chunk = []
                    current_length = 0
                
                # Split paragraph into sentences
                sentences = self._split_sentences(para)
                sentence_chunk = []
                sentence_length = 0
                sentence_start = char_position
                
                for sent in sentences:
                    sent_len = len(sent) + 1
                    
                    if sentence_length + sent_len > max_chars and sentence_chunk:
                        chunk_text = " ".join(sentence_chunk)
                        chunks.append((chunk_text, sentence_start, char_position))
                        
                        # Overlap: keep last few sentences
                        overlap_sents = []
                        overlap_len = 0
                        for s in reversed(sentence_chunk):
                            if overlap_len + len(s) < overlap_chars:
                                overlap_sents.insert(0, s)
                                overlap_len += len(s)
                            else:
                                break
                        
                        sentence_chunk = overlap_sents
                        sentence_length = overlap_len
                        sentence_start = char_position - overlap_len
                    
                    sentence_chunk.append(sent)
                    sentence_length += sent_len
                
                if sentence_chunk:
                    current_chunk = sentence_chunk
                    current_length = sentence_length
                    current_start = sentence_start
                
                char_position += para_len
                continue
            
            # Check if adding this paragraph would exceed max
            if current_length + para_len > max_chars and current_chunk:
                # Save current chunk
                chunk_text = "\n\n".join(current_chunk)
                chunks.append((chunk_text, current_start, char_position))
                
                # Create overlap from end of current chunk
                overlap_paras = []
                overlap_len = 0
                for p in reversed(current_chunk):
                    if overlap_len + len(p) < overlap_chars:
                        overlap_paras.insert(0, p)
                        overlap_len += len(p)
                    else:
                        break
                
                current_chunk = overlap_paras
                current_length = overlap_len
                current_start = char_position - overlap_len
            
            current_chunk.append(para)
            current_length += para_len
            char_position += para_len
        
        # Add final chunk
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            if len(chunk_text) >= min_chars or not chunks:
                chunks.append((chunk_text, current_start, char_position))
            elif chunks:
                # Merge with previous chunk if too small
                prev_text, prev_start, _ = chunks[-1]
                chunks[-1] = (prev_text + "\n\n" + chunk_text, prev_start, char_position)
        
        return chunks

