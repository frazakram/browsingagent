"""
Context Builder - Constructs LLM prompts with provenance metadata

Implements:
- Token budgeting
- Provenance token attachment
- Context compression
- Extraction highlights
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict

from .reranker import RankedPassage
from .query_processor import ProcessedQuery


@dataclass
class BuiltContext:
    """The constructed context ready for LLM."""
    system_prompt: str
    user_prompt: str
    passages_used: List[RankedPassage]
    total_tokens_estimate: int
    sources: List[Dict]  # For citation tracking


class ContextBuilder:
    """
    Builds optimized context for LLM generation.
    
    Features:
    - Token budgeting to fit model context
    - Provenance tokens for citation tracking
    - Context compression for efficiency
    - Source metadata preservation
    """
    
    def __init__(
        self,
        max_context_tokens: int = 8000,
        max_passages: int = 8,
        chars_per_token: int = 4,
    ):
        self.max_context_tokens = max_context_tokens
        self.max_passages = max_passages
        self.chars_per_token = chars_per_token
    
    def build(
        self,
        ranked_passages: List[RankedPassage],
        query: ProcessedQuery,
        task_type: str = "answer"  # "answer", "summarize", "compare"
    ) -> BuiltContext:
        """
        Build context for LLM generation.
        
        Returns BuiltContext with system prompt, user prompt, and source tracking.
        """
        # Select passages within token budget
        selected_passages = self._select_passages(ranked_passages)
        
        # Build source list for citations
        sources = self._build_source_list(selected_passages)
        
        # Build system prompt
        system_prompt = self._build_system_prompt(task_type, sources)
        
        # Build user prompt with context
        user_prompt = self._build_user_prompt(query, selected_passages, sources)
        
        # Estimate total tokens
        total_chars = len(system_prompt) + len(user_prompt)
        token_estimate = total_chars // self.chars_per_token
        
        return BuiltContext(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            passages_used=selected_passages,
            total_tokens_estimate=token_estimate,
            sources=sources
        )
    
    def _select_passages(self, ranked_passages: List[RankedPassage]) -> List[RankedPassage]:
        """Select passages within token budget."""
        selected = []
        total_chars = 0
        max_chars = self.max_context_tokens * self.chars_per_token
        
        # Reserve space for prompts (~2000 chars)
        available_chars = max_chars - 2000
        
        for rp in ranked_passages[:self.max_passages * 2]:  # Consider more than max
            passage_chars = len(rp.passage.content)
            
            if total_chars + passage_chars <= available_chars:
                selected.append(rp)
                total_chars += passage_chars
                
                if len(selected) >= self.max_passages:
                    break
        
        return selected
    
    def _build_source_list(self, passages: List[RankedPassage]) -> List[Dict]:
        """Build source list for citation tracking."""
        sources = []
        seen_urls = set()
        
        for i, rp in enumerate(passages):
            if rp.passage.url not in seen_urls:
                seen_urls.add(rp.passage.url)
                sources.append({
                    "id": i + 1,
                    "url": rp.passage.url,
                    "title": rp.passage.title,
                    "domain": rp.passage.domain,
                    "relevance": rp.final_score,
                })
        
        return sources
    
    def _build_system_prompt(self, task_type: str, sources: List[Dict]) -> str:
        """Build the system prompt based on task type."""
        
        source_list = "\n".join([
            f"[{s['id']}] {s['title']} ({s['domain']})"
            for s in sources
        ])
        
        base_prompt = f"""You are a knowledgeable AI assistant that provides accurate, well-cited answers based on retrieved information.

AVAILABLE SOURCES:
{source_list}

CITATION RULES:
1. ALWAYS cite sources using [1], [2], etc. when making claims
2. Place citations immediately after the relevant statement
3. You can cite multiple sources for a single claim: [1][3]
4. Only cite sources that directly support your statement
5. If information conflicts between sources, mention both perspectives with their citations
6. If no source supports a claim, clearly state it's your general knowledge

RESPONSE GUIDELINES:
- Be concise but comprehensive
- Start with a direct answer to the question
- Use bullet points for lists or comparisons
- Highlight key facts and numbers
- Acknowledge uncertainty when sources are limited"""

        if task_type == "compare":
            base_prompt += """

COMPARISON FORMAT:
- Create a clear comparison structure
- Use tables or bullet points to highlight differences
- Cite sources for each comparison point"""
        
        elif task_type == "summarize":
            base_prompt += """

SUMMARY FORMAT:
- Provide a concise summary (2-3 paragraphs max)
- Lead with the most important information
- Include key facts and figures with citations"""
        
        return base_prompt
    
    def _build_user_prompt(
        self,
        query: ProcessedQuery,
        passages: List[RankedPassage],
        sources: List[Dict]
    ) -> str:
        """Build the user prompt with context."""
        
        # Build context section with provenance
        context_parts = []
        url_to_source_id = {s['url']: s['id'] for s in sources}
        
        for rp in passages:
            source_id = url_to_source_id.get(rp.passage.url, "?")
            
            # Add passage with provenance marker
            passage_text = f"""---
SOURCE [{source_id}]: {rp.passage.title}
URL: {rp.passage.url}
CONTENT:
{rp.passage.content[:1500]}
---"""
            context_parts.append(passage_text)
        
        context_section = "\n\n".join(context_parts)
        
        # Build the complete user prompt
        user_prompt = f"""RETRIEVED CONTEXT:
{context_section}

USER QUESTION: {query.original}

Please provide a comprehensive answer based on the retrieved context. Remember to cite your sources using [1], [2], etc."""
        
        return user_prompt
    
    def compress_passage(self, passage: str, max_chars: int) -> str:
        """Compress a passage to fit within char limit while preserving key info."""
        if len(passage) <= max_chars:
            return passage
        
        # Simple truncation with ellipsis
        # In production, use extractive summarization
        return passage[:max_chars - 3] + "..."

