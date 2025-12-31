"""
Re-ranker - Cross-encoder scoring for passage ranking

Implements:
- Semantic relevance scoring
- Recency weighting
- Domain trust scoring
- Diversity penalty
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set
from collections import defaultdict

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from .chunker import Passage
from .query_processor import ProcessedQuery


@dataclass
class RankedPassage:
    """A passage with detailed ranking scores."""
    passage: Passage
    
    # Individual scores (0-1)
    semantic_score: float = 0.0
    recency_score: float = 0.0
    domain_trust_score: float = 0.0
    position_score: float = 0.0
    
    # Penalties
    diversity_penalty: float = 0.0
    
    # Final combined score
    final_score: float = 0.0
    
    # Rank position
    rank: int = 0


class Reranker:
    """
    Re-ranks passages using multiple signals:
    
    1. Semantic relevance (cross-encoder style using LLM)
    2. Recency (time-based decay)
    3. Domain trust (known reliable sources)
    4. Position (prefer earlier passages in document)
    5. Diversity (penalize redundant content)
    """
    
    # Domain trust scores (0-1, higher is more trusted)
    DOMAIN_TRUST = {
        # High trust - official sources
        "wikipedia.org": 0.9,
        "gov": 0.85,
        "edu": 0.85,
        "reuters.com": 0.85,
        "apnews.com": 0.85,
        
        # Good trust - established sources
        "nytimes.com": 0.8,
        "bbc.com": 0.8,
        "theguardian.com": 0.8,
        "washingtonpost.com": 0.8,
        "techcrunch.com": 0.75,
        "wired.com": 0.75,
        "arstechnica.com": 0.75,
        
        # E-commerce (good for products)
        "amazon.com": 0.7,
        "bestbuy.com": 0.7,
        "walmart.com": 0.7,
        "target.com": 0.7,
        
        # Tech documentation
        "docs.python.org": 0.9,
        "developer.mozilla.org": 0.9,
        "stackoverflow.com": 0.7,
        "github.com": 0.7,
        
        # Default
        "default": 0.5
    }
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        use_llm_scoring: bool = True,
        semantic_weight: float = 0.5,
        recency_weight: float = 0.15,
        trust_weight: float = 0.15,
        position_weight: float = 0.1,
        diversity_weight: float = 0.1,
    ):
        self.use_llm_scoring = use_llm_scoring
        
        # Weights must sum to 1
        total = semantic_weight + recency_weight + trust_weight + position_weight + diversity_weight
        self.semantic_weight = semantic_weight / total
        self.recency_weight = recency_weight / total
        self.trust_weight = trust_weight / total
        self.position_weight = position_weight / total
        self.diversity_weight = diversity_weight / total
        
        if use_llm_scoring:
            self.llm = ChatOpenAI(
                model=model,
                api_key=api_key,
                temperature=0,
            )
            
            self.scoring_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a relevance scoring expert. Given a query and a passage, rate how relevant the passage is to answering the query.

Score from 0 to 10:
- 0-2: Not relevant at all
- 3-4: Marginally relevant, tangentially related
- 5-6: Somewhat relevant, contains some useful information
- 7-8: Highly relevant, directly addresses the query
- 9-10: Perfectly relevant, exactly what the user needs

Respond with ONLY a number from 0 to 10."""),
                ("human", "Query: {query}\n\nPassage: {passage}\n\nRelevance score (0-10):")
            ])
    
    async def rerank(
        self,
        passages: List[Passage],
        query: ProcessedQuery,
        top_k: int = 10
    ) -> List[RankedPassage]:
        """
        Re-rank passages using multiple signals.
        
        Returns top_k passages with detailed scores.
        """
        if not passages:
            return []
        
        ranked_passages = []
        
        # Score each passage
        for passage in passages:
            ranked = RankedPassage(passage=passage)
            
            # 1. Semantic relevance score
            if self.use_llm_scoring:
                ranked.semantic_score = await self._score_semantic(passage, query)
            else:
                ranked.semantic_score = passage.relevance_score
            
            # 2. Recency score
            ranked.recency_score = self._score_recency(passage)
            
            # 3. Domain trust score
            ranked.domain_trust_score = self._score_domain_trust(passage)
            
            # 4. Position score (prefer earlier passages)
            ranked.position_score = self._score_position(passage)
            
            ranked_passages.append(ranked)
        
        # 5. Calculate diversity penalties (after initial ranking)
        self._apply_diversity_penalty(ranked_passages)
        
        # Calculate final scores
        for rp in ranked_passages:
            rp.final_score = (
                self.semantic_weight * rp.semantic_score +
                self.recency_weight * rp.recency_score +
                self.trust_weight * rp.domain_trust_score +
                self.position_weight * rp.position_score -
                self.diversity_weight * rp.diversity_penalty
            )
        
        # Sort by final score
        ranked_passages.sort(key=lambda x: x.final_score, reverse=True)
        
        # Assign ranks
        for i, rp in enumerate(ranked_passages):
            rp.rank = i + 1
        
        return ranked_passages[:top_k]
    
    async def _score_semantic(self, passage: Passage, query: ProcessedQuery) -> float:
        """Score semantic relevance using LLM."""
        try:
            # Truncate passage for efficiency
            passage_text = passage.content[:1500]
            
            chain = self.scoring_prompt | self.llm
            response = await chain.ainvoke({
                "query": query.normalized,
                "passage": passage_text
            })
            
            # Parse score
            score_text = response.content.strip()
            score = float(score_text)
            return min(max(score / 10.0, 0.0), 1.0)  # Normalize to 0-1
            
        except Exception as e:
            # Fallback to existing relevance score
            return passage.relevance_score
    
    def _score_recency(self, passage: Passage) -> float:
        """Score based on content recency (time decay)."""
        # For now, use a default score since we don't have timestamps
        # In production, this would check document timestamps
        return 0.7  # Assume moderately recent
    
    def _score_domain_trust(self, passage: Passage) -> float:
        """Score based on domain trustworthiness."""
        domain = passage.domain.lower()
        
        # Check exact match
        if domain in self.DOMAIN_TRUST:
            return self.DOMAIN_TRUST[domain]
        
        # Check TLD
        for tld in ["gov", "edu"]:
            if domain.endswith(f".{tld}"):
                return self.DOMAIN_TRUST[tld]
        
        # Check partial match
        for trusted_domain, score in self.DOMAIN_TRUST.items():
            if trusted_domain in domain:
                return score
        
        return self.DOMAIN_TRUST["default"]
    
    def _score_position(self, passage: Passage) -> float:
        """Score based on position in document (earlier is better)."""
        if passage.total_passages <= 1:
            return 1.0
        
        # Linear decay based on position
        position_ratio = passage.position_in_doc / passage.total_passages
        return 1.0 - (position_ratio * 0.5)  # Max 50% penalty
    
    def _apply_diversity_penalty(self, ranked_passages: List[RankedPassage]) -> None:
        """
        Apply diversity penalty to reduce redundant content.
        
        Penalizes passages from the same domain or with similar content.
        """
        domain_counts: Dict[str, int] = defaultdict(int)
        
        # Sort by semantic score first for penalty calculation
        sorted_passages = sorted(
            ranked_passages,
            key=lambda x: x.semantic_score,
            reverse=True
        )
        
        for rp in sorted_passages:
            domain = rp.passage.domain
            
            # Penalize repeated domains
            count = domain_counts[domain]
            if count > 0:
                rp.diversity_penalty = min(count * 0.15, 0.5)  # Max 50% penalty
            
            domain_counts[domain] += 1
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity using word overlap."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)

