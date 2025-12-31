"""
RAG Agent - Main orchestrator for Perplexity-style RAG pipeline

Coordinates all components:
- Query Processing
- Hybrid Retrieval
- Document Chunking
- Re-ranking
- Context Building
- LLM Generation
- Citation Formatting
"""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

from .query_processor import QueryProcessor, ProcessedQuery
from .retrieval import HybridRetriever, RetrievedDocument
from .chunker import DocumentChunker, Passage
from .reranker import Reranker, RankedPassage
from .context_builder import ContextBuilder, BuiltContext
from .llm_pipeline import LLMPipeline, GeneratedResponse
from .citation_formatter import CitationFormatter, FormattedResponse
from .cache import RAGCache


@dataclass
class RAGResponse:
    """Complete RAG pipeline response."""
    answer: str
    answer_html: str
    answer_markdown: str
    sources: List[Dict]
    citations_used: List[int]
    confidence: float
    verification_status: str
    
    # Metadata
    query_info: Dict
    retrieval_info: Dict
    timing: Dict


class RAGAgent:
    """
    Perplexity-style RAG Agent.
    
    Complete pipeline:
    1. Query Processing (normalize, expand)
    2. Hybrid Retrieval (web search + dense)
    3. Document Chunking (passage creation)
    4. Re-ranking (cross-encoder scoring)
    5. Context Building (prompt construction)
    6. LLM Pipeline (draft → refine → verify)
    7. Citation Formatting (inline citations + source cards)
    
    Features:
    - Caching for efficiency
    - Configurable components
    - Error handling
    - Timing metrics
    """
    
    def __init__(
        self,
        openai_api_key: str,
        serper_api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        browser_controller=None,
        enable_cache: bool = True,
        enable_verification: bool = True,
        max_sources: int = 8,
    ):
        self.api_key = openai_api_key
        
        # Initialize components
        self.query_processor = QueryProcessor(openai_api_key, model)
        self.retriever = HybridRetriever(
            openai_api_key,
            serper_api_key,
            browser_controller
        )
        self.chunker = DocumentChunker()
        self.reranker = Reranker(openai_api_key, model)
        self.context_builder = ContextBuilder(max_passages=max_sources)
        self.llm_pipeline = LLMPipeline(
            openai_api_key,
            draft_model=model,
            enable_verification=enable_verification
        )
        self.citation_formatter = CitationFormatter()
        
        # Cache
        self.cache = RAGCache() if enable_cache else None
        
        # Config
        self.max_sources = max_sources
    
    async def query(self, user_query: str) -> RAGResponse:
        """
        Process a user query through the full RAG pipeline.
        
        Returns RAGResponse with answer, sources, and metadata.
        """
        timing = {}
        
        try:
            # 1. Query Processing
            start = asyncio.get_event_loop().time()
            processed_query = await self._process_query(user_query)
            timing["query_processing"] = asyncio.get_event_loop().time() - start
            
            # 2. Hybrid Retrieval
            start = asyncio.get_event_loop().time()
            documents = await self._retrieve(processed_query)
            timing["retrieval"] = asyncio.get_event_loop().time() - start
            
            if not documents:
                return self._no_results_response(user_query, timing)
            
            # 3. Document Chunking
            start = asyncio.get_event_loop().time()
            passages = self.chunker.chunk_documents(documents)
            timing["chunking"] = asyncio.get_event_loop().time() - start
            
            if not passages:
                return self._no_results_response(user_query, timing)
            
            # 4. Re-ranking
            start = asyncio.get_event_loop().time()
            ranked_passages = await self.reranker.rerank(
                passages,
                processed_query,
                top_k=self.max_sources
            )
            timing["reranking"] = asyncio.get_event_loop().time() - start
            
            # 5. Context Building
            start = asyncio.get_event_loop().time()
            context = self.context_builder.build(
                ranked_passages,
                processed_query,
                task_type=self._determine_task_type(processed_query)
            )
            timing["context_building"] = asyncio.get_event_loop().time() - start
            
            # 6. LLM Generation
            start = asyncio.get_event_loop().time()
            response = await self.llm_pipeline.generate(context)
            timing["generation"] = asyncio.get_event_loop().time() - start
            
            # 7. Citation Formatting
            start = asyncio.get_event_loop().time()
            formatted = self.citation_formatter.format(response, context)
            timing["formatting"] = asyncio.get_event_loop().time() - start
            
            # Calculate total time
            timing["total"] = sum(timing.values())
            
            # Build final response
            return RAGResponse(
                answer=formatted.content,
                answer_html=formatted.content_html,
                answer_markdown=formatted.content_markdown,
                sources=self.citation_formatter.generate_source_cards_json(
                    formatted.source_cards
                ),
                citations_used=response.citations_used,
                confidence=formatted.confidence,
                verification_status=formatted.verification_status,
                query_info={
                    "original": user_query,
                    "normalized": processed_query.normalized,
                    "intent": processed_query.intent.value,
                    "expanded_queries": processed_query.expanded_queries,
                    "keywords": processed_query.keywords,
                },
                retrieval_info={
                    "documents_retrieved": len(documents),
                    "passages_created": len(passages),
                    "passages_used": len(ranked_passages),
                },
                timing=timing
            )
            
        except Exception as e:
            return self._error_response(user_query, str(e), timing)
    
    async def _process_query(self, query: str) -> ProcessedQuery:
        """Process and expand the query."""
        # Check cache
        if self.cache:
            cached = self.cache.get_query_expansion(query)
            if cached:
                return ProcessedQuery(**cached)
        
        # Process query
        processed = await self.query_processor.process(query)
        
        # Cache result
        if self.cache:
            self.cache.set_query_expansion(query, {
                "original": processed.original,
                "normalized": processed.normalized,
                "intent": processed.intent,
                "expanded_queries": processed.expanded_queries,
                "keywords": processed.keywords,
                "entities": processed.entities,
                "time_sensitivity": processed.time_sensitivity,
            })
        
        return processed
    
    async def _retrieve(self, query: ProcessedQuery) -> List[RetrievedDocument]:
        """Retrieve relevant documents."""
        # Check cache
        if self.cache:
            cached = self.cache.get_search_results(query.normalized)
            if cached:
                return [RetrievedDocument(**doc) for doc in cached]
        
        # Retrieve documents
        documents = await self.retriever.retrieve(
            query,
            max_results=self.max_sources * 2,  # Get more for re-ranking
            crawl_content=True
        )
        
        # Cache results (serialize documents)
        if self.cache and documents:
            serialized = [
                {
                    "id": doc.id,
                    "url": doc.url,
                    "title": doc.title,
                    "content": doc.content,
                    "snippet": doc.snippet,
                    "domain": doc.domain,
                    "timestamp": doc.timestamp.isoformat(),
                    "source_type": doc.source_type,
                    "relevance_score": doc.relevance_score,
                }
                for doc in documents
            ]
            self.cache.set_search_results(query.normalized, serialized)
        
        return documents
    
    def _determine_task_type(self, query: ProcessedQuery) -> str:
        """Determine the task type from query intent."""
        from .query_processor import QueryIntent
        
        intent_to_task = {
            QueryIntent.COMPARISON: "compare",
            QueryIntent.FACTUAL: "answer",
            QueryIntent.EXPLORATION: "summarize",
            QueryIntent.NAVIGATION: "answer",
            QueryIntent.TRANSACTION: "answer",
        }
        
        return intent_to_task.get(query.intent, "answer")
    
    def _no_results_response(self, query: str, timing: Dict) -> RAGResponse:
        """Generate response when no results found."""
        return RAGResponse(
            answer="I couldn't find relevant information to answer your question. "
                   "Please try rephrasing your query or asking about a different topic.",
            answer_html='<p class="no-results">I couldn\'t find relevant information to answer your question.</p>',
            answer_markdown="*I couldn't find relevant information to answer your question.*",
            sources=[],
            citations_used=[],
            confidence=0.0,
            verification_status="no_sources",
            query_info={"original": query},
            retrieval_info={"documents_retrieved": 0},
            timing=timing
        )
    
    def _error_response(self, query: str, error: str, timing: Dict) -> RAGResponse:
        """Generate response for errors."""
        return RAGResponse(
            answer=f"I encountered an error while processing your query: {error}",
            answer_html=f'<p class="error">Error: {error}</p>',
            answer_markdown=f"**Error:** {error}",
            sources=[],
            citations_used=[],
            confidence=0.0,
            verification_status="error",
            query_info={"original": query, "error": error},
            retrieval_info={},
            timing=timing
        )
    
    async def close(self):
        """Clean up resources."""
        await self.retriever.close()
        if self.cache:
            self.cache.clear_all()


# Convenience function for simple usage
async def answer_query(
    query: str,
    openai_api_key: str,
    serper_api_key: Optional[str] = None,
    model: str = "gpt-4o-mini"
) -> RAGResponse:
    """
    Simple function to answer a query using the RAG pipeline.
    
    Example:
        response = await answer_query("What is quantum computing?", api_key)
        print(response.answer)
    """
    agent = RAGAgent(
        openai_api_key=openai_api_key,
        serper_api_key=serper_api_key,
        model=model
    )
    
    try:
        return await agent.query(query)
    finally:
        await agent.close()

