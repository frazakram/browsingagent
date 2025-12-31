"""
Hybrid Retrieval - Combines Lexical and Dense Retrieval

Implements:
- Lexical Search (Web APIs, BM25)
- Dense Retrieval (Embeddings + Vector similarity)
- Web crawling for content extraction
"""

from __future__ import annotations

import asyncio
import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse, quote_plus

import httpx
from langchain_openai import OpenAIEmbeddings

from .query_processor import ProcessedQuery


@dataclass
class RetrievedDocument:
    """A retrieved document with provenance metadata."""
    id: str
    url: str
    title: str
    content: str
    snippet: str
    domain: str
    timestamp: datetime
    source_type: str  # "web_search", "direct_crawl", "vector_db"
    relevance_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.id:
            self.id = hashlib.md5(f"{self.url}{self.title}".encode()).hexdigest()[:12]


class WebSearcher:
    """
    Performs web searches using available APIs.
    Falls back to DuckDuckGo HTML scraping if no API key available.
    """
    
    def __init__(self, serper_api_key: Optional[str] = None):
        self.serper_api_key = serper_api_key
        self.client = httpx.AsyncClient(
            timeout=30.0,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
        )
    
    async def search(self, query: str, num_results: int = 10) -> List[RetrievedDocument]:
        """Perform web search and return results."""
        if self.serper_api_key:
            return await self._search_serper(query, num_results)
        else:
            return await self._search_duckduckgo(query, num_results)
    
    async def _search_serper(self, query: str, num_results: int) -> List[RetrievedDocument]:
        """Search using Serper.dev Google Search API."""
        try:
            response = await self.client.post(
                "https://google.serper.dev/search",
                headers={"X-API-KEY": self.serper_api_key},
                json={"q": query, "num": num_results}
            )
            response.raise_for_status()
            data = response.json()
            
            documents = []
            for result in data.get("organic", []):
                doc = RetrievedDocument(
                    id="",
                    url=result.get("link", ""),
                    title=result.get("title", ""),
                    content="",  # Will be filled by crawler
                    snippet=result.get("snippet", ""),
                    domain=urlparse(result.get("link", "")).netloc,
                    timestamp=datetime.now(),
                    source_type="web_search",
                    relevance_score=1.0 - (len(documents) * 0.05),  # Position-based score
                    metadata={"position": len(documents) + 1}
                )
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            print(f"Serper search failed: {e}")
            return await self._search_duckduckgo(query, num_results)
    
    async def _search_duckduckgo(self, query: str, num_results: int) -> List[RetrievedDocument]:
        """Fallback search using DuckDuckGo HTML."""
        try:
            encoded_query = quote_plus(query)
            response = await self.client.get(
                f"https://html.duckduckgo.com/html/?q={encoded_query}",
                follow_redirects=True
            )
            response.raise_for_status()
            
            # Parse results from HTML
            documents = []
            html = response.text
            
            # Extract result blocks
            result_pattern = r'<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>([^<]*)</a>'
            snippet_pattern = r'<a[^>]*class="result__snippet"[^>]*>([^<]*)</a>'
            
            urls = re.findall(result_pattern, html)
            snippets = re.findall(snippet_pattern, html)
            
            for i, (url, title) in enumerate(urls[:num_results]):
                snippet = snippets[i] if i < len(snippets) else ""
                
                # Clean URL (DuckDuckGo redirects)
                if "uddg=" in url:
                    url_match = re.search(r'uddg=([^&]+)', url)
                    if url_match:
                        from urllib.parse import unquote
                        url = unquote(url_match.group(1))
                
                doc = RetrievedDocument(
                    id="",
                    url=url,
                    title=title.strip(),
                    content="",
                    snippet=snippet.strip(),
                    domain=urlparse(url).netloc,
                    timestamp=datetime.now(),
                    source_type="web_search",
                    relevance_score=1.0 - (i * 0.05),
                    metadata={"position": i + 1}
                )
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            print(f"DuckDuckGo search failed: {e}")
            return []
    
    async def close(self):
        await self.client.aclose()


class WebCrawler:
    """
    Crawls web pages to extract content.
    Uses the existing browser controller when available.
    """
    
    def __init__(self, browser_controller=None):
        self.browser_controller = browser_controller
        self.client = httpx.AsyncClient(
            timeout=30.0,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            },
            follow_redirects=True
        )
    
    async def crawl(self, url: str) -> str:
        """Extract text content from a URL."""
        try:
            if self.browser_controller:
                # Use browser for JavaScript-heavy pages
                return await self._crawl_with_browser(url)
            else:
                return await self._crawl_simple(url)
        except Exception as e:
            print(f"Crawl failed for {url}: {e}")
            return ""
    
    async def _crawl_simple(self, url: str) -> str:
        """Simple HTTP crawl for static pages."""
        try:
            response = await self.client.get(url)
            response.raise_for_status()
            html = response.text
            
            # Extract text content
            return self._extract_text(html)
        except Exception:
            return ""
    
    async def _crawl_with_browser(self, url: str) -> str:
        """Crawl using Playwright browser for JS-heavy pages."""
        try:
            await self.browser_controller.ensure_session()
            await self.browser_controller.page.goto(url, timeout=15000)
            await asyncio.sleep(1)  # Wait for JS to render
            
            # Get text content
            content = await self.browser_controller.page.evaluate("""
                () => {
                    // Remove scripts, styles, nav, footer
                    const elementsToRemove = document.querySelectorAll(
                        'script, style, nav, footer, header, aside, .ad, .advertisement, .sidebar'
                    );
                    elementsToRemove.forEach(el => el.remove());
                    
                    // Get main content
                    const main = document.querySelector('main, article, .content, #content, .post');
                    if (main) return main.innerText;
                    return document.body.innerText;
                }
            """)
            return content or ""
        except Exception as e:
            print(f"Browser crawl failed: {e}")
            return await self._crawl_simple(url)
    
    def _extract_text(self, html: str) -> str:
        """Extract readable text from HTML."""
        # Remove scripts and styles
        html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', html)
        
        # Decode entities
        text = re.sub(r'&nbsp;', ' ', text)
        text = re.sub(r'&amp;', '&', text)
        text = re.sub(r'&lt;', '<', text)
        text = re.sub(r'&gt;', '>', text)
        text = re.sub(r'&quot;', '"', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    async def close(self):
        await self.client.aclose()


class DenseRetriever:
    """
    Dense retrieval using embeddings and vector similarity.
    Uses in-memory storage for simplicity (can be replaced with vector DB).
    """
    
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        self.embeddings = OpenAIEmbeddings(
            api_key=api_key,
            model=model
        )
        self.document_store: Dict[str, Dict] = {}  # id -> {embedding, document}
    
    async def add_documents(self, documents: List[RetrievedDocument]) -> None:
        """Add documents to the vector store."""
        texts = [f"{doc.title}\n{doc.snippet}\n{doc.content[:1000]}" for doc in documents]
        
        if not texts:
            return
        
        embeddings = await self.embeddings.aembed_documents(texts)
        
        for doc, embedding in zip(documents, embeddings):
            self.document_store[doc.id] = {
                "embedding": embedding,
                "document": doc
            }
    
    async def search(self, query: str, top_k: int = 10) -> List[RetrievedDocument]:
        """Search for similar documents using embedding similarity."""
        if not self.document_store:
            return []
        
        query_embedding = await self.embeddings.aembed_query(query)
        
        # Calculate similarities
        similarities = []
        for doc_id, data in self.document_store.items():
            similarity = self._cosine_similarity(query_embedding, data["embedding"])
            similarities.append((similarity, data["document"]))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        # Return top-k
        results = []
        for score, doc in similarities[:top_k]:
            doc.relevance_score = score
            doc.source_type = "vector_db"
            results.append(doc)
        
        return results
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    def clear(self):
        """Clear the document store."""
        self.document_store.clear()


class HybridRetriever:
    """
    Combines lexical (web search) and dense (embedding) retrieval
    for comprehensive document retrieval.
    """
    
    def __init__(
        self,
        api_key: str,
        serper_api_key: Optional[str] = None,
        browser_controller=None
    ):
        self.web_searcher = WebSearcher(serper_api_key)
        self.web_crawler = WebCrawler(browser_controller)
        self.dense_retriever = DenseRetriever(api_key)
        self.api_key = api_key
    
    async def retrieve(
        self,
        processed_query: ProcessedQuery,
        max_results: int = 10,
        crawl_content: bool = True
    ) -> List[RetrievedDocument]:
        """
        Perform hybrid retrieval:
        1. Web search for all expanded queries
        2. Crawl content from top results
        3. Add to dense retriever
        4. Re-retrieve using dense similarity
        5. Merge and deduplicate results
        """
        all_documents: Dict[str, RetrievedDocument] = {}
        
        # 1. Web search for all queries (parallel)
        search_tasks = []
        queries_to_search = [processed_query.normalized] + processed_query.expanded_queries[:3]
        
        for query in queries_to_search:
            search_tasks.append(self.web_searcher.search(query, num_results=5))
        
        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        # Collect all search results
        for results in search_results:
            if isinstance(results, list):
                for doc in results:
                    if doc.url and doc.url not in all_documents:
                        all_documents[doc.url] = doc
        
        # 2. Crawl content (parallel, limited concurrency)
        if crawl_content:
            docs_to_crawl = list(all_documents.values())[:15]  # Limit crawling
            crawl_tasks = [self.web_crawler.crawl(doc.url) for doc in docs_to_crawl]
            crawled_contents = await asyncio.gather(*crawl_tasks, return_exceptions=True)
            
            for doc, content in zip(docs_to_crawl, crawled_contents):
                if isinstance(content, str) and content:
                    doc.content = content[:5000]  # Limit content size
        
        # 3. Add documents to dense retriever
        documents_list = list(all_documents.values())
        await self.dense_retriever.add_documents(documents_list)
        
        # 4. Dense retrieval for semantic matching
        dense_results = await self.dense_retriever.search(
            processed_query.normalized,
            top_k=max_results
        )
        
        # 5. Merge results (prefer dense retrieval scores)
        final_results = []
        seen_urls = set()
        
        # Add dense results first (higher semantic relevance)
        for doc in dense_results:
            if doc.url not in seen_urls:
                seen_urls.add(doc.url)
                final_results.append(doc)
        
        # Add remaining web search results
        for doc in documents_list:
            if doc.url not in seen_urls and len(final_results) < max_results:
                seen_urls.add(doc.url)
                final_results.append(doc)
        
        return final_results[:max_results]
    
    async def close(self):
        """Clean up resources."""
        await self.web_searcher.close()
        await self.web_crawler.close()
        self.dense_retriever.clear()

