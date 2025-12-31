"""
Query Processor - Query Normalization and Expansion

Handles:
- Query normalization (cleaning, lowercasing, etc.)
- Intent classification
- Query expansion (synonyms, related terms, sub-queries)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


class QueryIntent(Enum):
    """Classification of user query intent."""
    FACTUAL = "factual"           # Looking for facts/information
    COMPARISON = "comparison"      # Comparing products/options
    NAVIGATION = "navigation"      # Looking for a specific site/page
    TRANSACTION = "transaction"    # Looking to buy/do something
    EXPLORATION = "exploration"    # Open-ended research


@dataclass
class ProcessedQuery:
    """Result of query processing."""
    original: str
    normalized: str
    intent: QueryIntent
    expanded_queries: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    time_sensitivity: str = "none"  # none, recent, realtime


class QueryProcessor:
    """
    Processes user queries for optimal retrieval.
    
    Pipeline:
    1. Normalize query (clean, standardize)
    2. Classify intent
    3. Extract keywords and entities
    4. Expand into sub-queries for broader retrieval
    """
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            temperature=0.1,
        )
        
        self.expansion_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a query expansion expert. Given a user query, generate:
1. 3-5 expanded search queries that capture different aspects of the user's intent
2. Key entities mentioned (products, brands, locations, people)
3. Important keywords for search
4. Whether the query needs recent/realtime information

Respond in this exact format:
INTENT: [factual|comparison|navigation|transaction|exploration]
TIME_SENSITIVITY: [none|recent|realtime]
KEYWORDS: keyword1, keyword2, keyword3
ENTITIES: entity1, entity2
EXPANDED_QUERIES:
- query 1
- query 2
- query 3"""),
            ("human", "{query}")
        ])
    
    def normalize(self, query: str) -> str:
        """
        Normalize the query for consistent processing.
        
        - Strip whitespace
        - Remove excessive punctuation
        - Normalize unicode
        - Keep case for proper nouns (handled by LLM)
        """
        # Strip and normalize whitespace
        normalized = " ".join(query.split())
        
        # Remove multiple punctuation marks
        normalized = re.sub(r'[!?]{2,}', '?', normalized)
        normalized = re.sub(r'\.{2,}', '.', normalized)
        
        # Remove leading/trailing quotes if they wrap the whole query
        if (normalized.startswith('"') and normalized.endswith('"')) or \
           (normalized.startswith("'") and normalized.endswith("'")):
            normalized = normalized[1:-1]
        
        return normalized.strip()
    
    async def process(self, query: str) -> ProcessedQuery:
        """
        Full query processing pipeline.
        
        Returns ProcessedQuery with:
        - Normalized query
        - Classified intent
        - Expanded sub-queries
        - Extracted keywords and entities
        """
        normalized = self.normalize(query)
        
        # Use LLM for intelligent expansion
        try:
            chain = self.expansion_prompt | self.llm
            response = await chain.ainvoke({"query": normalized})
            parsed = self._parse_expansion_response(response.content)
            
            return ProcessedQuery(
                original=query,
                normalized=normalized,
                intent=parsed["intent"],
                expanded_queries=parsed["expanded_queries"],
                keywords=parsed["keywords"],
                entities=parsed["entities"],
                time_sensitivity=parsed["time_sensitivity"],
            )
        except Exception as e:
            # Fallback to basic processing if LLM fails
            return ProcessedQuery(
                original=query,
                normalized=normalized,
                intent=QueryIntent.FACTUAL,
                expanded_queries=[normalized],
                keywords=self._extract_basic_keywords(normalized),
                entities=[],
                time_sensitivity="none",
            )
    
    def _parse_expansion_response(self, response: str) -> dict:
        """Parse the LLM expansion response."""
        result = {
            "intent": QueryIntent.FACTUAL,
            "time_sensitivity": "none",
            "keywords": [],
            "entities": [],
            "expanded_queries": [],
        }
        
        lines = response.strip().split('\n')
        in_queries = False
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("INTENT:"):
                intent_str = line.replace("INTENT:", "").strip().lower()
                try:
                    result["intent"] = QueryIntent(intent_str)
                except ValueError:
                    result["intent"] = QueryIntent.FACTUAL
                    
            elif line.startswith("TIME_SENSITIVITY:"):
                result["time_sensitivity"] = line.replace("TIME_SENSITIVITY:", "").strip().lower()
                
            elif line.startswith("KEYWORDS:"):
                keywords = line.replace("KEYWORDS:", "").strip()
                result["keywords"] = [k.strip() for k in keywords.split(",") if k.strip()]
                
            elif line.startswith("ENTITIES:"):
                entities = line.replace("ENTITIES:", "").strip()
                result["entities"] = [e.strip() for e in entities.split(",") if e.strip()]
                
            elif line.startswith("EXPANDED_QUERIES:"):
                in_queries = True
                
            elif in_queries and line.startswith("-"):
                query = line[1:].strip()
                if query:
                    result["expanded_queries"].append(query)
        
        # Ensure we have at least one expanded query
        if not result["expanded_queries"]:
            result["expanded_queries"] = [lines[0] if lines else ""]
        
        return result
    
    def _extract_basic_keywords(self, query: str) -> List[str]:
        """Basic keyword extraction without LLM."""
        # Remove common stop words
        stop_words = {
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
            'ought', 'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
            'from', 'as', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'between', 'under', 'again', 'further', 'then',
            'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
            'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
            'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just',
            'and', 'but', 'if', 'or', 'because', 'until', 'while', 'what',
            'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am',
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
            'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
            'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
            'they', 'them', 'their', 'theirs', 'themselves', 'give', 'get',
            'find', 'show', 'tell', 'best', 'good', 'want', 'please', 'help',
        }
        
        # Tokenize and filter
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        return keywords[:10]  # Limit to top 10

