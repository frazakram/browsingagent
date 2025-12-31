"""
Citation Formatter - Formats responses with inline citations and source cards

Implements:
- Inline citation formatting
- Source card generation
- Clickable citation links
- Quote extraction
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Dict, Optional

from .llm_pipeline import GeneratedResponse
from .context_builder import BuiltContext


@dataclass
class SourceCard:
    """A formatted source card for display."""
    id: int
    title: str
    url: str
    domain: str
    snippet: str
    favicon_url: str
    relevance_score: float


@dataclass  
class FormattedResponse:
    """Final formatted response with citations."""
    content: str  # Response with inline citations
    content_html: str  # HTML formatted version
    content_markdown: str  # Markdown formatted version
    source_cards: List[SourceCard]
    citations_map: Dict[int, str]  # citation_id -> url
    confidence: float
    verification_status: str


class CitationFormatter:
    """
    Formats responses with proper citations and source cards.
    
    Features:
    - Inline citation formatting [1], [2]
    - Superscript citations for HTML
    - Source cards with snippets
    - Favicon/domain display
    - Quote extraction and highlighting
    """
    
    def __init__(self):
        pass
    
    def format(
        self,
        response: GeneratedResponse,
        context: BuiltContext
    ) -> FormattedResponse:
        """
        Format the generated response with citations.
        
        Returns FormattedResponse with multiple format options.
        """
        # Build source cards
        source_cards = self._build_source_cards(context)
        
        # Build citations map
        citations_map = {s.id: s.url for s in source_cards}
        
        # Format content for different outputs
        content_plain = self._format_plain(response.content)
        content_html = self._format_html(response.content, citations_map)
        content_markdown = self._format_markdown(response.content, citations_map)
        
        return FormattedResponse(
            content=content_plain,
            content_html=content_html,
            content_markdown=content_markdown,
            source_cards=source_cards,
            citations_map=citations_map,
            confidence=response.confidence_score,
            verification_status=response.verification_status.value
        )
    
    def _build_source_cards(self, context: BuiltContext) -> List[SourceCard]:
        """Build source cards from context."""
        cards = []
        
        for source in context.sources:
            # Find matching passage for snippet
            snippet = ""
            for rp in context.passages_used:
                if rp.passage.url == source['url']:
                    snippet = rp.passage.content[:200] + "..."
                    break
            
            # Generate favicon URL (using Google's favicon service)
            domain = source['domain']
            favicon_url = f"https://www.google.com/s2/favicons?domain={domain}&sz=32"
            
            card = SourceCard(
                id=source['id'],
                title=source['title'],
                url=source['url'],
                domain=domain,
                snippet=snippet,
                favicon_url=favicon_url,
                relevance_score=source.get('relevance', 0.0)
            )
            cards.append(card)
        
        return cards
    
    def _format_plain(self, content: str) -> str:
        """Format for plain text output."""
        # Keep citations as-is
        return content.strip()
    
    def _format_html(self, content: str, citations_map: Dict[int, str]) -> str:
        """Format for HTML output with clickable citations."""
        html = content
        
        # Escape HTML entities (but preserve our processing)
        html = html.replace("&", "&amp;")
        html = html.replace("<", "&lt;")
        html = html.replace(">", "&gt;")
        
        # Convert newlines to HTML
        html = html.replace("\n\n", "</p><p>")
        html = html.replace("\n", "<br>")
        html = f"<p>{html}</p>"
        
        # Convert citations to superscript links
        def citation_to_link(match):
            citation_id = int(match.group(1))
            url = citations_map.get(citation_id, "#")
            return f'<sup><a href="{url}" target="_blank" class="citation" data-citation="{citation_id}">[{citation_id}]</a></sup>'
        
        html = re.sub(r'\[(\d+)\]', citation_to_link, html)
        
        # Convert bullet points
        html = re.sub(r'<br>•\s*', '<br>• ', html)
        html = re.sub(r'<br>-\s+', '<br>• ', html)
        
        # Wrap in container
        html = f'<div class="rag-response">{html}</div>'
        
        return html
    
    def _format_markdown(self, content: str, citations_map: Dict[int, str]) -> str:
        """Format for Markdown output."""
        md = content
        
        # Convert citations to markdown links
        def citation_to_md(match):
            citation_id = int(match.group(1))
            url = citations_map.get(citation_id, "#")
            return f'[[{citation_id}]]({url})'
        
        md = re.sub(r'\[(\d+)\]', citation_to_md, md)
        
        # Add source list at the end
        md += "\n\n---\n\n**Sources:**\n"
        for cid, url in sorted(citations_map.items()):
            md += f"- [{cid}] {url}\n"
        
        return md
    
    def extract_quotes(self, content: str, passages: List) -> List[Dict]:
        """
        Extract quoted snippets that could be highlighted.
        
        Returns list of {quote, source_id, passage_match} dicts.
        """
        quotes = []
        
        # Find quoted text
        quote_pattern = r'"([^"]+)"'
        matches = re.findall(quote_pattern, content)
        
        for quote in matches:
            if len(quote) > 20:  # Only significant quotes
                # Try to find matching passage
                for rp in passages:
                    if quote.lower() in rp.passage.content.lower():
                        quotes.append({
                            "quote": quote,
                            "source_id": rp.rank,
                            "url": rp.passage.url,
                            "matched": True
                        })
                        break
                else:
                    quotes.append({
                        "quote": quote,
                        "source_id": None,
                        "url": None,
                        "matched": False
                    })
        
        return quotes
    
    def generate_source_cards_html(self, source_cards: List[SourceCard]) -> str:
        """Generate HTML for source cards display."""
        cards_html = '<div class="source-cards">'
        
        for card in source_cards:
            cards_html += f'''
<div class="source-card" data-source-id="{card.id}">
    <div class="source-header">
        <img src="{card.favicon_url}" alt="" class="favicon">
        <span class="source-domain">{card.domain}</span>
        <span class="source-id">[{card.id}]</span>
    </div>
    <a href="{card.url}" target="_blank" class="source-title">{card.title}</a>
    <p class="source-snippet">{card.snippet}</p>
</div>
'''
        
        cards_html += '</div>'
        return cards_html
    
    def generate_source_cards_json(self, source_cards: List[SourceCard]) -> List[Dict]:
        """Generate JSON-serializable source cards."""
        return [
            {
                "id": card.id,
                "title": card.title,
                "url": card.url,
                "domain": card.domain,
                "snippet": card.snippet,
                "favicon_url": card.favicon_url,
                "relevance": card.relevance_score
            }
            for card in source_cards
        ]

