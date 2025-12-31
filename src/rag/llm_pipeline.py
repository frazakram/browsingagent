"""
LLM Pipeline - Multi-stage generation with verification

Implements:
- Draft generation (fast model)
- Refinement (larger model)  
- Verification (fact-checking)
- Hallucination detection
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from enum import Enum

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from .context_builder import BuiltContext


class VerificationStatus(Enum):
    """Status of fact verification."""
    VERIFIED = "verified"
    PARTIALLY_VERIFIED = "partially_verified"
    UNVERIFIED = "unverified"
    CONTRADICTED = "contradicted"


@dataclass
class GeneratedResponse:
    """Response from the LLM pipeline."""
    content: str
    draft_content: str
    refined_content: str
    verification_status: VerificationStatus
    verification_notes: List[str]
    citations_used: List[int]
    confidence_score: float
    metadata: Dict


class LLMPipeline:
    """
    Multi-stage LLM generation pipeline.
    
    Stages:
    1. Draft: Fast generation with smaller model
    2. Refine: Enhancement with larger model (optional)
    3. Verify: Fact-check against sources
    
    Features:
    - Conservative decoding (low temperature)
    - Hallucination detection
    - Citation verification
    - Confidence scoring
    """
    
    def __init__(
        self,
        api_key: str,
        draft_model: str = "gpt-4o-mini",
        refine_model: str = "gpt-4o-mini",  # Use gpt-4o for better quality
        verify_model: str = "gpt-4o-mini",
        enable_refinement: bool = False,  # Disabled by default for speed
        enable_verification: bool = True,
    ):
        self.enable_refinement = enable_refinement
        self.enable_verification = enable_verification
        
        # Draft model - fast, good enough quality
        self.draft_llm = ChatOpenAI(
            model=draft_model,
            api_key=api_key,
            temperature=0.1,  # Conservative
            max_tokens=2000,
        )
        
        # Refine model - higher quality
        if enable_refinement:
            self.refine_llm = ChatOpenAI(
                model=refine_model,
                api_key=api_key,
                temperature=0.1,
                max_tokens=2000,
            )
        
        # Verify model - fact-checking
        if enable_verification:
            self.verify_llm = ChatOpenAI(
                model=verify_model,
                api_key=api_key,
                temperature=0,  # Deterministic for verification
                max_tokens=1000,
            )
        
        # Prompts
        self.refine_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert editor improving AI-generated responses.

Your task:
1. Improve clarity and readability
2. Ensure all claims have proper citations
3. Fix any grammatical issues
4. Make the response more concise if needed
5. Preserve all factual content and citations

Keep the same citation format [1], [2], etc."""),
            ("human", "Original response:\n{draft}\n\nPlease improve this response while keeping all citations:")
        ])
        
        self.verify_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a fact-checker verifying claims against source material.

For each claim in the response, check if it's supported by the provided sources.

Respond in this format:
VERIFICATION_STATUS: [verified|partially_verified|unverified|contradicted]
CONFIDENCE: [0.0-1.0]
ISSUES:
- Issue 1 (if any)
- Issue 2 (if any)
UNSUPPORTED_CLAIMS:
- Claim 1 (if any)"""),
            ("human", """SOURCES:
{sources}

RESPONSE TO VERIFY:
{response}

Verify all claims in the response against the sources:""")
        ])
    
    async def generate(self, context: BuiltContext) -> GeneratedResponse:
        """
        Run the full generation pipeline.
        
        1. Generate draft
        2. Optionally refine
        3. Verify against sources
        """
        # Stage 1: Draft generation
        draft = await self._generate_draft(context)
        
        # Stage 2: Refinement (optional)
        if self.enable_refinement:
            refined = await self._refine_response(draft, context)
        else:
            refined = draft
        
        # Stage 3: Verification
        if self.enable_verification:
            verification = await self._verify_response(refined, context)
        else:
            verification = {
                "status": VerificationStatus.UNVERIFIED,
                "notes": [],
                "confidence": 0.7,
            }
        
        # Extract citations used
        citations_used = self._extract_citations(refined)
        
        return GeneratedResponse(
            content=refined,
            draft_content=draft,
            refined_content=refined,
            verification_status=verification["status"],
            verification_notes=verification["notes"],
            citations_used=citations_used,
            confidence_score=verification["confidence"],
            metadata={
                "draft_model": self.draft_llm.model_name,
                "refinement_enabled": self.enable_refinement,
                "verification_enabled": self.enable_verification,
            }
        )
    
    async def _generate_draft(self, context: BuiltContext) -> str:
        """Generate initial draft response."""
        messages = [
            {"role": "system", "content": context.system_prompt},
            {"role": "user", "content": context.user_prompt}
        ]
        
        response = await self.draft_llm.ainvoke(messages)
        return response.content
    
    async def _refine_response(self, draft: str, context: BuiltContext) -> str:
        """Refine the draft response."""
        try:
            chain = self.refine_prompt | self.refine_llm
            response = await chain.ainvoke({"draft": draft})
            return response.content
        except Exception as e:
            print(f"Refinement failed: {e}")
            return draft
    
    async def _verify_response(self, response: str, context: BuiltContext) -> Dict:
        """Verify response against sources."""
        try:
            # Build source summary for verification
            source_text = ""
            for rp in context.passages_used[:5]:  # Limit for efficiency
                source_text += f"\n[Source {rp.rank}] {rp.passage.title}:\n{rp.passage.content[:800]}\n"
            
            chain = self.verify_prompt | self.verify_llm
            result = await chain.ainvoke({
                "sources": source_text,
                "response": response
            })
            
            return self._parse_verification(result.content)
            
        except Exception as e:
            print(f"Verification failed: {e}")
            return {
                "status": VerificationStatus.UNVERIFIED,
                "notes": [f"Verification error: {str(e)}"],
                "confidence": 0.5,
            }
    
    def _parse_verification(self, verification_text: str) -> Dict:
        """Parse verification response."""
        result = {
            "status": VerificationStatus.PARTIALLY_VERIFIED,
            "notes": [],
            "confidence": 0.7,
        }
        
        lines = verification_text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("VERIFICATION_STATUS:"):
                status_str = line.replace("VERIFICATION_STATUS:", "").strip().lower()
                try:
                    result["status"] = VerificationStatus(status_str)
                except ValueError:
                    pass
            
            elif line.startswith("CONFIDENCE:"):
                try:
                    conf = float(line.replace("CONFIDENCE:", "").strip())
                    result["confidence"] = min(max(conf, 0.0), 1.0)
                except ValueError:
                    pass
            
            elif line.startswith("-") and "ISSUES" in verification_text[:verification_text.find(line)]:
                result["notes"].append(line[1:].strip())
        
        return result
    
    def _extract_citations(self, text: str) -> List[int]:
        """Extract citation numbers from text."""
        pattern = r'\[(\d+)\]'
        matches = re.findall(pattern, text)
        return sorted(set(int(m) for m in matches))
    
    async def regenerate_if_needed(
        self,
        response: GeneratedResponse,
        context: BuiltContext
    ) -> GeneratedResponse:
        """
        Regenerate response if verification failed.
        
        Called when verification finds contradictions or hallucinations.
        """
        if response.verification_status in [
            VerificationStatus.VERIFIED,
            VerificationStatus.PARTIALLY_VERIFIED
        ]:
            return response
        
        # Add verification feedback to context
        feedback = "\n".join(response.verification_notes)
        enhanced_user_prompt = f"""{context.user_prompt}

IMPORTANT: A previous response had issues:
{feedback}

Please provide a more accurate response, being careful to only state facts that are directly supported by the sources."""
        
        # Create new context with feedback
        enhanced_context = BuiltContext(
            system_prompt=context.system_prompt,
            user_prompt=enhanced_user_prompt,
            passages_used=context.passages_used,
            total_tokens_estimate=context.total_tokens_estimate + len(feedback) // 4,
            sources=context.sources
        )
        
        # Regenerate
        return await self.generate(enhanced_context)

