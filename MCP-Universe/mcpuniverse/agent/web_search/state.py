"""
Web Search State Manager.

This module defines the state management classes for the multi-agent web search system.
The WebSearchState class tracks all data accumulated during the search process.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class SearchResult:
    """Single search result from SERP."""
    position: int
    title: str
    link: str
    snippet: str
    displayed_link: str = ""
    date: Optional[str] = None


@dataclass
class KnowledgeGraph:
    """Knowledge graph data from search."""
    title: str
    entity_type: str
    description: str
    facts: Dict[str, str] = field(default_factory=dict)
    source: str = ""


@dataclass
class FetchedContent:
    """Content fetched from a URL."""
    url: str
    title: str
    content: str
    fetch_time: datetime = field(default_factory=datetime.now)
    content_type: str = "text/html"


@dataclass
class ExtractedFact:
    """A verified fact with sources."""
    statement: str
    value: Optional[Any] = None
    unit: Optional[str] = None
    confidence: float = 0.0
    sources: List[str] = field(default_factory=list)
    verified: bool = False


@dataclass
class WebSearchState:
    """
    Central state manager for web search workflow.
    Tracks all data accumulated during the search process.
    """
    original_query: str = ""
    queries: List[str] = field(default_factory=list)
    serp_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    fetched_content: Dict[str, FetchedContent] = field(default_factory=dict)
    extracted_facts: List[ExtractedFact] = field(default_factory=list)
    citations: Dict[str, List[Dict[str, str]]] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)
    iteration_count: int = 0

    def add_query(self, query: str):
        """Add a search query to the state."""
        if query and query not in self.queries:
            self.queries.append(query)

    def add_serp_result(self, query: str, result: Dict[str, Any]):
        """Store SERP results for a query."""
        self.serp_results[query] = result

    def add_fetched_content(self, url: str, content: FetchedContent):
        """Store fetched page content."""
        self.fetched_content[url] = content

    def add_fact(self, fact: ExtractedFact):
        """Add an extracted fact."""
        self.extracted_facts.append(fact)

    def add_history_entry(self, thought: str, action: str, observation: str):
        """Record a Thought-Action-Observation entry."""
        self.history.append({
            "iteration": self.iteration_count,
            "thought": thought,
            "action": action,
            "observation": observation
        })

    def get_context_summary(self) -> str:
        """Generate a summary of current state for context."""
        summary_parts = [
            f"Queries executed: {len(self.queries)}",
            f"SERP results cached: {len(self.serp_results)}",
            f"Pages fetched: {len(self.fetched_content)}",
            f"Facts extracted: {len(self.extracted_facts)}"
        ]
        if self.queries:
            summary_parts.append(f"Recent queries: {self.queries[-3:]}")
        return "\n".join(summary_parts)

    def get_all_sources(self) -> List[str]:
        """Get all source URLs from fetched content."""
        return list(self.fetched_content.keys())

    def get_verified_facts(self) -> List[ExtractedFact]:
        """Get only verified facts."""
        return [f for f in self.extracted_facts if f.verified]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to dictionary."""
        return {
            "original_query": self.original_query,
            "queries": self.queries,
            "serp_results": self.serp_results,
            "fetched_content": {
                k: {
                    "url": v.url,
                    "title": v.title,
                    "content": v.content[:500] + "..." if len(v.content) > 500 else v.content,
                    "fetch_time": str(v.fetch_time),
                    "content_type": v.content_type
                }
                for k, v in self.fetched_content.items()
            },
            "extracted_facts": [
                {
                    "statement": f.statement,
                    "value": f.value,
                    "unit": f.unit,
                    "confidence": f.confidence,
                    "sources": f.sources,
                    "verified": f.verified
                }
                for f in self.extracted_facts
            ],
            "history": self.history,
            "iteration_count": self.iteration_count
        }

    def reset(self):
        """Reset the state to initial values."""
        self.queries = []
        self.serp_results = {}
        self.fetched_content = {}
        self.extracted_facts = []
        self.citations = {}
        self.history = []
        self.iteration_count = 0
