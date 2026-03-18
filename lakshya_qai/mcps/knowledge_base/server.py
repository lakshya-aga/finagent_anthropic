"""MCP server for the Knowledge Base.

Provides semantic search over research papers, text documents, and
good practices stored in ChromaDB.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from lakshya_qai.config.settings import get_settings

logger = logging.getLogger(__name__)
_settings = get_settings()


class KnowledgeBaseStore:
    """ChromaDB-backed vector store for research documents and practices."""

    COLLECTION_NAME = "qai_knowledge_base"

    def __init__(self) -> None:
        self._client = None
        self._collection = None

    def _ensure_initialized(self):
        if self._client is None:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            persist_dir = str(_settings.chroma_persist_dir)
            self._client = chromadb.PersistentClient(
                path=persist_dir,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            self._collection = self._client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )

    def add_chunks(
        self,
        chunks: list[dict],
        source_id: str,
    ) -> int:
        """Add document chunks to the vector store.

        Args:
            chunks: List of dicts with keys: text, chunk_type, metadata.
            source_id: Unique identifier for the source document.

        Returns:
            Number of chunks added.
        """
        self._ensure_initialized()

        ids = []
        documents = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            chunk_id = f"{source_id}_chunk_{i}"
            ids.append(chunk_id)
            documents.append(chunk["text"])
            meta = chunk.get("metadata", {})
            meta["chunk_type"] = chunk.get("chunk_type", "other")
            meta["source_id"] = source_id
            # ChromaDB requires flat metadata values
            metadatas.append({
                k: json.dumps(v) if isinstance(v, (list, dict)) else str(v)
                for k, v in meta.items()
            })

        self._collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        logger.info("Added %d chunks from %s to knowledge base", len(ids), source_id)
        return len(ids)

    def search(
        self,
        query: str,
        top_k: int = 5,
        chunk_type: str | None = None,
    ) -> list[dict]:
        """Semantic search over the knowledge base.

        Args:
            query: Natural language query.
            top_k: Number of results.
            chunk_type: Optional filter by chunk type.

        Returns:
            List of dicts with keys: text, metadata, distance.
        """
        self._ensure_initialized()

        where_filter = None
        if chunk_type:
            where_filter = {"chunk_type": chunk_type}

        results = self._collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where_filter,
        )

        output = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                output.append({
                    "text": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else None,
                })
        return output

    def count(self) -> int:
        """Return total number of chunks in the store."""
        self._ensure_initialized()
        return self._collection.count()


# ── Singleton ────────────────────────────────────────────────────────────
_store = KnowledgeBaseStore()


# ── MCP Tool Functions ───────────────────────────────────────────────────

async def search_knowledge_base(args: dict) -> dict:
    """Search the knowledge base for relevant research content.

    Args (via MCP):
        query: Natural language search query.
        top_k: Max results (default 5).
        chunk_type: Filter by type — "abstract", "methodology", "table", "equation", etc.
    """
    query = args.get("query", "")
    if not query:
        return {"content": [{"type": "text", "text": "Error: 'query' required."}]}

    top_k = args.get("top_k", 5)
    chunk_type = args.get("chunk_type", None)

    results = _store.search(query, top_k=top_k, chunk_type=chunk_type)

    if not results:
        return {"content": [{"type": "text", "text": f"No results for '{query}'."}]}

    lines = [f"Found {len(results)} result(s):\n"]
    for i, r in enumerate(results, 1):
        meta = r["metadata"]
        lines.append(f"### Result {i} (distance: {r['distance']:.4f})")
        lines.append(f"**Type:** {meta.get('chunk_type', 'unknown')}")
        if meta.get("source_id"):
            lines.append(f"**Source:** {meta['source_id']}")
        if meta.get("section_title"):
            lines.append(f"**Section:** {meta['section_title']}")
        lines.append(f"\n{r['text'][:500]}{'...' if len(r['text']) > 500 else ''}")
        lines.append("---")

    return {"content": [{"type": "text", "text": "\n".join(lines)}]}


async def get_knowledge_base_stats(args: dict) -> dict:
    """Get statistics about the knowledge base."""
    count = _store.count()
    return {"content": [{"type": "text", "text": f"Knowledge base contains {count} chunks."}]}


def create_knowledge_base_mcp():
    """Create the MCP server for the knowledge base."""
    from claude_agent_sdk import tool, create_sdk_mcp_server

    return create_sdk_mcp_server(
        name="qai_knowledge_base",
        tools=[
            tool("search_knowledge_base", "Semantic search over research papers and documents", {"query": str, "top_k": int, "chunk_type": str})(search_knowledge_base),
            tool("get_knowledge_base_stats", "Get knowledge base statistics", {})(get_knowledge_base_stats),
        ],
    )
