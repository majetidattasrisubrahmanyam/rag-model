"""
Research Paper RAG Pipeline
Uses ChromaDB (onnxruntime embeddings, no PyTorch) + Groq LLM.
No sentence-transformers / torchaudio dependency issues.
"""

import os
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Generator

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

# ─────────────────────────────────────────────────────────────────
CHROMA_DIR     = "research_chroma_store"
COLLECTION     = "research_papers"
CHUNK_SIZE     = 800
CHUNK_OVERLAP  = 150
TOP_K          = 5
LLM_MODEL      = "llama-3.1-70b-versatile"
# ─────────────────────────────────────────────────────────────────


class ResearchRAGPipeline:
    """
    End-to-end RAG pipeline for research papers using ChromaDB + Groq.
    ChromaDB uses onnxruntime for embeddings — no PyTorch required.
    """

    def __init__(self):
        # ChromaDB persistent client
        os.makedirs(CHROMA_DIR, exist_ok=True)
        self.client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.embed_fn = DefaultEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION,
            embedding_function=self.embed_fn,
            metadata={"hnsw:space": "cosine"},
        )
        self.loaded_papers: List[str] = self._get_loaded_papers()
        self.llm: ChatGroq | None = None
        self._init_llm()
        print(f"[INFO] ChromaDB ready — {self.collection.count()} vectors in store.")

    # ── LLM ──────────────────────────────────────────────────────

    def _init_llm(self):
        api_key = os.getenv("GROQ_API_KEY", "")
        if api_key and api_key != "your_groq_api_key_here":
            self.llm = ChatGroq(groq_api_key=api_key, model_name=LLM_MODEL)
            print(f"[INFO] Groq LLM ready: {LLM_MODEL}")
        else:
            print("[WARN] GROQ_API_KEY not set — enter key in the UI.")

    def set_api_key(self, api_key: str):
        """Dynamically update the Groq API key from the UI."""
        os.environ["GROQ_API_KEY"] = api_key
        self.llm = ChatGroq(groq_api_key=api_key, model_name=LLM_MODEL)
        print("[INFO] Groq LLM updated with new API key.")

    # ── Helpers ──────────────────────────────────────────────────

    def _get_loaded_papers(self) -> List[str]:
        """Return list of already-indexed paper names from ChromaDB metadata."""
        if self.collection.count() == 0:
            return []
        results = self.collection.get(limit=1000, include=["metadatas"])
        papers = sorted({m.get("paper", "") for m in results["metadatas"] if m.get("paper")})
        return papers

    def _chunk_id(self, text: str, source: str, page: int) -> str:
        h = hashlib.md5(f"{source}|{page}|{text[:80]}".encode()).hexdigest()
        return h

    # ── Indexing ─────────────────────────────────────────────────

    def load_and_index_papers(self, pdf_folder: str, progress_cb=None) -> Tuple[int, List[str]]:
        """
        Load all PDFs, chunk, embed via ChromaDB, persist.
        Returns (total_chunks, paper_names).
        """
        pdf_folder = Path(pdf_folder)
        pdf_files  = sorted(pdf_folder.glob("*.pdf"))

        if not pdf_files:
            raise FileNotFoundError(f"No PDF files in: {pdf_folder}")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        # Clear existing collection and recreate
        self.client.delete_collection(COLLECTION)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION,
            embedding_function=self.embed_fn,
            metadata={"hnsw:space": "cosine"},
        )

        total_chunks = 0
        paper_names  = []

        for i, pdf_path in enumerate(pdf_files):
            paper_name = pdf_path.stem
            paper_names.append(paper_name)
            print(f"[INFO] Loading: {pdf_path.name}")
            if progress_cb:
                progress_cb((i + 0.5) / len(pdf_files), desc=f"Processing {pdf_path.name}...")

            try:
                loader = PyPDFLoader(str(pdf_path))
                pages  = loader.load()
                chunks = splitter.split_documents(pages)

                texts     = [c.page_content for c in chunks]
                metadatas = [
                    {
                        "source": pdf_path.name,
                        "paper":  paper_name,
                        "page":   str(c.metadata.get("page", 0) + 1),
                    }
                    for c in chunks
                ]
                ids = [
                    self._chunk_id(c.page_content, pdf_path.name, c.metadata.get("page", 0))
                    for c in chunks
                ]

                # Add in batches of 100
                batch = 100
                for start in range(0, len(texts), batch):
                    self.collection.add(
                        documents=texts[start : start + batch],
                        metadatas=metadatas[start : start + batch],
                        ids=ids[start : start + batch],
                    )

                total_chunks += len(chunks)
                print(f"  → {len(chunks)} chunks from {len(pages)} pages")

            except Exception as e:
                print(f"[ERROR] {pdf_path.name}: {e}")

        self.loaded_papers = paper_names
        print(f"[INFO] Total vectors: {self.collection.count()}")
        return total_chunks, paper_names

    def index_ready(self) -> bool:
        return self.collection.count() > 0

    def load_existing_index(self) -> bool:
        """Check if a persistent index already exists."""
        count = self.collection.count()
        if count > 0:
            self.loaded_papers = self._get_loaded_papers()
            print(f"[INFO] Existing index loaded: {count} vectors")
            return True
        return False

    # ── Retrieval ─────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = TOP_K) -> List[Dict]:
        """Query ChromaDB and return top-K chunks with metadata."""
        if not self.index_ready():
            return []
        results = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        chunks = []
        seen   = set()
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            key = doc[:80]
            if key not in seen:
                seen.add(key)
                chunks.append({
                    "text":   doc,
                    "source": meta.get("source", ""),
                    "paper":  meta.get("paper", ""),
                    "page":   meta.get("page", "?"),
                    "score":  float(dist),
                })
        return chunks

    def get_sources_for_query(self, query: str) -> List[Dict]:
        return self.retrieve(query, top_k=TOP_K)

    # ── Q&A ───────────────────────────────────────────────────────

    def answer_question(
        self,
        query: str,
        chat_history: List[Tuple[str, str]],
    ) -> Generator[str, None, None]:
        """Retrieve relevant chunks and stream an LLM answer."""
        if not self.index_ready():
            yield "⚠️ Papers not indexed yet. Please click **🔄 Index Papers** first."
            return
        if self.llm is None:
            yield "⚠️ Groq API key not set. Please enter your key in the sidebar and click **Save Key**."
            return

        chunks = self.retrieve(query, top_k=TOP_K)
        if not chunks:
            yield "I couldn't find relevant content in your papers. Try rephrasing your question."
            return

        # Build context
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[Source {i}: {chunk['source']}, Page {chunk['page']}]\n{chunk['text']}"
            )
        context = "\n\n---\n\n".join(context_parts)

        # Recent history (last 3 turns)
        history_text = ""
        for human_msg, ai_msg in chat_history[-3:]:
            if ai_msg:
                history_text += f"User: {human_msg}\nAssistant: {ai_msg}\n\n"

        system_prompt = """You are an expert research assistant specializing in analyzing academic papers.
Answer questions accurately based ONLY on the provided research paper excerpts.

Guidelines:
- Be clear, precise, and concise
- Always cite which paper and page number supports your answer
- If the answer spans multiple papers, synthesize them coherently  
- If the question cannot be answered from the context, say so clearly
- Use markdown formatting for readability (bold key terms, bullet points for lists)"""

        user_prompt = f"""{"Previous conversation:\n" + history_text if history_text else ""}
Research Paper Excerpts:
{context}

Question: {query}

Answer based on the papers above, citing sources (paper name + page number) for key points."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        try:
            full_response = ""
            for chunk in self.llm.stream(messages):
                if chunk.content:
                    full_response += chunk.content
                    yield full_response
        except Exception as e:
            yield f"❌ LLM Error: {str(e)}\n\nPlease verify your Groq API key is valid."
