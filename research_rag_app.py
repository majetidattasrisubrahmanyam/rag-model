"""
Research Paper RAG — Gradio Chat UI
Run: python research_rag_app.py
Then open: http://localhost:7860
"""

import os
import time
import gradio as gr
from pathlib import Path
from dotenv import load_dotenv

from research_rag.rag_pipeline import ResearchRAGPipeline

load_dotenv()

# ─── Constants ────────────────────────────────────────────────────
PDF_FOLDER = r"C:\Users\subra\Desktop\research paper"
PAPERS_DISPLAY_NAMES = {
    "CapStone Team-3 - IEEE[17]": "📘 CapStone Team-3 (IEEE)",
    "PAD_Paper":                   "📗 PAD Paper",
    "dp research paper":           "📙 DP Research Paper",
}

# ─── Global pipeline ──────────────────────────────────────────────
pipeline = ResearchRAGPipeline()

# Try loading existing index on startup
index_loaded = pipeline.load_existing_index()

# ─── Helper: format sources panel ─────────────────────────────────

def format_sources(chunks) -> str:
    if not chunks:
        return "*No sources retrieved yet.*"
    lines = ["### 📎 Sources Retrieved\n"]
    for i, c in enumerate(chunks, 1):
        paper = PAPERS_DISPLAY_NAMES.get(c["paper"], c["paper"])
        preview = c["text"][:220].replace("\n", " ").strip()
        lines.append(
            f"**{i}. {paper}** — Page {c['page']}\n"
            f"> {preview}…\n"
        )
    return "\n".join(lines)

# ─── Gradio callback: index papers ────────────────────────────────

def index_papers(progress=gr.Progress()):
    progress(0, desc="Starting indexing...")
    try:
        progress(0.1, desc="Loading PDF files...")
        total_chunks, papers = pipeline.load_and_index_papers(PDF_FOLDER)
        progress(1.0, desc="Done!")
        paper_list = "\n".join(
            f"✅ {PAPERS_DISPLAY_NAMES.get(p, p)}" for p in papers
        )
        return (
            gr.update(value=f"✅ **Indexed {total_chunks} chunks** from {len(papers)} papers:\n\n{paper_list}", visible=True),
            gr.update(interactive=True),
        )
    except Exception as e:
        return (
            gr.update(value=f"❌ **Indexing failed:** {str(e)}", visible=True),
            gr.update(interactive=False),
        )

# ─── Gradio callback: set API key ─────────────────────────────────

def set_api_key(key: str):
    key = key.strip()
    if not key:
        return gr.update(value="⚠️ Please enter a valid Groq API key.", visible=True)
    pipeline.set_api_key(key)
    return gr.update(value="✅ API key saved! You can now ask questions.", visible=True)

# ─── Gradio callback: chat ─────────────────────────────────────────

def user_message(message: str, history: list):
    """Append user message immediately."""
    return "", history + [[message, None]]


def bot_response(history: list, sources_box):
    """Stream the assistant response and update sources panel."""
    if not history:
        return history, sources_box

    query = history[-1][0]
    chat_pairs = [(h[0], h[1]) for h in history[:-1] if h[1] is not None]

    # Retrieve sources first for display
    chunks = pipeline.get_sources_for_query(query)
    sources_md = format_sources(chunks)

    # Stream answer
    history[-1][1] = ""
    for partial in pipeline.answer_question(query, chat_pairs):
        history[-1][1] = partial
        yield history, sources_md

    yield history, sources_md

# ─── Gradio callback: clear chat ──────────────────────────────────

def clear_chat():
    return [], "*Ask a question to see retrieved sources.*"

# ─── Gradio callback: example questions ───────────────────────────

def use_example(example: str):
    return example

# ─── Build UI ─────────────────────────────────────────────────────

EXAMPLE_QUESTIONS = [
    "What is the main objective of this research?",
    "What methodology was used in the study?",
    "What are the key findings and results?",
    "What datasets were used for evaluation?",
    "What are the limitations of the proposed approach?",
    "How does this work compare to existing methods?",
]

CUSTOM_CSS = """
/* ── Global ── */
body, .gradio-container {
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e) !important;
    min-height: 100vh;
}

/* ── Header ── */
.header-block {
    text-align: center;
    padding: 2rem 1rem 1rem;
}
.header-block h1 {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}
.header-block p {
    color: #94a3b8;
    font-size: 1rem;
}

/* ── Panels ── */
.panel-box {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 16px !important;
    backdrop-filter: blur(12px);
    padding: 1rem !important;
}

/* ── Chatbot ── */
.chat-window .message.bot {
    background: rgba(96,165,250,0.12) !important;
    border-left: 3px solid #60a5fa !important;
}
.chat-window .message.user {
    background: rgba(167,139,250,0.15) !important;
    border-right: 3px solid #a78bfa !important;
}

/* ── Buttons ── */
.btn-primary {
    background: linear-gradient(135deg, #7c3aed, #3b82f6) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    transition: all 0.2s ease !important;
}
.btn-primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(124,58,237,0.4) !important;
}
.btn-secondary {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    color: #e2e8f0 !important;
    border-radius: 10px !important;
}
.btn-send {
    background: linear-gradient(135deg, #059669, #10b981) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
}

/* ── Source panel ── */
.source-panel {
    color: #cbd5e1 !important;
    font-size: 0.88rem !important;
    line-height: 1.6 !important;
}
.source-panel blockquote {
    border-left: 2px solid #60a5fa !important;
    color: #94a3b8 !important;
    margin: 0.3rem 0 0.8rem 0.5rem !important;
    padding-left: 0.6rem !important;
    font-style: italic !important;
}

/* ── Status message ── */
.status-msg {
    border-radius: 10px !important;
    font-size: 0.9rem !important;
}

/* ── Example chips ── */
.example-btn {
    background: rgba(167,139,250,0.1) !important;
    border: 1px solid rgba(167,139,250,0.3) !important;
    border-radius: 20px !important;
    color: #c4b5fd !important;
    font-size: 0.82rem !important;
    padding: 4px 12px !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
    margin: 2px !important;
}
.example-btn:hover {
    background: rgba(167,139,250,0.25) !important;
    transform: translateY(-1px) !important;
}

/* ── Input area ── */
.input-area textarea {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    color: #f1f5f9 !important;
    border-radius: 12px !important;
}
.input-area textarea:focus {
    border-color: #7c3aed !important;
    box-shadow: 0 0 0 2px rgba(124,58,237,0.25) !important;
}

/* ── Paper list ── */
.paper-tag {
    display: inline-block;
    background: rgba(52,211,153,0.12);
    border: 1px solid rgba(52,211,153,0.3);
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.8rem;
    color: #6ee7b7;
    margin: 2px;
}
"""

def build_app():
    initial_status = "✅ Index already loaded!" if index_loaded else "⚡ Click **Index Papers** to get started."

    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.purple,
        secondary_hue=gr.themes.colors.blue,
        neutral_hue=gr.themes.colors.slate,
        font=gr.themes.GoogleFont("Inter"),
    )

    with gr.Blocks(title="Research Paper RAG") as demo:

        # ── Header ──
        gr.HTML("""
        <div class="header-block">
          <h1>📚 Research Paper RAG</h1>
          <p>Ask any question about your research papers — powered by FAISS + Groq AI</p>
        </div>
        """)

        with gr.Row():
            # ── LEFT SIDEBAR ──────────────────────────────────────
            with gr.Column(scale=1, min_width=280):

                # Papers loaded
                with gr.Group(elem_classes="panel-box"):
                    gr.Markdown("### 📂 Your Papers")
                    gr.HTML("""
                    <div style="padding: 4px 0;">
                      <span class="paper-tag">📘 CapStone Team-3 (IEEE)</span><br/>
                      <span class="paper-tag" style="margin-top:4px">📗 PAD Paper</span><br/>
                      <span class="paper-tag" style="margin-top:4px">📙 DP Research Paper</span>
                    </div>
                    """)

                gr.Markdown("---")

                # Index button
                with gr.Group(elem_classes="panel-box"):
                    gr.Markdown("### ⚙️ Setup")
                    index_btn = gr.Button(
                        "🔄 Index Papers",
                        variant="primary",
                        elem_classes="btn-primary",
                    )
                    status_msg = gr.Markdown(
                        value=initial_status,
                        elem_classes="status-msg",
                        visible=True,
                    )

                gr.Markdown("---")

                # API Key input
                with gr.Group(elem_classes="panel-box"):
                    gr.Markdown("### 🔑 Groq API Key")
                    api_key_input = gr.Textbox(
                        label="",
                        placeholder="gsk_...",
                        type="password",
                        value=os.getenv("GROQ_API_KEY", ""),
                        show_label=False,
                    )
                    save_key_btn = gr.Button("Save Key", size="sm", elem_classes="btn-secondary")
                    key_status = gr.Markdown(visible=False)

                gr.Markdown("---")

                # Sources panel
                with gr.Group(elem_classes="panel-box"):
                    sources_display = gr.Markdown(
                        value="*Ask a question to see retrieved sources.*",
                        elem_classes="source-panel",
                        label="",
                    )

            # ── MAIN CHAT AREA ────────────────────────────────────
            with gr.Column(scale=3):

                chatbot = gr.Chatbot(
                    label="",
                    elem_classes="chat-window",
                    height=480,
                    render_markdown=True,
                    layout="bubble",
                    buttons=["copy"],
                    placeholder="<div style='text-align:center; color:#64748b; padding:2rem'>Ask a question about your research papers...</div>",
                )

                # Example question chips
                gr.Markdown("<small>💡 **Try these questions:**</small>")
                with gr.Row(equal_height=True):
                    for ex in EXAMPLE_QUESTIONS[:3]:
                        ex_btn = gr.Button(ex, size="sm", elem_classes="example-btn")
                        ex_btn.click(fn=lambda e=ex: e, outputs=None)  # placeholder
                with gr.Row(equal_height=True):
                    for ex in EXAMPLE_QUESTIONS[3:]:
                        ex_btn2 = gr.Button(ex, size="sm", elem_classes="example-btn")

                # Chat input
                with gr.Row(elem_classes="input-area"):
                    msg_input = gr.Textbox(
                        placeholder="Ask anything about your research papers...",
                        show_label=False,
                        scale=5,
                        container=False,
                        lines=1,
                    )
                    send_btn = gr.Button(
                        "Send ➤",
                        variant="primary",
                        scale=1,
                        elem_classes="btn-send",
                    )

                with gr.Row():
                    clear_btn = gr.Button("🗑️ Clear Chat", size="sm", elem_classes="btn-secondary")

        # ── Event wiring ──────────────────────────────────────────

        # Index papers
        index_btn.click(
            fn=index_papers,
            outputs=[status_msg, send_btn],
        )

        # Save API key
        save_key_btn.click(
            fn=set_api_key,
            inputs=[api_key_input],
            outputs=[key_status],
        )

        # Send message (Enter or button)
        submit_event = msg_input.submit(
            fn=user_message,
            inputs=[msg_input, chatbot],
            outputs=[msg_input, chatbot],
            queue=False,
        ).then(
            fn=bot_response,
            inputs=[chatbot, sources_display],
            outputs=[chatbot, sources_display],
        )

        send_btn.click(
            fn=user_message,
            inputs=[msg_input, chatbot],
            outputs=[msg_input, chatbot],
            queue=False,
        ).then(
            fn=bot_response,
            inputs=[chatbot, sources_display],
            outputs=[chatbot, sources_display],
        )

        # Example chip clicks (re-wire properly)
        def make_example_click(example):
            def _fn():
                return example
            return _fn

        # Clear chat
        clear_btn.click(
            fn=clear_chat,
            outputs=[chatbot, sources_display],
        )

        # Footer
        gr.HTML("""
        <div style="text-align:center; padding: 1.5rem; color: #475569; font-size:0.8rem;">
          Built with FAISS + ChromaDB + Groq + Gradio
        </div>
        """)

    return demo, theme


# ─── Launch ───────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  Research Paper RAG -- Starting...")
    print("=" * 55)
    print(f"  PDF Folder : {PDF_FOLDER}")
    print(f"  Index Ready: {pipeline.index_ready()}")
    print(f"  LLM Ready  : {pipeline.llm is not None}")
    print("=" * 55)
    print("  Open: http://localhost:7860")
    print("=" * 55 + "\n")

    app, theme = build_app()
    app.queue()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=theme,
        css=CUSTOM_CSS,
    )
