import gradio as gr

from rag_service import (
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_MODEL_OPTIONS,
    KB_FILE_EMPTY_STATE,
    PipelineState,
    build_pipeline,
    format_kb_file_preview,
    get_chunk_choices,
    get_chunk_info,
    get_overview_stats,
    get_plot,
    get_vector_stats,
    list_kb_files,
    prewarm_pipeline,
    rag_query,
    validate_embedding_model,
)

CUSTOM_CSS = """
.gradio-container {
    max-width: 1280px !important;
    margin: 0 auto;
}
.hero-header {
    text-align: center;
    padding: 2.5rem 2rem 1.75rem;
    background: #1e293b;
    border-radius: 12px;
    margin-bottom: 1.25rem;
}
.hero-header h1 {
    font-size: 2.75rem;
    font-weight: 800;
    margin: 0 0 0.75rem 0;
    color: #f8fafc;
    line-height: 1.15;
}
.hero-header .subtitle {
    color: #94a3b8;
    font-size: 1.1rem;
    max-width: 720px;
    margin: 0 auto 1rem auto;
    line-height: 1.5;
}
.hero-header .tech-stack {
    color: #64748b;
    font-size: 0.9rem;
    margin: 0;
    line-height: 1.6;
}
.hero-header .tech-stack strong {
    color: #94a3b8;
    font-weight: 600;
}
.toolbar-row {
    margin-bottom: 0.5rem;
}
.build-status {
    color: #94a3b8;
    font-size: 0.9rem;
    margin-bottom: 0.75rem;
}
.file-preview-panel {
    min-height: 60vh;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 1.25rem;
    background: #0f172a;
    overflow-y: auto;
}
.chunk-preview-panel {
    min-height: 50vh;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 1.25rem;
    background: #0f172a;
    overflow-y: auto;
}
.footer-note {
    text-align: center;
    color: #64748b;
    font-size: 0.85rem;
    margin-top: 1.5rem;
    padding-bottom: 1rem;
}
"""

_pipeline = PipelineState()


def get_pipeline() -> PipelineState:
    return _pipeline


def set_pipeline(state: PipelineState) -> None:
    global _pipeline
    _pipeline = state


def on_kb_file_change(file_path: str | None) -> str:
    return format_kb_file_preview(file_path)


def _chunk_updates(state: PipelineState, index: int) -> tuple:
    choices = get_chunk_choices(state.chunks)
    max_index = len(state.chunks) - 1 if state.chunks else 0
    clamped = max(0, min(index, max_index))
    return (
        gr.update(value=clamped),
        gr.update(value=clamped + 1),
        get_chunk_info(state.chunks, clamped),
    )


def on_build_vector_store(model_label: str, progress=gr.Progress()) -> tuple:
    progress(0, desc="Validating embedding model...")
    error = validate_embedding_model(model_label)
    if error:
        gr.Warning(error)
        set_pipeline(PipelineState())
        return (
            f"Build failed: {error}",
            gr.update(choices=[], value=None, interactive=False),
            gr.update(maximum=1, value=1, interactive=False),
            get_vector_stats(None),
            None,
            get_chunk_info(None, 0),
        )

    progress(0.2, desc="Chunking knowledge base...")
    state, build_error = build_pipeline(model_label)
    if build_error:
        gr.Warning(build_error)
        set_pipeline(PipelineState())
        return (
            f"Build failed: {build_error}",
            gr.update(choices=[], value=None, interactive=False),
            gr.update(maximum=1, value=1, interactive=False),
            get_vector_stats(None),
            None,
            get_chunk_info(None, 0),
        )

    set_pipeline(state)
    progress(0.8, desc="Generating visualization...")
    chunk_choices = get_chunk_choices(state.chunks)
    first_chunk_index = chunk_choices[0][1] if chunk_choices else 0
    plot = get_plot(state.vectorstore, "2D")
    chunk_count = len(state.chunks) if state.chunks else 1

    progress(1.0, desc="Complete")
    return (
        f"Vector store built with **{model_label}** ({len(state.chunks):,} chunks).",
        gr.update(choices=chunk_choices, value=first_chunk_index, interactive=True),
        gr.update(maximum=chunk_count, value=first_chunk_index + 1, interactive=True),
        get_vector_stats(state.vectorstore),
        plot,
        get_chunk_info(state.chunks, first_chunk_index),
    )


def on_plot_dim_change(dim: str):
    state = get_pipeline()
    if state.vectorstore is None:
        return None
    return get_plot(state.vectorstore, dim)


def on_chunk_change(chunk_index: int | None):
    state = get_pipeline()
    index = 0 if chunk_index is None else chunk_index
    return gr.update(value=index + 1), get_chunk_info(state.chunks, index)


def on_chunk_slider_change(slider_value: int):
    state = get_pipeline()
    index = max(0, int(slider_value) - 1)
    return _chunk_updates(state, index)


def on_chunk_nav(direction: str, current_index: int | None):
    state = get_pipeline()
    if not state.chunks:
        return gr.update(), gr.update(), get_chunk_info(None, 0)

    index = current_index if current_index is not None else 0
    if direction == "prev":
        index = max(0, index - 1)
    else:
        index = min(len(state.chunks) - 1, index + 1)

    return _chunk_updates(state, index)


def on_ask_question(question: str):
    state = get_pipeline()
    answer, sources = rag_query(state.vectorstore, question)
    if not sources and "OPENAI_API_KEY" in answer:
        gr.Warning(answer)
    return answer, sources


def create_demo(prewarm: bool = True) -> gr.Blocks:
    kb_files = list_kb_files()
    if prewarm:
        set_pipeline(prewarm_pipeline())

    initial_state = get_pipeline()
    initial_chunk_choices = get_chunk_choices(initial_state.chunks)
    initial_chunk_index = initial_chunk_choices[0][1] if initial_chunk_choices else None
    initial_chunk_count = len(initial_state.chunks) if initial_state.chunks else 1
    initial_plot = get_plot(initial_state.vectorstore, "2D") if initial_state.vectorstore else None
    initial_status = (
        f"Pre-built vector store with **{DEFAULT_EMBEDDING_MODEL}** "
        f"({len(initial_state.chunks):,} chunks)."
        if initial_state.vectorstore
        else "Click **Build Vector Store** to chunk documents and create embeddings."
    )

    with gr.Blocks(title="Insurellm RAG Explorer") as demo:
        gr.HTML(
            """
            <div class="hero-header">
                <h1>Insurellm RAG Knowledge Base Explorer</h1>
                <p class="subtitle">
                    Explore document ingestion, chunking, embeddings, vector visualization,
                    and semantic Q&amp;A against a fictional insurance company knowledge base.
                </p>
                <p class="tech-stack">
                    <strong>Built with:</strong>
                    Python &middot; Gradio &middot; ChromaDB &middot;
                    HuggingFace sentence-transformers &middot; OpenAI GPT (Q&amp;A) &middot;
                    Plotly/t-SNE &middot; LangChain document loaders &amp; text splitter
                </p>
            </div>
            """
        )

        with gr.Row(elem_classes=["toolbar-row"]):
            embedding_dropdown = gr.Dropdown(
                choices=list(EMBEDDING_MODEL_OPTIONS.keys()),
                value=DEFAULT_EMBEDDING_MODEL,
                label="Embedding Model",
                scale=3,
            )
            build_button = gr.Button("Build Vector Store", variant="primary", scale=1)

        build_status = gr.Markdown(value=initial_status, elem_classes=["build-status"])

        with gr.Tabs():
            with gr.Tab("Overview"):
                overview_stats = gr.Markdown(value=get_overview_stats())
                kb_file_dropdown = gr.Dropdown(
                    choices=kb_files,
                    value=None,
                    label="Knowledge Base File",
                    info="Select a document to preview the full file below.",
                    allow_custom_value=False,
                )
                kb_file_preview = gr.Markdown(
                    value=KB_FILE_EMPTY_STATE,
                    elem_classes=["file-preview-panel"],
                )

            with gr.Tab("Embeddings"):
                vector_stats = gr.Markdown(value=get_vector_stats(initial_state.vectorstore))
                plot_dim_dropdown = gr.Dropdown(
                    choices=["2D", "3D"],
                    value="2D",
                    label="Scatter Plot View",
                    info="t-SNE projection of embedding vectors.",
                )
                scatter_plot = gr.Plot(value=initial_plot, label="Vector Space")

            with gr.Tab("Chunks"):
                with gr.Row():
                    chunk_dropdown = gr.Dropdown(
                        choices=initial_chunk_choices,
                        value=initial_chunk_index,
                        label="Chunk",
                        info="Select a chunk to inspect.",
                        interactive=bool(initial_chunk_choices),
                        scale=4,
                    )
                    prev_button = gr.Button("◀ Prev", scale=1)
                    next_button = gr.Button("Next ▶", scale=1)
                chunk_slider = gr.Slider(
                    minimum=1,
                    maximum=initial_chunk_count,
                    value=(initial_chunk_index or 0) + 1,
                    step=1,
                    label="Chunk position",
                    interactive=bool(initial_chunk_choices),
                )
                chunk_preview = gr.Markdown(
                    value=get_chunk_info(initial_state.chunks, initial_chunk_index or 0),
                    elem_classes=["chunk-preview-panel"],
                )

            with gr.Tab("Ask (RAG)"):
                gr.Markdown(
                    "Ask questions against the vector store. "
                    "Answers are grounded in retrieved chunks and cite source files."
                )
                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="What products does Insurellm offer?",
                    lines=2,
                )
                ask_button = gr.Button("Ask", variant="primary")
                gr.Examples(
                    examples=[
                        "What products does Insurellm offer?",
                        "Who is the CEO of Insurellm?",
                        "What is Insurellm's company culture?",
                    ],
                    inputs=question_input,
                )
                answer_output = gr.Markdown(label="Answer")
                sources_output = gr.Markdown(label="Retrieved Sources")

        gr.HTML(
            """
            <div class="footer-note">
                Built for portfolio demonstration
            </div>
            """
        )

        kb_file_dropdown.change(
            on_kb_file_change,
            inputs=[kb_file_dropdown],
            outputs=[kb_file_preview],
        )

        build_button.click(
            on_build_vector_store,
            inputs=[embedding_dropdown],
            outputs=[
                build_status,
                chunk_dropdown,
                chunk_slider,
                vector_stats,
                scatter_plot,
                chunk_preview,
            ],
        )

        plot_dim_dropdown.change(
            on_plot_dim_change,
            inputs=[plot_dim_dropdown],
            outputs=[scatter_plot],
        )

        chunk_dropdown.change(
            on_chunk_change,
            inputs=[chunk_dropdown],
            outputs=[chunk_slider, chunk_preview],
        )

        chunk_slider.change(
            on_chunk_slider_change,
            inputs=[chunk_slider],
            outputs=[chunk_dropdown, chunk_slider, chunk_preview],
        )

        prev_button.click(
            lambda idx: on_chunk_nav("prev", idx),
            inputs=[chunk_dropdown],
            outputs=[chunk_dropdown, chunk_slider, chunk_preview],
        )
        next_button.click(
            lambda idx: on_chunk_nav("next", idx),
            inputs=[chunk_dropdown],
            outputs=[chunk_dropdown, chunk_slider, chunk_preview],
        )

        ask_button.click(
            on_ask_question,
            inputs=[question_input],
            outputs=[answer_output, sources_output],
        )
        question_input.submit(
            on_ask_question,
            inputs=[question_input],
            outputs=[answer_output, sources_output],
        )

    return demo


THEME = gr.themes.Base(
    primary_hue="slate",
    secondary_hue="blue",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
)

demo = create_demo(prewarm=True)


def launch_demo(**kwargs):
    return demo.queue().launch(theme=THEME, css=CUSTOM_CSS, **kwargs)


if __name__ == "__main__":
    launch_demo()
