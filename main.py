from rag_service import (
    DEFAULT_EMBEDDING_MODEL,
    build_pipeline,
    get_chunk_info,
    get_plot,
    get_vector_stats,
)
from visualize_vectors.display_scatter_plots import display_2d_scatter, display_3d_scatter


def main():
    state, error = build_pipeline(DEFAULT_EMBEDDING_MODEL)
    if error:
        print(f"Pipeline failed: {error}")
        return

    print(get_chunk_info(state.chunks, 0))
    print(get_vector_stats(state.vectorstore))

    display_2d_scatter(state.vectorstore._collection)
    display_3d_scatter(state.vectorstore._collection)
    get_plot(state.vectorstore, "2D")


if __name__ == "__main__":
    main()
