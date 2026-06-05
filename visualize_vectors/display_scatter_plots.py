import numpy as np
import plotly.graph_objects as go
from sklearn.manifold import TSNE

DOC_TYPES = ["products", "employees", "contracts", "company"]
DOC_TYPE_COLORS = {
    "products": "#3b82f6",
    "employees": "#22c55e",
    "contracts": "#ef4444",
    "company": "#f97316",
}


def _extract_collection_data(collection):
    result = collection.get(include=["embeddings", "documents", "metadatas"])
    vectors = np.array(result["embeddings"])
    documents = result["documents"]
    metadatas = result["metadatas"]
    doc_types = [metadata.get("doc_type", "unknown") for metadata in metadatas]
    return vectors, documents, doc_types


def _run_tsne(vectors: np.ndarray, n_components: int) -> np.ndarray:
    n_samples = len(vectors)
    if n_samples < 2:
        raise ValueError("Need at least 2 vectors to generate a scatter plot.")

    perplexity = min(30, max(2, n_samples - 1))
    tsne = TSNE(n_components=n_components, random_state=42, perplexity=perplexity)
    return tsne.fit_transform(vectors)


def _build_scatter_traces(reduced_vectors, doc_types, documents, n_components: int):
    traces = []
    for doc_type in DOC_TYPES:
        indices = [i for i, t in enumerate(doc_types) if t == doc_type]
        if not indices:
            continue

        hover_text = [
            f"Type: {doc_types[i]}<br>Text: {documents[i][:100]}..."
            for i in indices
        ]
        coords = reduced_vectors[indices]

        if n_components == 2:
            trace = go.Scatter(
                x=coords[:, 0],
                y=coords[:, 1],
                mode="markers",
                name=doc_type.title(),
                marker=dict(size=6, color=DOC_TYPE_COLORS[doc_type], opacity=0.8),
                text=hover_text,
                hoverinfo="text",
            )
        else:
            trace = go.Scatter3d(
                x=coords[:, 0],
                y=coords[:, 1],
                z=coords[:, 2],
                mode="markers",
                name=doc_type.title(),
                marker=dict(size=4, color=DOC_TYPE_COLORS[doc_type], opacity=0.8),
                text=hover_text,
                hoverinfo="text",
            )
        traces.append(trace)

    return traces


def _build_scatter_figure(collection, n_components: int, title: str):
    vectors, documents, doc_types = _extract_collection_data(collection)
    reduced_vectors = _run_tsne(vectors, n_components)
    traces = _build_scatter_traces(reduced_vectors, doc_types, documents, n_components)

    fig = go.Figure(data=traces)
    if n_components == 2:
        fig.update_layout(
            title=title,
            xaxis_title="t-SNE 1",
            yaxis_title="t-SNE 2",
            width=900,
            height=600,
            margin=dict(r=20, b=10, l=10, t=40),
            legend=dict(title="Document Type"),
            template="plotly_white",
        )
    else:
        fig.update_layout(
            title=title,
            scene=dict(xaxis_title="t-SNE 1", yaxis_title="t-SNE 2", zaxis_title="t-SNE 3"),
            width=900,
            height=700,
            margin=dict(r=10, b=10, l=10, t=40),
            legend=dict(title="Document Type"),
            template="plotly_white",
        )

    return fig


def display_2d_scatter(collection):
    return _build_scatter_figure(
        collection,
        n_components=2,
        title="2D Chroma Vector Store Visualization",
    )


def display_3d_scatter(collection):
    return _build_scatter_figure(
        collection,
        n_components=3,
        title="3D Chroma Vector Store Visualization",
    )


def get_scatter_fig(collection, dim: str):
    """Return a Plotly figure for the requested scatter plot dimension."""
    if dim == "3D":
        return display_3d_scatter(collection)
    return display_2d_scatter(collection)
