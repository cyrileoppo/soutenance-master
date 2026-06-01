from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import umap

from utils.constants import EMBEDDING_PROJECTION_NEIGHBORS, RANDOM_STATE


def compute_projection(dataset: pd.DataFrame, profile_embeddings, color_by: str) -> pd.DataFrame:
    reducer = umap.UMAP(n_neighbors=EMBEDDING_PROJECTION_NEIGHBORS, min_dist=0.15, random_state=RANDOM_STATE)
    coords = reducer.fit_transform(profile_embeddings)
    projection = dataset.copy()
    projection['umap_x'] = coords[:, 0]
    projection['umap_y'] = coords[:, 1]
    projection['color_value'] = projection[color_by].astype(str)
    return projection


def build_embedding_figure(projection: pd.DataFrame):
    fig = px.scatter(
        projection,
        x='umap_x',
        y='umap_y',
        color='color_value',
        hover_name='title',
        hover_data={
            'propertyType': True,
            'bedrooms': True,
            'bathrooms': True,
            'price': True,
            'umap_x': False,
            'umap_y': False,
            'color_value': False,
        },
        custom_data=['listing_id'],
        template='plotly_dark',
        opacity=0.88,
        height=620,
    )
    fig.update_traces(marker={'size': 14, 'line': {'width': 0}}, selector={'mode': 'markers'})
    fig.update_layout(
        margin={'l': 0, 'r': 0, 't': 16, 'b': 0},
        legend_title_text='',
        xaxis_title='Dimension UMAP 1',
        yaxis_title='Dimension UMAP 2',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,23,42,0.35)',
    )
    return fig
