from __future__ import annotations

import pandas as pd
import streamlit as st


def render_embedding_plot(fig, key: str = 'embedding-space'):
    return st.plotly_chart(fig, use_container_width=True, key=key, on_select='rerun', selection_mode='points')
