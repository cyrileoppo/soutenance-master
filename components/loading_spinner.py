from __future__ import annotations

import time
from collections.abc import Iterable

import streamlit as st


def render_pipeline(stages: Iterable[str], delay: float = 0.2) -> None:
    status = st.status('Semantic pipeline running', expanded=True)
    for stage in stages:
        status.write(stage)
        time.sleep(delay)
    status.update(label='Semantic pipeline complete', state='complete', expanded=False)
