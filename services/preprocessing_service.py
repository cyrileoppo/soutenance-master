from __future__ import annotations

from typing import Any

import pandas as pd
from bs4 import BeautifulSoup

from utils.helpers import price_range_label


COLUMNS_TO_FILL = ['title', 'propertyType', 'sizeSqFeetMax', 'bedrooms', 'bathrooms', 'price']


def clean_html(html_text: Any) -> str:
    if not isinstance(html_text, str):
        return ''
    return BeautifulSoup(html_text, 'html.parser').get_text(separator=' ').strip()


def _format_decimal_string(value: str) -> str:
    return value.replace('.0', '') if value.endswith('.0') else value


def build_rich_anchor(row: pd.Series) -> str:
    return (
        row['title']
        + ' [ATTR] Type: '
        + row['propertyType']
        + ' [ATTR] Beds: '
        + row['bedrooms']
        + ' [ATTR] Baths: '
        + row['bathrooms']
        + ' [ATTR] Size: '
        + row['sizeSqFeetMax']
        + ' sqft'
        + ' [ATTR] Price: '
        + row['price']
    )


def encode_anchor_text(anchor: str) -> str:
    return '[TITLE] ' + anchor


def encode_description_text(description: str) -> str:
    return '[DESC] ' + description


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    prepared['clean_description'] = prepared['descriptionHtml'].apply(clean_html)

    for column in COLUMNS_TO_FILL:
        prepared[column] = prepared[column].fillna('N/A').astype(str)

    prepared['bedrooms'] = prepared['bedrooms'].apply(_format_decimal_string)
    prepared['bathrooms'] = prepared['bathrooms'].apply(_format_decimal_string)
    prepared['rich_anchor'] = prepared.apply(build_rich_anchor, axis=1)
    prepared['anchor_text'] = prepared['rich_anchor'].apply(encode_anchor_text)
    prepared['description_text'] = prepared['clean_description'].apply(encode_description_text)
    prepared['price_range'] = prepared['price'].apply(price_range_label)
    prepared = prepared.reset_index(drop=True)
    prepared['listing_id'] = prepared.index
    return prepared
