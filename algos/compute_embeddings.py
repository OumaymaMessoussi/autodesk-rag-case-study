import logging

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from global_variables import PAGES_EMBEDDINGS_PATH, PREPROCESSED_DATA_CSV_PATH

logging.basicConfig(level=logging.INFO)


def compute_embeddings(text: list, model_name: str, save_npy: bool = True) -> None:
    embeddings_path_with_model_name = PAGES_EMBEDDINGS_PATH + f"_{model_name}.npy"
    logging.info(f"Generating embeddings of {len(text)} text chunks using {model_name}")

    model = SentenceTransformer(
        model_name,
    )
    embeddings = model.encode(text).tolist()
    if save_npy:
        np.save(embeddings_path_with_model_name, embeddings)


if __name__ == "__main__":
    pages_data = pd.read_csv(PREPROCESSED_DATA_CSV_PATH)
    pages_data["content"].fillna("", inplace=True)
    # models tested: all-MiniLM-L6-v2, all-mpnet-base-v2
    model_name = "all-mpnet-base-v2"
    compute_embeddings(pages_data["content"].tolist(), model_name=model_name, save_npy=True)
