from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / "dataset"
METADATA_PATH = BASE_DIR / "metadata" / "dataset_outfit_final_revisi.csv"

EMBEDDING_DIR = BASE_DIR / "embeddings"
FAISS_DIR = BASE_DIR / "faiss_index"

OUTPUT_DIR = BASE_DIR / "outputs"