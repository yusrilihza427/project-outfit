import faiss
import pandas as pd
import numpy as np
import hashlib
import os


class FaissSearch:

    def __init__(
        self,
        index_path="faiss_index/fashion.index",
        mapping_path="embeddings/image_mapping.csv",
        dataset_path="dataset"
    ):

        self.index = faiss.read_index(index_path)
        self.mapping = pd.read_csv(mapping_path)
        self.dataset_path = dataset_path

        print("✅ FAISS Index Loaded")
        print(f"Jumlah vector : {self.index.ntotal}")

    def _get_image_hash(self, nama_gambar):
        """
        Membuat hash berdasarkan isi file gambar.
        Digunakan untuk mendeteksi exact duplicate.
        """

        image_path = os.path.join(
            self.dataset_path,
            nama_gambar
        )

        try:
            with open(image_path, "rb") as f:
                return hashlib.md5(f.read()).hexdigest()

        except Exception:
            return nama_gambar

    def search(
        self,
        query_embedding,
        top_k=5,
        candidate_ids=None
    ):
        """
        Melakukan pencarian menggunakan FAISS
        dan menghapus gambar duplikat dari hasil retrieval.
        """

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32
        ).reshape(1, -1)

        # Ambil kandidat lebih banyak agar setelah
        # deduplikasi tetap tersedia cukup hasil.
        if candidate_ids is not None:
            search_k = self.index.ntotal
        else:
            search_k = min(
                max(top_k * 10, 50),
                self.index.ntotal
            )

        scores, indices = self.index.search(
            query_embedding,
            search_k
        )

        results = []

        seen_ids = set()
        seen_hashes = set()

        candidate_set = (
            set(candidate_ids)
            if candidate_ids is not None
            else None
        )

        for score, idx in zip(scores[0], indices[0]):

            if idx == -1:
                continue

            row = self.mapping.iloc[idx]

            item = {
                "embedding_index": int(idx),
                "id": int(row["id"]),
                "nama_gambar": str(row["nama_gambar"]),
                "score": float(score)
            }

            # Filter Rule-Based terlebih dahulu
            if (
                candidate_set is not None
                and item["id"] not in candidate_set
            ):
                continue

            # Hindari ID yang sama
            if item["id"] in seen_ids:
                continue

            # Cek exact duplicate berdasarkan isi gambar
            image_hash = self._get_image_hash(
                item["nama_gambar"]
            )

            if image_hash in seen_hashes:
                continue

            seen_ids.add(item["id"])
            seen_hashes.add(image_hash)

            results.append(item)

            if len(results) >= top_k:
                break

        return results

