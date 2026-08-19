import os
import sys
import numpy as np
import faiss

# ==================================================
# Agar Python menemukan project root
# ==================================================
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

# ==================================================
# Load embedding
# ==================================================
embeddings = np.load("embeddings/image_embeddings.npy")

print("Embedding berhasil dimuat")
print("Shape :", embeddings.shape)

# ==================================================
# Pastikan float32
# ==================================================
embeddings = embeddings.astype("float32")

# ==================================================
# Buat FAISS Index
# ==================================================
dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)

print("\nFAISS Index berhasil dibuat")
print("Jumlah vector :", index.ntotal)

# ==================================================
# Simpan Index
# ==================================================
os.makedirs("faiss_index", exist_ok=True)

faiss.write_index(
    index,
    "faiss_index/fashion.index"
)

print("\nIndex berhasil disimpan")
print("Lokasi : faiss_index/fashion.index")