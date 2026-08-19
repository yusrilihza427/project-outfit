import os
import sys
import numpy as np
import pandas as pd

# ==================================================
# Agar Python dapat menemukan folder utils
# ==================================================
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from utils.embedding import FashionEmbedding

# ==================================================
# Load metadata
# ==================================================
metadata = pd.read_csv("metadata/dataset_outfit_final_revisi.csv")

# ==================================================
# MODE TEST
# Uncomment baris di bawah kalau hanya ingin test 10 gambar
# ==================================================
# metadata = metadata.head(10)

print(f"Total gambar yang akan diproses: {len(metadata)}")

# ==================================================
# Load FashionSigLIP
# ==================================================
embedder = FashionEmbedding()

embeddings = []
mapping = []
failed_images = []

# ==================================================
# Generate Embedding
# ==================================================
for idx, row in metadata.iterrows():

    image_path = os.path.join(
        "dataset",
        row["nama_gambar"]
    )

    try:

        embedding = embedder.encode_image(image_path)

        embeddings.append(embedding)

        mapping.append({
            "embedding_index": len(mapping),
            "id": row["id"],
            "nama_gambar": row["nama_gambar"]
        })

        print(
            f"[{idx+1}/{len(metadata)}] "
            f"{row['nama_gambar']} ✓"
        )

    except Exception as e:

        failed_images.append({
            "id": row["id"],
            "nama_gambar": row["nama_gambar"],
            "error": str(e)
        })

        print(
            f"[{idx+1}/{len(metadata)}] "
            f"{row['nama_gambar']} ❌"
        )

# ==================================================
# Simpan Embedding
# ==================================================
embeddings = np.array(embeddings)

os.makedirs("embeddings", exist_ok=True)

np.save(
    "embeddings/image_embeddings.npy",
    embeddings
)

pd.DataFrame(mapping).to_csv(
    "embeddings/image_mapping.csv",
    index=False
)

# ==================================================
# Simpan Log Error (jika ada)
# ==================================================
if failed_images:

    pd.DataFrame(failed_images).to_csv(
        "embeddings/failed_images.csv",
        index=False
    )

# ==================================================
# Ringkasan
# ==================================================
print("\n===================================")
print("Embedding selesai dibuat!")
print("===================================")
print(f"Total metadata      : {len(metadata)}")
print(f"Berhasil diproses   : {len(mapping)}")
print(f"Gagal diproses      : {len(failed_images)}")
print(f"Shape embedding     : {embeddings.shape}")

print("\nFile yang dihasilkan:")
print("- embeddings/image_embeddings.npy")
print("- embeddings/image_mapping.csv")

if failed_images:
    print("- embeddings/failed_images.csv")

