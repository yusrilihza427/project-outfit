import pandas as pd


class RuleBasedFilter:

    # ==================================================
    # VALID INPUT
    # ==================================================
    VALID_ACTIVITY = {
        "coffee_shop_hopping",
        "dinner",
        "gym_to_caffe",
        "hiking",
        "smart_casual"
    }

    VALID_UNDERTONE = {
        "warm",
        "cool",
        "neutral"
    }

    VALID_HIJAB = {
        "ya",
        "tidak"
    }

    # ==================================================
    # RULE WARNA
    # Hasil majority voting / penyusunan metadata
    # ==================================================
    COLOR_RULES = {

        "warm": [
            "brown",
            "mustard_yellow",
            "terracota",
            "warm_red"
        ],

        "cool": [
            "cool_grey",
            "dusty_pink",
            "emerald",
            "lavender",
            "navy",
            "plum",
            "royal_blue",
            "silver"
        ],

        "neutral": [
            "beige",
            "black",
            "charcoal",
            "olive_green",
            "white"
        ]
    }

    # ==================================================
    # INITIALIZATION
    # ==================================================
    def __init__(self, metadata_path):

        self.metadata = pd.read_csv(metadata_path)

        # Normalisasi metadata agar konsisten
        columns_to_normalize = [
            "aktivitas",
            "undertone",
            "hijab",
            "warna_dominan"
        ]

        for column in columns_to_normalize:

            if column not in self.metadata.columns:
                raise ValueError(
                    f"Kolom '{column}' tidak ditemukan "
                    f"pada metadata."
                )

            self.metadata[column] = (
                self.metadata[column]
                .astype(str)
                .str.strip()
                .str.lower()
            )

        print("✅ Rule-Based metadata loaded")
        print(f"Total metadata : {len(self.metadata)}")

    # ==================================================
    # NORMALIZATION
    # ==================================================
    def _normalize(self, value):

        return str(value).strip().lower()

    # ==================================================
    # GET RECOMMENDED COLORS
    # ==================================================
    def _get_recommended_colors(self, undertone):
        """
        Menentukan daftar warna dominan berdasarkan
        kategori skin undertone.
        """

        if undertone not in self.COLOR_RULES:
            raise ValueError(
                f"Rule warna untuk undertone "
                f"'{undertone}' tidak tersedia."
            )

        return self.COLOR_RULES[undertone]

    # ==================================================
    # GENERATE TEXT QUERY
    # ==================================================
    def _generate_query(
        self,
        aktivitas,
        hijab,
        recommended_colors
    ):
        """
        Membentuk text query yang akan di-encode
        menggunakan Marqo-FashionSigLIP.
        """

        hijab_text = (
            "wearing a modest hijab"
            if hijab == "ya"
            else "without hijab"
        )

        activity_prompt = {

            "coffee_shop_hopping":
                "casual chic coffee shop outfit",

            "dinner":
                "elegant dinner outfit",

            "gym_to_caffe":
                "sporty athleisure outfit",

            "hiking":
                "comfortable outdoor hiking outfit",

            "smart_casual":
                "smart casual office inspired outfit"
        }

        # Untuk query FashionSigLIP:
        # mustard_yellow -> mustard yellow
        # dusty_pink -> dusty pink
        color_text = ", ".join(
            color.replace("_", " ")
            for color in recommended_colors
        )

        query = (
            f"A high quality fashion photo of a woman "
            f"wearing {activity_prompt[aktivitas]}, "
            f"{hijab_text}, "
            f"with dominant outfit colors such as "
            f"{color_text}, "
            f"full body outfit, fashionable clothing."
        )

        return query

    # ==================================================
    # MAIN RULE-BASED FILTER
    # ==================================================
    def filter(self, user_input):
        """
        Menjalankan Rule-Based System.

        Input:
        - aktivitas
        - undertone
        - hijab

        Output:
        - warna rekomendasi
        - kandidat metadata
        - candidate IDs
        - query FashionSigLIP
        """

        # ----------------------------------------------
        # Ambil dan normalisasi input
        # ----------------------------------------------
        aktivitas = self._normalize(
            user_input["aktivitas"]
        )

        undertone = self._normalize(
            user_input["undertone"]
        )

        hijab = self._normalize(
            user_input["hijab"]
        )

        # ----------------------------------------------
        # Validasi aktivitas
        # ----------------------------------------------
        if aktivitas not in self.VALID_ACTIVITY:

            raise ValueError(
                f"Aktivitas '{aktivitas}' tidak valid."
            )

        # ----------------------------------------------
        # Validasi undertone
        # ----------------------------------------------
        if undertone not in self.VALID_UNDERTONE:

            raise ValueError(
                f"Undertone '{undertone}' tidak valid."
            )

        # ----------------------------------------------
        # Validasi hijab
        # ----------------------------------------------
        if hijab not in self.VALID_HIJAB:

            raise ValueError(
                f"Hijab '{hijab}' tidak valid."
            )

        # ----------------------------------------------
        # RULE-BASED:
        # Undertone -> Recommended Colors
        # ----------------------------------------------
        recommended_colors = (
            self._get_recommended_colors(
                undertone
            )
        )

        # ----------------------------------------------
        # METADATA FILTERING
        #
        # Kandidat harus memenuhi:
        # 1. aktivitas
        # 2. undertone
        # 3. hijab
        # 4. warna hasil rule
        # ----------------------------------------------
        candidate = self.metadata[
            (
                self.metadata["aktivitas"]
                == aktivitas
            )
            &
            (
                self.metadata["undertone"]
                == undertone
            )
            &
            (
                self.metadata["hijab"]
                == hijab
            )
            &
            (
                self.metadata["warna_dominan"]
                .isin(recommended_colors)
            )
        ].reset_index(drop=True)

        # ----------------------------------------------
        # Generate FashionSigLIP query
        # ----------------------------------------------
        query = self._generate_query(
            aktivitas,
            hijab,
            recommended_colors
        )

        # ----------------------------------------------
        # Return hasil Rule-Based
        # ----------------------------------------------
        return {

            "candidate_df":
                candidate,

            "candidate_ids":
                candidate["id"]
                .astype(int)
                .tolist(),

            "recommended_colors":
                recommended_colors,

            "query":
                query,

            "total_candidate":
                len(candidate)
        }