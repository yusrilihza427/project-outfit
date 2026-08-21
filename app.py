import streamlit as st
from pathlib import Path

from utils.recommendation import RecommendationEngine


# ============================================================
# KONFIGURASI HALAMAN
# ============================================================

st.set_page_config(
    page_title="Outfit Recommendation",
    page_icon="♡",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# PATH PROJECT
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / "dataset"
CSS_FILE = BASE_DIR / "style.css"


# ============================================================
# LOAD CSS
# ============================================================

def load_css():

    if CSS_FILE.exists():

        with open(
            CSS_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            css = f.read()

        st.markdown(
            f"<style>{css}</style>",
            unsafe_allow_html=True
        )

    else:

        st.warning(
            "File style.css belum ditemukan."
        )


load_css()


# ============================================================
# LOAD RECOMMENDATION ENGINE
# ============================================================

@st.cache_resource
def load_engine():

    return RecommendationEngine()


engine = load_engine()


# ============================================================
# HEADER
# ============================================================


 

# ============================================================
# USER PREFERENCES
# ============================================================

st.markdown(
    '<div class="section-title">Your Preferences</div>',
    unsafe_allow_html=True
)


# ------------------------------------------------------------
# INPUT
# ------------------------------------------------------------

col1, col2, col3 = st.columns(3)


with col1:

    aktivitas = st.selectbox(
        "Aktivitas",
        options=[
            "coffee_shop_hopping",
            "dinner",
            "gym_to_caffe",
            "hiking",
            "smart_casual"
        ]
    )


with col2:

    undertone = st.selectbox(
        "Undertone",
        options=[
            "warm",
            "cool",
            "neutral"
        ]
    )


with col3:

    hijab = st.selectbox(
        "Hijab",
        options=[
            "ya",
            "tidak"
        ]
    )


# ============================================================
# RECOMMENDATION BUTTON
# ============================================================

st.write("")


recommend_button = st.button(
    "♡  REKOMENDASIKAN OUTFIT",
    type="primary",
    use_container_width=False
)


# ============================================================
# RECOMMENDATION PROCESS
# ============================================================

if recommend_button:

    # --------------------------------------------------------
    # USER INPUT
    # --------------------------------------------------------

    user_input = {

        "aktivitas": aktivitas,

        "undertone": undertone,

        "hijab": hijab

    }


    # --------------------------------------------------------
    # PROSES REKOMENDASI
    # --------------------------------------------------------

    with st.spinner(
        "Sedang mencari outfit yang paling sesuai..."
    ):

        try:

            results = engine.recommend(
                user_input=user_input,
                top_k=3
            )


        except Exception as e:

            st.error(
                "Terjadi kesalahan saat menjalankan "
                "sistem rekomendasi."
            )

            st.exception(e)

            results = None


    # --------------------------------------------------------
    # HASIL
    # --------------------------------------------------------

    if results is not None:

        st.markdown(
            '<div class="section-title">Your Outfit Matches</div>',
            unsafe_allow_html=True
        )


        # ====================================================
        # HANDLE HASIL REKOMENDASI
        # ====================================================

        if isinstance(results, dict):

            # Kalau FAISS mengembalikan dictionary
            # yang memiliki key "results"
            if "results" in results:

                recommendation_results = results["results"]

            else:

                recommendation_results = results


        else:

            recommendation_results = results


        # Pastikan bentuknya list
        if not isinstance(
            recommendation_results,
            list
        ):

            recommendation_results = list(
                recommendation_results
            )


        # ====================================================
        # TOP 3 SAJA
        # ====================================================

        recommendation_results = recommendation_results[:3]


        # ====================================================
        # CEK HASIL
        # ====================================================

        if len(recommendation_results) == 0:

            st.warning(
                "Tidak ditemukan outfit yang sesuai "
                "dengan pilihan kamu."
            )


        else:

            # =================================================
            # TAMPILKAN 3 HASIL
            # =================================================

            result_columns = st.columns(3)


            for index, item in enumerate(
                recommendation_results
            ):

                with result_columns[index]:

                    # -----------------------------------------
                    # NAMA GAMBAR
                    # -----------------------------------------

                    image_name = item.get(
                        "nama_gambar"
                    )


                    # -----------------------------------------
                    # SCORE
                    # -----------------------------------------

                    score = item.get(
                        "score"
                    )


                    # -----------------------------------------
                    # IMAGE PATH
                    # -----------------------------------------

                    if image_name:

                        image_path = (
                            DATASET_DIR / image_name
                        )


                        if image_path.exists():

                            st.image(
                                str(image_path),
                                use_container_width=True
                            )

                        else:

                            st.warning(
                                f"Gambar tidak ditemukan: "
                                f"{image_name}"
                            )

                    else:

                        st.warning(
                            "Nama gambar tidak tersedia."
                        )


                    # -----------------------------------------
                    # RANKING
                    # -----------------------------------------

                    st.markdown(
                        f"""
                        <div class="recommendation-rank">
                            Recommendation #{index + 1}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


                    # -----------------------------------------
                    # SCORE
                    # -----------------------------------------

                    if score is not None:

                        try:

                            score_text = f"{float(score):.4f}"

                        except:

                            score_text = str(score)


                        st.markdown(
                            f"""
                            <div class="recommendation-score">
                                Similarity Score:
                                <b>{score_text}</b>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )


                    # -----------------------------------------
                    # NAMA FILE
                    # -----------------------------------------

                    if image_name:

                        st.caption(
                            image_name
                        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Outfit Recommendation System
        <br>
        AI-Based Outfit Recommendation
    </div>
    """,
    unsafe_allow_html=True
)