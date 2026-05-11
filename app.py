"""
Steam Game Recommender — Streamlit demo UI.

Exposes all 7 recommendation algorithms behind a unified interface:
- Non-personalized popularity ranking
- User-based collaborative filtering
- Item-based collaborative filtering
- Content-based filtering (TF-IDF + multi-hot)
- ALS matrix factorization (in-house NumPy/CuPy)
- ALS matrix factorization (PySpark, distributed)
- Autoencoder content-based filtering (TensorFlow)
- Two-tower neural retrieval (TensorFlow)

Run with: streamlit run app.py
"""

import os
import sys
import pickle
from pathlib import Path

# Make the processing/ package importable both as `processing.xxx` and bare `xxx`
# (the original code uses `from config import config` etc. which depends on cwd)
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "processing"))

import pandas as pd
import streamlit as st


PROCESSED_DIR = Path("processing/processedData")


# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Steam Game Recommender",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ----------------------------------------------------------------------
# Data loading (cached)
# ----------------------------------------------------------------------
@st.cache_data
def load_user2game():
    p = PROCESSED_DIR / "user2game_dict.pkl"
    if not p.exists():
        return {}
    with open(p, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_game_metadata():
    p = Path("data/steam_games.json")
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_json(p, lines=True)
    return df


def has_artifact(name: str) -> bool:
    return (PROCESSED_DIR / name).exists()


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
with st.sidebar:
    st.title("🎮 Steam Recommender")
    st.caption("CMP 49412 — Intelligent Recommendation Systems")

    algorithm = st.selectbox(
        "Algorithm",
        [
            "Non-personalized (Popularity)",
            "User-based Collaborative Filtering",
            "Item-based Collaborative Filtering",
            "Content-based Filtering (TF-IDF)",
            "ALS Matrix Factorization (NumPy/CuPy)",
            "ALS Matrix Factorization (PySpark)",
            "Autoencoder Content-Based (TensorFlow)",
            "Two-Tower Neural Retrieval (TensorFlow)",
        ],
        help="Each algorithm models user-game preference differently. See the About tab.",
    )

    st.divider()

    user2game = load_user2game()
    available_users = sorted([u for u, games in user2game.items() if len(games) >= 5])

    if not available_users:
        st.warning("No processed user data yet. Run `python main.py --evaluate` first.")
        user_id = None
    else:
        user_id = st.selectbox(
            "User",
            available_users[:200],  # cap for UI perf
            help="Pick a user to recommend games for. Limited to top 200 active users in the dropdown.",
        )

    top_n = st.slider("Top N recommendations", 5, 30, 10)
    st.divider()
    st.caption("Built by Omar Elshall  ·  [github.com/Omar-Elshall](https://github.com/Omar-Elshall)")


# ----------------------------------------------------------------------
# Main tabs
# ----------------------------------------------------------------------
tab_recs, tab_eval, tab_about = st.tabs(["Recommendations", "Evaluation", "About"])


with tab_recs:
    st.header(f"Top {top_n} for {algorithm}")
    if user_id is None:
        st.info("Select a user from the sidebar.")
    else:
        st.caption(f"User: `{user_id}`  ·  Currently owns {len(user2game.get(user_id, []))} games")

        with st.spinner("Computing recommendations…"):
            recs = []
            try:
                if algorithm == "Non-personalized (Popularity)":
                    from processing.utils import get_trending_recommendations  # type: ignore
                    recs = get_trending_recommendations(top_n)
                elif algorithm == "User-based Collaborative Filtering":
                    from processing.utils import get_user_based_cf_recommendations  # type: ignore
                    recs = get_user_based_cf_recommendations(user_id, top_n)
                elif algorithm == "Item-based Collaborative Filtering":
                    from processing.utils import get_item_based_cf_recommendations  # type: ignore
                    recs = get_item_based_cf_recommendations(user_id, top_n)
                elif algorithm == "Content-based Filtering (TF-IDF)":
                    # Content-based needs a SEED game (not a user); pick the first owned game
                    from processing.utils import get_content_based_recommendations  # type: ignore
                    seed = (user2game.get(user_id) or [None])[0]
                    if seed is None:
                        recs = []
                    else:
                        st.caption(f"Seed game (first owned): `{seed}`")
                        recs = get_content_based_recommendations(seed, top_n)
                elif algorithm == "ALS Matrix Factorization (NumPy/CuPy)":
                    if not has_artifact("U_matrix.pkl"):
                        st.warning("ALS model not trained yet. Run `python main.py` and choose ALS training.")
                    else:
                        from processing.utils import get_als_recommendations  # type: ignore
                        recs = get_als_recommendations(user_id, top_n)
                elif algorithm == "ALS Matrix Factorization (PySpark)":
                    if not has_artifact("pyspark_als_model"):
                        st.warning("PySpark ALS model not trained yet. Run `python -m processing.pyspark_als`.")
                    else:
                        from processing.pyspark_als import (  # type: ignore
                            get_spark,
                            recommend_for_user,
                        )
                        from pyspark.ml.recommendation import ALSModel
                        from pyspark.ml import PipelineModel

                        spark = get_spark()
                        model = ALSModel.load(str(PROCESSED_DIR / "pyspark_als_model"))
                        indexer = PipelineModel.load(str(PROCESSED_DIR / "pyspark_indexer"))
                        recs = recommend_for_user(model, indexer, user_id, top_n)
                elif algorithm == "Autoencoder Content-Based (TensorFlow)":
                    if not has_artifact("tf_autoencoder_encoder.keras"):
                        st.warning("Autoencoder not trained yet. Run `python -m processing.tf_autoencoder`.")
                    else:
                        st.info("Autoencoder recommendations require a content feature matrix. "
                                "See `processing/tf_autoencoder.recommend_for_user`.")
                elif algorithm == "Two-Tower Neural Retrieval (TensorFlow)":
                    if not has_artifact("tf_two_tower_user.keras"):
                        st.warning("Two-tower model not trained yet. Run `python -m processing.tf_two_tower`.")
                    else:
                        st.info("Two-tower recommendations require user + item feature matrices. "
                                "See `processing/tf_two_tower.recommend_for_user`.")
            except ModuleNotFoundError as e:
                st.error(f"Missing dependency: `{e.name}`. Install with `pip install -r requirements.txt`.")
            except Exception as e:
                st.error(f"Error generating recommendations: {e}")

        # Normalize: the underlying functions return mixed shapes — DataFrame, list[tuple], list[str], or None.
        # Convert to a uniform list[(game_id, score|None)] for downstream display.
        def _normalize(r):
            if r is None:
                return []
            if isinstance(r, pd.DataFrame):
                if r.empty:
                    return []
                # Try common column patterns
                id_col = next((c for c in ("game_id", "id", "item", "item_id") if c in r.columns), r.columns[0])
                score_col = next((c for c in ("score", "rating", "prediction", "similarity") if c in r.columns), None)
                if score_col:
                    return list(zip(r[id_col].astype(str).tolist(), r[score_col].astype(float).tolist()))
                return [(str(v), None) for v in r[id_col].tolist()]
            if isinstance(r, dict):
                return [(str(k), float(v)) for k, v in r.items()]
            # Assume list-like
            out = []
            for item in r:
                if isinstance(item, tuple) and len(item) >= 2:
                    out.append((str(item[0]), float(item[1]) if item[1] is not None else None))
                elif isinstance(item, (list, tuple)) and len(item) == 1:
                    out.append((str(item[0]), None))
                else:
                    out.append((str(item), None))
            return out

        recs_norm = _normalize(recs)
        if recs_norm:
            games_df = load_game_metadata()
            rows = []
            for rank, (game_id, score) in enumerate(recs_norm, 1):
                title = ""
                if not games_df.empty and "id" in games_df.columns:
                    try:
                        match = games_df[games_df["id"].astype(str) == game_id]
                        if not match.empty:
                            title = match.iloc[0].get("app_name", "") or ""
                    except Exception:
                        pass
                rows.append({
                    "Rank": rank,
                    "Game ID": game_id,
                    "Title": title,
                    "Score": f"{score:.4f}" if score is not None else "",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        elif recs is not None:
            st.info("Recommendation function returned an empty result.")


with tab_eval:
    st.header("Algorithm Comparison")
    st.markdown(
        "Run `python main.py --evaluate` to compute Precision@k, Recall@k, F1, and RMSE for all "
        "algorithms. Results are written to `processing/processedData/evaluation_results.json`."
    )
    eval_path = PROCESSED_DIR / "evaluation_results.json"
    if eval_path.exists():
        results = pd.read_json(eval_path)
        st.dataframe(results, use_container_width=True)
    else:
        st.info("No evaluation results found yet.")


with tab_about:
    st.header("About this project")
    st.markdown(
        """
This project benchmarks 8 recommendation algorithms on the Steam ownership dataset
(Australian Steam users; user-game ownership, reviews, and metadata from Kaggle).

### Algorithms

| # | Family | Algorithm | Module |
|---|---|---|---|
| 1 | Non-personalized | Popularity ranking with Bayesian scoring | `processing/non_personalized_trending.py` |
| 2 | Memory-based CF | User-based (cosine similarity) | `processing/user_item_based_cf.py` |
| 3 | Memory-based CF | Item-based (cosine similarity) | `processing/user_item_based_cf.py` |
| 4 | Content-based | TF-IDF + genre/tag multi-hot | `processing/content_based_filtering.py` |
| 5 | Model-based CF | ALS matrix factorization (in-house NumPy/CuPy) | `processing/ALS_matrix_factorization.py` |
| 6 | Model-based CF | ALS matrix factorization (PySpark, distributed) | `processing/pyspark_als.py` |
| 7 | Deep learning | Autoencoder content-based filtering | `processing/tf_autoencoder.py` |
| 8 | Deep learning | Two-tower neural retrieval | `processing/tf_two_tower.py` |

### Evaluation

All algorithms are evaluated on a 90/10 train/test split with these metrics:

- **Precision@k, Recall@k, F1** — top-N ranking quality
- **RMSE** — score calibration vs. ownership labels (for regression-style models)

Per the course (CMP 49412 Lesson 9), Precision@k and Recall@k are the
primary ranking metrics; RMSE is added for the matrix-factorization and
two-tower outputs to characterize score calibration.

### Stack

Python · NumPy · Pandas · scikit-learn · **CuPy** (GPU acceleration) ·
**PySpark** (distributed ALS) · **TensorFlow** (autoencoder + two-tower)
· Streamlit (this UI) · Jupyter (research notebooks)
        """
    )
