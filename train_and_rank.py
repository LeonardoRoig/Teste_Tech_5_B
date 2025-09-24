
import pandas as pd
import joblib

ID_VAGA_COL, ID_CAND_COL = "id_vaga", "id_candidato"

def load_model(model_path="modelo.joblib"):
    return joblib.load(model_path)

def rank_candidates(model, df_pending, top_k=10):
    df_pred = df_pending.copy()
    df_pred["score"] = model.predict_proba(df_pred)[:, 1]
    df_pred["rank"] = df_pred.groupby(ID_VAGA_COL)["score"].rank(ascending=False, method="first")
    ranking = (
        df_pred[df_pred["rank"] <= top_k]
        .sort_values([ID_VAGA_COL, "rank"])
        .reset_index(drop=True)
    )
    return ranking
