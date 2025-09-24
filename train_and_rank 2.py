
import re, json
import pandas as pd, numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ID_VAGA_COL, ID_CAND_COL, TARGET_COL = "id_vaga", "id_candidato", "target"

def top_terms_overlap(job_text, cand_text, k=8):
    tok = lambda s: set(re.findall(r"[\w\-\+]+", str(s).lower()))
    jset, cset = tok(job_text), tok(cand_text)
    return {"match_terms": list(jset & cset)[:k], "missing_terms": list(jset - cset)[:k]}

def local_similarity(job_text, cand_text):
    vec = TfidfVectorizer(max_features=2000)
    tfidf = vec.fit_transform([str(job_text), str(cand_text)])
    return float(cosine_similarity(tfidf[0], tfidf[1])[0][0])

def run_pipeline(train_path, pending_path, export_dir="/content/artifacts", top_k=10):
    df_train, df_pending = pd.read_csv(train_path), pd.read_csv(pending_path)
    X, y = df_train.drop(columns=[TARGET_COL]), df_train[TARGET_COL]
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    for c in [ID_VAGA_COL, ID_CAND_COL]:
        if c in cat_cols: cat_cols.remove(c)
    preprocessor = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)], remainder="passthrough")
    pipe = Pipeline([("preprocess", preprocessor), ("clf", RandomForestClassifier(n_estimators=300, random_state=42))])
    pipe.fit(X, y)
    df_pred = df_pending.copy()
    df_pred["score"] = pipe.predict_proba(df_pred)[:,1]
    df_pred["rank"] = df_pred.groupby(ID_VAGA_COL)["score"].rank(ascending=False, method="first")
    ranking = df_pred[df_pred["rank"] <= top_k].sort_values([ID_VAGA_COL,"rank"]).reset_index(drop=True)

    # Calculate recruiter details
    ranking['recruiter_details'] = ranking.apply(
        lambda row: top_terms_overlap(
            row['perfil_competencia_tecnicas_e_comportamentais'],
            row['conhecimentos_tecnicos']
        ),
        axis=1
    )

    ranking.to_csv(f"{export_dir}/ranking_por_vaga.csv", index=False)

    # Include recruiter details in the cards
    cards = ranking[[ID_VAGA_COL, ID_CAND_COL, "score", "rank", "recruiter_details"]].to_dict(orient="records")
    with open(f"{export_dir}/cards_recrutador.json", "w", encoding="utf-8") as f:
        json.dump(cards, f, ensure_ascii=False, indent=2)
    return ranking
