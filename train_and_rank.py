
import re, json
import pandas as pd, numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Define standard columns
ID_VAGA_COL = "id_vaga"
ID_CAND_COL = "id_candidato"
TARGET_COL = "target"

# Define relevant columns for the model
relevant_cols = [
    'inf_titulo_vaga',
    'perfil_nivel_academico',
    'perfil_nivel profissional',
    'perfil_nivel_ingles',
    'perfil_nivel_espanhol',
    'perfil_competencia_tecnicas_e_comportamentais',
    'perfil_principais_atividades',
    'objetivo_profissional',
    'titulo_profissional',
    'area_atuacao',
    'conhecimentos_tecnicos',
    'qualificacoes',
    'certificacoes',
    'nivel_academico',
    'nivel_ingles',
    'nivel_espanhol',
    'cargo_atual',
    'nivel_profissional',
    'outro_idioma',
    'cursos',
    TARGET_COL # Include target for training
]

# Pre-compile the regex for tokenization
TOKEN_REGEX = re.compile(r"[\w\-\+]+")

def top_terms_overlap(job_text, cand_text, k=8):
    tok = lambda s: set(TOKEN_REGEX.findall(str(s).lower()))
    jset, cset = tok(job_text), tok(cand_text)
    return {"match_terms": list(jset & cset)[:k], "missing_terms": list(jset - cset)[:k]}

def local_similarity(job_text, cand_text):
    vec = TfidfVectorizer(max_features=2000)
    tfidf = vec.fit_transform([str(job_text), str(cand_text)])
    return float(cosine_similarity(tfidf[0], tfidf[1])[0][0])

def run_pipeline(train_path, pending_path, export_dir="/content/artifacts", top_k=10):
    df_train = pd.read_csv(train_path)
    df_pending = pd.read_csv(pending_path)

    # Select only relevant columns for training
    df_train_relevant = df_train[relevant_cols].copy()
    df_pending_relevant = df_pending[[col for col in relevant_cols if col != TARGET_COL] + [ID_VAGA_COL, ID_CAND_COL]].copy() # Keep IDs for merging/ranking

    X, y = df_train_relevant.drop(columns=[TARGET_COL]), df_train_relevant[TARGET_COL]

    # Identify categorical columns from the relevant columns
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    preprocessor = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)], remainder="passthrough")
    pipe = Pipeline([("preprocess", preprocessor), ("clf", RandomForestClassifier(n_estimators=300, random_state=42))])
    pipe.fit(X, y)

    # Prepare pending data for prediction using relevant columns
    df_pred = df_pending_relevant.copy()
    df_pred["score"] = pipe.predict_proba(df_pred.drop(columns=[ID_VAGA_COL, ID_CAND_COL]))[:,1] # Drop IDs for prediction

    df_pred["rank"] = df_pred.groupby(ID_VAGA_COL)["score"].rank(ascending=False, method="first")
    ranking = df_pred[df_pred["rank"] <= top_k].sort_values([ID_VAGA_COL,"rank"]).reset_index(drop=True)

    # Calculate recruiter details for the ranked candidates
    # Need to merge with original pending data to get the text columns for top_terms_overlap
    ranking = ranking.merge(df_pending[[ID_VAGA_COL, ID_CAND_COL, 'perfil_competencia_tecnicas_e_comportamentais', 'conhecimentos_tecnicos']],
                            on=[ID_VAGA_COL, ID_CAND_COL], how='left')

    ranking['recruiter_details'] = ranking.apply(
        lambda row: top_terms_overlap(
            row['perfil_competencia_tecnicas_e_comportamentais'],
            row['conhecimentos_tecnicos']
        ),
        axis=1
    )

    # Drop the merged text columns after calculating details
    ranking = ranking.drop(columns=['perfil_competencia_tecnicas_e_comportamentais', 'conhecimentos_tecnicos'])


    ranking.to_csv(f"{export_dir}/ranking_por_vaga.csv", index=False)

    # Include recruiter details in the cards
    cards = ranking[[ID_VAGA_COL, ID_CAND_COL, "score", "rank", "recruiter_details"]].to_dict(orient="records")
    with open(f"{export_dir}/cards_recrutador.json", "w", encoding="utf-8") as f:
        json.dump(cards, f, ensure_ascii=False, indent=2)

    return ranking
