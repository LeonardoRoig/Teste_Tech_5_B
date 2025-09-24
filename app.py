
import os
import json
import pandas as pd
import streamlit as st
import importlib.util

# Define the export directory
EXPORT_DIR = "/content/artifacts"

# Ensure the directory exists
os.makedirs(EXPORT_DIR, exist_ok=True)

# Define the path to train_and_rank.py
train_and_rank_path = os.path.join(EXPORT_DIR, 'train_and_rank.py')

# Dynamically import the function from the saved file
spec = importlib.util.spec_from_file_location("train_and_rank", train_and_rank_path)
train_and_rank = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_and_rank)
run_pipeline = train_and_rank.run_pipeline


st.set_page_config(page_title="Netflix das Vagas", layout="wide")
st.title("🎬 Netflix das Vagas — Top 10 Candidatos por Vaga")

st.markdown("Faça upload dos arquivos para rodar o pipeline:")

train_file = st.file_uploader("📂 CSV de treino (aprovados/reprovados)", type=["csv"])
pending_file = st.file_uploader("📂 CSV de pendentes (não classificados)", type=["csv"])

if train_file and pending_file:
    # Save uploaded files to the artifacts directory for run_pipeline to access
    train_path = os.path.join(EXPORT_DIR, 'train_uploaded.csv')
    pending_path = os.path.join(EXPORT_DIR, 'pending_uploaded.csv')

    with open(train_path, "wb") as f:
        f.write(train_file.getbuffer())
    with open(pending_path, "wb") as f:
        f.write(pending_file.getbuffer())

    # Run the pipeline
    ranking = run_pipeline(train_path, pending_path, export_dir=EXPORT_DIR)
    st.success("✅ Pipeline executado!")

    # Filter for top 10 vacancies with most candidates ranked 1
    top_10_vagas = ranking[ranking['rank'] == 1.0]['id_vaga'].value_counts().nlargest(10).index.tolist()
    ranking_filtered = ranking[ranking['id_vaga'].isin(top_10_vagas)]

    # Sidebar para escolher vaga (filtered)
    vagas = sorted(ranking_filtered["id_vaga"].unique().tolist())
    vaga_sel = st.sidebar.selectbox("Selecione a vaga", vagas)

    top = ranking_filtered[ranking_filtered["id_vaga"] == vaga_sel].sort_values("rank")
    st.subheader(f"Top {len(top)} candidatos para a vaga {vaga_sel}")

    # Define important columns to display in the table, excluding recruiter_details
    important_cols_table = [
        'id_candidato', 'score', 'rank', 'perfil_nivel_academico',
        'perfil_nivel profissional', 'perfil_nivel_ingles',
        'perfil_nivel_espanhol', 'objetivo_profissional', 'qualificacoes',
        'certificacoes', 'experiencias', 'nivel_academico', 'nivel_ingles',
        'nivel_espanhol', 'cargo_atual', 'nivel_profissional', 'outro_idioma',
        'cursos'
    ]


    # Cards estilo Netflix (10 candidatos = 2 linhas de 5)
    cols = st.columns(5)
    for i, (_, row) in enumerate(top.iterrows()):
        col = cols[i % 5]
        with col:
            st.markdown(f"### 👤 Candidato {row['id_candidato']}")
            st.metric("Score (modelo)", f"{row['score']:.3f}")
            st.caption(f"Rank: {row['rank']}")

            # Expander with recruiter details only
            with st.expander("Detalhes do Recrutador"):
                details = row.get('recruiter_details', {}) # Use .get for safety
                st.write("**Termos que deram match:**", details.get("match_terms", []))
                st.write("**Lacunas (temos na vaga não encontrados no candidato):**", details.get("missing_terms", []))

                st.markdown("---") # Separator


    # Tabela completa com colunas importantes and without "Não informado"
    # Select only the columns for the table
    top_display = top[[col for col in important_cols_table if col in top.columns]].copy()
    # Replace "Não informado" and None with empty string for better display in table
    top_display.replace(['Não informado', None], '', inplace=True)


    st.markdown("---") # Separator
    with st.expander("📊 Tabela completa da vaga (informações importantes)"):
        st.dataframe(top_display)

else:
    st.info("⏳ Aguarde: faça upload dos dois arquivos CSV para rodar o pipeline.")

