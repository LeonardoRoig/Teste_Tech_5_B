
import os
import json
import pandas as pd
import streamlit as st

# Change to the directory where train_and_rank.py is located
original_dir = os.getcwd()
os.chdir('/content/artifacts')

# Import the function
from train_and_rank import run_pipeline

# Change back to the original directory
os.chdir(original_dir)

st.set_page_config(page_title="Netflix das Vagas", layout="wide")
st.title("🎬 Netflix das Vagas — Top 10 Candidatos por Vaga")

st.markdown("Faça upload dos arquivos para rodar o pipeline:")

train_file = st.file_uploader("📂 CSV de treino (aprovados/reprovados)", type=["csv"])
pending_file = st.file_uploader("📂 CSV de pendentes (não classificados)", type=["csv"])

if train_file and pending_file:
    # Save uploaded files to the artifacts directory for run_pipeline to access
    os.makedirs(os.path.join(original_dir, 'artifacts'), exist_ok=True)
    train_path = os.path.join(original_dir, 'artifacts', 'train.csv')
    pending_path = os.path.join(original_dir, 'artifacts', 'pending.csv')

    with open(train_path, "wb") as f:
        f.write(train_file.getbuffer())
    with open(pending_path, "wb") as f:
        f.write(pending_file.getbuffer())

    # Run the pipeline
    # Specify the export_dir as the artifacts directory
    ranking = run_pipeline(train_path, pending_path, export_dir=os.path.join(original_dir, 'artifacts'))
    st.success("✅ Pipeline executado!")

    # Sidebar para escolher vaga
    vagas = sorted(ranking["id_vaga"].unique().tolist())
    vaga_sel = st.sidebar.selectbox("Selecione a vaga", vagas)

    top = ranking[ranking["id_vaga"] == vaga_sel].sort_values("rank")
    st.subheader(f"Top {len(top)} candidatos para a vaga {vaga_sel}")

    # Cards estilo Netflix (10 candidatos = 2 linhas de 5)
    cols = st.columns(5)
    for i, (_, row) in enumerate(top.iterrows()):
        col = cols[i % 5]
        with col:
            st.markdown(f"### 👤 Candidato {row['id_candidato']}")
            st.metric("Score (modelo)", f"{row['score']:.3f}")
            st.caption(f"Rank: {row['rank']}")

            # Expander com detalhes do recrutador
            with st.expander("Detalhes do Recrutador"):
                details = row['recruiter_details']
                st.write("**Termos que deram match:**", details.get("match_terms", []))
                st.write("**Lacunas (termos na vaga não encontrados no candidato):**", details.get("missing_terms", []))


    # Tabela completa
    with st.expander("📊 Tabela completa da vaga"):
        st.dataframe(top)
else:
    st.info("⏳ Aguarde: faça upload dos dois arquivos CSV para rodar o pipeline.")

