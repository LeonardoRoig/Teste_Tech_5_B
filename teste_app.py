
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

    # Filter for top 10 vacancies with most candidates ranked 1
    top_10_vagas = ranking[ranking['rank'] == 1.0]['id_vaga'].value_counts().nlargest(10).index.tolist()
    ranking_filtered = ranking[ranking['id_vaga'].isin(top_10_vagas)]

    # Sidebar para escolher vaga (filtered)
    vagas = sorted(ranking_filtered["id_vaga"].unique().tolist())
    vaga_sel = st.sidebar.selectbox("Selecione a vaga", vagas)

    top = ranking_filtered[ranking_filtered["id_vaga"] == vaga_sel].sort_values("rank")
    st.subheader(f"Top {len(top)} candidatos para a vaga {vaga_sel}")

    # Define important columns to display
    important_cols = [
        'id_candidato', 'score', 'rank', 'perfil_nivel_academico',
        'perfil_nivel profissional', 'perfil_nivel_ingles',
        'perfil_nivel_espanhol', 'objetivo_profissional', 'qualificacoes',
        'certificacoes', 'experiencias', 'nivel_academico', 'nivel_ingles',
        'nivel_espanhol', 'cargo_atual', 'nivel_profissional', 'outro_idioma',
        'cursos', 'recruiter_details' # Include recruiter_details for the expander
    ]

    # Cards estilo Netflix (10 candidatos = 2 linhas de 5)
    cols = st.columns(5)
    for i, (_, row) in enumerate(top.iterrows()):
        col = cols[i % 5]
        with col:
            st.markdown(f"### 👤 Candidato {row['id_candidato']}")
            st.metric("Score (modelo)", f"{row['score']:.3f}")
            st.caption(f"Rank: {row['rank']}")

            # Expander with recruiter details and other important info
            with st.expander("Detalhes do Recrutador e Candidato"):
                details = row['recruiter_details']
                st.write("**Termos que deram match:**", details.get("match_terms", []))
                st.write("**Lacunas (termos na vaga não encontrados no candidato):**", details.get("missing_terms", []))

                st.markdown("---") # Separator

                # Display other important details, excluding "Não informado"
                for col_name in important_cols:
                    if col_name not in ['id_candidato', 'score', 'rank', 'recruiter_details']:
                        value = row[col_name]
                        if pd.notna(value) and str(value).lower() != 'não informado':
                            st.write(f"**{col_name.replace('_', ' ').title()}:** {value}")


    # Tabela completa com colunas importantes e sem "Não informado"
    top_display = top[important_cols].copy()
    # Replace "Não informado" and None with empty string for better display in table
    top_display.replace(['Não informado', None], '', inplace=True)
    st.markdown("---") # Separator
    with st.expander("📊 Tabela completa da vaga (informações importantes)"):
        st.dataframe(top_display)
else:
    st.info("⏳ Aguarde: faça upload dos dois arquivos CSV para rodar o pipeline.")

