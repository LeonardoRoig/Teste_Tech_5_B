
import pandas as pd
import streamlit as st
from train_and_rank import load_model, rank_candidates

st.set_page_config(page_title="Netflix das Vagas", layout="wide")
st.title("🎬 Netflix das Vagas — Top 10 Candidatos por Vaga")

pending_file = st.file_uploader("📂 CSV de pendentes (não classificados)", type=["csv"])

if pending_file:
    df_pending = pd.read_csv(pending_file)
    model = load_model("modelo.joblib")
    ranking = rank_candidates(model, df_pending, top_k=10)
    st.success("✅ Ranking gerado!")

    vagas = sorted(ranking["id_vaga"].unique().tolist())
    vaga_sel = st.sidebar.selectbox("Selecione a vaga", vagas)

    top = ranking[ranking["id_vaga"] == vaga_sel].sort_values("rank")
    st.subheader(f"Top {len(top)} candidatos para a vaga {vaga_sel}")

    cols = st.columns(5)
    for i, (_, row) in enumerate(top.iterrows()):
        col = cols[i % 5]
        with col:
            st.markdown(f"### 👤 Candidato {row['id_candidato']}")
            st.metric("Score", f"{row['score']:.3f}")
            st.caption(f"Rank: {int(row['rank'])}")

    with st.expander("📊 Tabela completa da vaga"):
        st.dataframe(top)
else:
    st.info("⏳ Faça upload do CSV de pendentes para rodar o ranking.")
