"""
Dashboard Streamlit — Precificação Atuarial e Triângulos de Run-off.

Execução:
    streamlit run src/app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
import pandas as pd

import ingest
import triangle as tri_mod
import pricing as prc_mod
import charts

st.set_page_config(
    page_title="Atuarial Run-off Dashboard",
    page_icon="📊",
    layout="wide",
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("Parâmetros")

lob = st.sidebar.selectbox(
    "Linha de Negócio",
    options=["auto", "liability", "property"],
    format_func=lambda x: {"auto": "Auto", "liability": "Responsabilidade Civil",
                            "property": "Patrimonial"}[x],
)

metodo = st.sidebar.radio(
    "Método Atuarial",
    options=["chain_ladder", "bf"],
    format_func=lambda x: "Chain Ladder" if x == "chain_ladder" else "Bornhuetter-Ferguson",
)

elr = st.sidebar.slider(
    "ELR — Taxa de Perda Esperada (apenas BF)",
    min_value=0.40, max_value=0.95, value=0.70, step=0.01,
    disabled=(metodo == "chain_ladder"),
)

trend = st.sidebar.slider(
    "Trend Inflacionário Anual (%)",
    min_value=0.0, max_value=10.0, value=3.0, step=0.5,
) / 100

loading = st.sidebar.slider(
    "Loading de Segurança (%)",
    min_value=5, max_value=50, value=25, step=5,
) / 100

anos_bc = st.sidebar.slider(
    "Anos para Burning Cost",
    min_value=3, max_value=8, value=5,
)

# ── Processamento com cache ───────────────────────────────────────────────────
@st.cache_data(show_spinner="Carregando triângulo...")
def _carregar(lob):
    return ingest.carregar(lob)

@st.cache_data(show_spinner="Calculando IBNR...")
def _calcular(lob, metodo, elr):
    dados = _carregar(lob)
    if metodo == "chain_ladder":
        res = tri_mod.chain_ladder(dados["triangle"])
    else:
        res = tri_mod.bornhuetter_ferguson(dados["triangle"], dados["premio"], elr=elr)
    res.lob = lob
    return res, dados

resultado, dados = _calcular(lob, metodo, elr)
preco = prc_mod.calcular_premio(
    dados["triangle"], dados["exposicao"], dados["premio"],
    resultado.ultimates, trend_anual=trend, anos_base=anos_bc,
    loading_factor=loading, lob=lob,
)

# ── Header ────────────────────────────────────────────────────────────────────
lob_nome = {"auto": "Auto", "liability": "Responsabilidade Civil", "property": "Patrimonial"}[lob]
st.title(f"Precificação Atuarial — {lob_nome}")
st.caption(f"Método: **{resultado.metodo}** | Trend: {trend:.1%} | Loading: {loading:.0%}")

# ── KPIs ──────────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("IBNR Total", f"${resultado.ibnr_total/1e6:.1f}M")
c2.metric("Burning Cost Médio", f"${preco.burning_cost.mean():,.0f}")
c3.metric("Prêmio Puro", f"${preco.premio_puro:,.0f}")
c4.metric("Prêmio Comercial", f"${preco.premio_comercial:,.0f}")

st.divider()

# ── Triângulo ─────────────────────────────────────────────────────────────────
st.subheader("Triângulo de Run-off")
fig_tri = charts.heatmap_triangulo(dados["triangle"], lob_nome)
st.plotly_chart(fig_tri, use_container_width=True)

# ── Fatores + IBNR ────────────────────────────────────────────────────────────
col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Fatores de Desenvolvimento")
    fig_lr = charts.grafico_fatores(resultado.link_ratios, lob_nome)
    st.plotly_chart(fig_lr, use_container_width=True)

with col_b:
    st.subheader("Pago vs IBNR por Ano")
    fig_ibnr = charts.grafico_ibnr(
        resultado.ibnr, resultado.ultimates, resultado.pagos, lob_nome
    )
    st.plotly_chart(fig_ibnr, use_container_width=True)

# ── Burning Cost ──────────────────────────────────────────────────────────────
st.subheader("Burning Cost Histórico (Indexado)")
fig_bc = charts.grafico_burning_cost(preco.burning_cost, lob_nome)
st.plotly_chart(fig_bc, use_container_width=True)

# ── Tabela de resultados ───────────────────────────────────────────────────────
st.subheader("Tabela de Projeções")
df_res = pd.DataFrame({
    "Pago ($)":        resultado.pagos.map("${:,.0f}".format),
    "Ultimate ($)":    resultado.ultimates.map("${:,.0f}".format),
    "IBNR ($)":        resultado.ibnr.map("${:,.0f}".format),
    "% Desenvolvido":  resultado.pct_desenvolvido.map("{:.1%}".format),
    "Burning Cost":    preco.burning_cost.map("${:,.0f}".format),
})
st.dataframe(df_res, use_container_width=True)

# ── Download ───────────────────────────────────────────────────────────────────
csv = pd.DataFrame({
    "pago":             resultado.pagos,
    "ultimate":         resultado.ultimates,
    "ibnr":             resultado.ibnr,
    "pct_desenvolvido": resultado.pct_desenvolvido,
    "burning_cost":     preco.burning_cost,
}).to_csv().encode()

st.download_button(
    label="Baixar resultados (CSV)",
    data=csv,
    file_name=f"runoff_{lob}_{metodo}.csv",
    mime="text/csv",
)
