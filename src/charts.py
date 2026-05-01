"""
Visualizações: heatmap do triângulo, fatores de desenvolvimento, IBNR e Burning Cost.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def heatmap_triangulo(triangle: pd.DataFrame, lob: str = "") -> go.Figure:
    """
    Heatmap interativo do triângulo de run-off.
    Intensidade de cor reflete o valor pago; células futuras aparecem em cinza.
    """
    tri = triangle.copy().astype(float)
    anos  = [str(a) for a in tri.index]
    devs  = [str(d) for d in tri.columns]

    # Texto para hover: valor formatado ou "Futuro"
    text = []
    for i in tri.index:
        row_text = []
        for j in tri.columns:
            v = tri.loc[i, j]
            row_text.append(f"${v:,.0f}" if not np.isnan(v) else "Futuro")
        text.append(row_text)

    z = tri.values

    fig = go.Figure(go.Heatmap(
        z=z,
        x=devs,
        y=anos,
        text=text,
        texttemplate="%{text}",
        colorscale="Blues",
        showscale=True,
        colorbar=dict(title="Pago (USD)"),
        hoverongaps=False,
    ))

    fig.update_layout(
        title=f"Triângulo de Run-off — {lob.title()}",
        xaxis_title="Ano de Desenvolvimento",
        yaxis_title="Ano de Ocorrência",
        template="plotly_dark",
        height=500,
    )
    return fig


def grafico_fatores(link_ratios: pd.Series, lob: str = "") -> go.Figure:
    """Gráfico de barras dos fatores de desenvolvimento (link-ratios)."""
    fig = go.Figure(go.Bar(
        x=link_ratios.index.tolist(),
        y=link_ratios.values,
        marker_color="#2E86AB",
        text=[f"{v:.4f}" for v in link_ratios.values],
        textposition="outside",
    ))
    fig.add_hline(y=1.0, line_dash="dash", line_color="orange",
                  annotation_text="1.000 (sem desenvolvimento)")
    fig.update_layout(
        title=f"Fatores de Desenvolvimento (Link-Ratios) — {lob.title()}",
        xaxis_title="Período",
        yaxis_title="Fator",
        template="plotly_dark",
        height=400,
    )
    return fig


def grafico_ibnr(ibnr: pd.Series, ultimates: pd.Series, pagos: pd.Series,
                 lob: str = "") -> go.Figure:
    """Gráfico de barras empilhadas: Pago vs IBNR por ano de ocorrência."""
    anos = [str(a) for a in pagos.index]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Pago",
        x=anos,
        y=pagos.values,
        marker_color="#52B788",
    ))
    fig.add_trace(go.Bar(
        name="IBNR",
        x=anos,
        y=ibnr.values,
        marker_color="#E84855",
    ))
    fig.update_layout(
        barmode="stack",
        title=f"Pago vs IBNR por Ano de Ocorrência — {lob.title()}",
        xaxis_title="Ano de Ocorrência",
        yaxis_title="USD",
        template="plotly_dark",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def grafico_burning_cost(burning_cost: pd.Series, lob: str = "") -> go.Figure:
    """Linha temporal do Burning Cost histórico por ano."""
    anos = [str(a) for a in burning_cost.index]
    fig = go.Figure(go.Scatter(
        x=anos,
        y=burning_cost.values,
        mode="lines+markers",
        line=dict(color="#F4A261", width=2),
        marker=dict(size=8),
        text=[f"${v:,.0f}" for v in burning_cost.values],
        hovertemplate="%{x}<br>BC: %{text}<extra></extra>",
    ))
    fig.update_layout(
        title=f"Burning Cost Histórico (Indexado) — {lob.title()}",
        xaxis_title="Ano de Ocorrência",
        yaxis_title="Burning Cost (USD/contrato)",
        template="plotly_dark",
        height=400,
    )
    return fig


def salvar_todos(figs: dict[str, go.Figure], prefixo: str = "runoff") -> None:
    """Salva todos os gráficos como HTML interativo e PNG estático."""
    for nome, fig in figs.items():
        html_path = OUTPUT_DIR / f"{prefixo}_{nome}.html"
        png_path  = OUTPUT_DIR / f"{prefixo}_{nome}.png"
        fig.write_html(str(html_path))
        try:
            fig.write_image(str(png_path), width=1000, height=500)
        except Exception:
            pass  # kaleido opcional para PNG
