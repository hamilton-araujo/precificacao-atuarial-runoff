"""
Geração e carga do triângulo de run-off cumulativo.

Produz triângulos realistas para três Linhas de Negócio (LoB):
    - auto:      Seguro de automóvel (desenvolvimento rápido, ~5 anos)
    - liability: Responsabilidade civil (desenvolvimento lento, ~10 anos)
    - property:  Danos patrimoniais (desenvolvimento médio, ~7 anos)

Cada LoB tem seus próprios parâmetros de desenvolvimento e volatilidade,
calibrados com base em benchmarks atuariais de mercado.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# Parâmetros por LoB: (fatores_CDF_verdadeiros, volatilidade, ELR, trend_anual)
LOB_PARAMS = {
    "auto": {
        "n_dev":          6,
        "link_ratios":    [2.80, 1.45, 1.18, 1.07, 1.02, 1.005],
        "volatilidade":   0.08,
        "elr":            0.68,
        "trend":          0.03,
        "exposicao_base": 50_000,    # número de apólices
        "sinistro_medio": 1_200,     # USD por apólice-ano
    },
    "liability": {
        "n_dev":          10,
        "link_ratios":    [3.50, 1.90, 1.45, 1.25, 1.15, 1.09, 1.05, 1.03, 1.015, 1.005],
        "volatilidade":   0.15,
        "elr":            0.72,
        "trend":          0.05,
        "exposicao_base": 30_000,
        "sinistro_medio": 4_500,
    },
    "property": {
        "n_dev":          7,
        "link_ratios":    [2.20, 1.35, 1.12, 1.06, 1.03, 1.01, 1.003],
        "volatilidade":   0.10,
        "elr":            0.60,
        "trend":          0.025,
        "exposicao_base": 40_000,
        "sinistro_medio": 2_800,
    },
}

N_ANOS = 10  # anos de ocorrência
ANO_BASE = 2014


def carregar(lob: str = "auto", force_reload: bool = False) -> dict:
    """
    Carrega ou gera os dados atuariais para a LoB especificada.

    Returns:
        dict com chaves:
            triangle    — DataFrame cumulativo (linhas=ano_ocorrência, colunas=ano_dev)
            exposicao   — Series com exposição por ano de ocorrência
            premio      — Series com prêmio ganho por ano de ocorrência
            params      — dict com parâmetros da LoB
    """
    cache = DATA_DIR / f"{lob}_triangle.parquet"
    meta_cache = DATA_DIR / f"{lob}_meta.parquet"

    if not force_reload and cache.exists() and meta_cache.exists():
        logger.info("Carregando cache: %s", cache.name)
        triangle = pd.read_parquet(cache)
        meta = pd.read_parquet(meta_cache)
        return {
            "triangle":  triangle,
            "exposicao": meta["exposicao"],
            "premio":    meta["premio"],
            "params":    LOB_PARAMS[lob],
        }

    if lob not in LOB_PARAMS:
        raise ValueError(f"LoB '{lob}' inválida. Opções: {list(LOB_PARAMS)}")

    triangle, exposicao, premio = _gerar_triangulo(lob)
    triangle.to_parquet(cache)
    pd.DataFrame({"exposicao": exposicao, "premio": premio}).to_parquet(meta_cache)

    logger.info("Triângulo '%s' gerado: %s", lob, triangle.shape)
    return {
        "triangle":  triangle,
        "exposicao": exposicao,
        "premio":    premio,
        "params":    LOB_PARAMS[lob],
    }


def _gerar_triangulo(lob: str) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Gera triângulo cumulativo sintético com padrão realista de desenvolvimento."""
    rng = np.random.default_rng(hash(lob) % (2**32))
    p = LOB_PARAMS[lob]

    n_dev  = p["n_dev"]
    lrs    = np.array(p["link_ratios"])
    vol    = p["volatilidade"]
    elr    = p["elr"]
    trend  = p["trend"]
    exp_b  = p["exposicao_base"]

    anos = list(range(ANO_BASE, ANO_BASE + N_ANOS))

    # Exposição cresce ~3% ao ano com ruído
    exposicao = pd.Series(
        [exp_b * (1.03 ** i) * rng.uniform(0.92, 1.08) for i in range(N_ANOS)],
        index=anos, name="exposicao",
    )

    # Prêmio ganho: exposição × taxa × 12/24 (regra dos 24 avos simplificada)
    taxa = elr * 1.25  # loading de ~25% sobre ELR
    premio = (exposicao * taxa).rename("premio")

    # CDFs acumulados a partir dos link-ratios
    cdfs = np.cumprod(lrs[::-1])[::-1]  # CDF para cada coluna de desenvolvimento
    cdfs = np.append(cdfs, 1.0)          # último período = desenvolvido

    sinistro_medio = p["sinistro_medio"]

    # Ultimate sintético por ano: apólices × sinistro médio × ELR × tendência
    ultimates = [
        exposicao.iloc[i] * sinistro_medio * elr * (1 + trend) ** i * rng.uniform(0.85, 1.15)
        for i in range(N_ANOS)
    ]

    # Construir triângulo: célula (i, j) visível se i + j <= N_ANOS - 1
    data = {}
    for j in range(n_dev):
        col = ANO_BASE + j  # ano de desenvolvimento (proxy de coluna)
        col_data = []
        for i in range(N_ANOS):
            if i + j < N_ANOS:  # célula visível no triângulo
                fator = 1.0 / cdfs[j] if cdfs[j] > 0 else 1.0
                pago = ultimates[i] * fator * rng.lognormal(0, vol * 0.3)
                col_data.append(round(max(pago, 0), 0))
            else:
                col_data.append(np.nan)
        data[col] = col_data

    triangle = pd.DataFrame(data, index=anos)
    triangle.index.name = "ano_ocorrencia"
    triangle.columns.name = "ano_desenvolvimento"

    # Garantir cumulatividade (cada coluna >= anterior)
    for j in range(1, n_dev):
        col_prev = triangle.columns[j - 1]
        col_curr = triangle.columns[j]
        mask = triangle[col_curr].notna() & triangle[col_prev].notna()
        triangle.loc[mask, col_curr] = np.maximum(
            triangle.loc[mask, col_curr],
            triangle.loc[mask, col_prev],
        )

    return triangle, exposicao, premio


def resumo(dados: dict) -> None:
    """Imprime resumo do triângulo carregado."""
    tri = dados["triangle"]
    print(f"\n{'─'*55}")
    print(f"  Triângulo de Run-off  |  {tri.shape[0]} anos × {tri.shape[1]} períodos")
    print(f"  Período  : {tri.index[0]} → {tri.index[-1]}")
    print(f"  Células  : {tri.notna().sum().sum()} observadas / {tri.size} total")
    print(f"  Pagos    : ${tri.sum().sum()/1e6:.1f}M (diagonal atual)")
    print(f"  Exposição: ${dados['exposicao'].sum()/1e3:.0f}k contratos")
    print(f"{'─'*55}\n")
