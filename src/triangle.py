"""
Motor atuarial: Chain Ladder, Bornhuetter-Ferguson e Tail Factor.

Referências:
    - Mack (1993): Distribution-Free Calculation of the Standard Error of CL
    - Bornhuetter & Ferguson (1972): The Actuary and IBNR
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ResultadoIBNR:
    """Resultado completo da projeção atuarial."""
    metodo:         str
    lob:            str
    link_ratios:    pd.Series         # fatores age-to-age ponderados
    tail_factor:    float             # fator de cauda
    cdfs:           pd.Series         # CDFs acumulados por coluna
    ultimates:      pd.Series         # projeção do custo final por ano
    pagos:          pd.Series         # total pago por ano (diagonal atual)
    ibnr:           pd.Series         # IBNR = Ultimate - Pago
    ibnr_total:     float
    pct_desenvolvido: pd.Series       # % desenvolvido por ano


def chain_ladder(
    triangle: pd.DataFrame,
    tail_factor: float | None = None,
) -> ResultadoIBNR:
    """
    Método Chain Ladder determinístico com fatores ponderados por volume.

    Args:
        triangle:    Triângulo cumulativo (linhas=ano_ocorrência, colunas=dev).
        tail_factor: Fator de cauda manual. Se None, estimado via curva exponencial.

    Returns:
        ResultadoIBNR com ultimates e IBNR por ano de ocorrência.
    """
    tri = triangle.copy()
    cols = tri.columns.tolist()
    n_cols = len(cols)

    # ── Calcular link-ratios ponderados por volume ────────────────────────
    lrs = {}
    for k in range(n_cols - 1):
        c_from = cols[k]
        c_to   = cols[k + 1]
        mask   = tri[c_from].notna() & tri[c_to].notna()
        if mask.sum() < 2:
            lrs[k] = 1.0
        else:
            lrs[k] = tri.loc[mask, c_to].sum() / tri.loc[mask, c_from].sum()

    link_ratios = pd.Series(lrs, name="link_ratio")
    link_ratios.index = [f"{cols[k]}-{cols[k+1]}" for k in range(n_cols - 1)]

    # ── Tail Factor ───────────────────────────────────────────────────────
    if tail_factor is None:
        tail_factor = _estimar_tail(link_ratios.values)
    logger.info("Tail factor: %.4f", tail_factor)

    # ── CDFs acumulados (direita para esquerda) ───────────────────────────
    all_factors = np.append(link_ratios.values, tail_factor)
    cdfs_arr = np.cumprod(all_factors[::-1])[::-1]
    cdfs = pd.Series(cdfs_arr, index=cols, name="CDF")

    # ── Diagonal atual (valores mais recentes por ano) ────────────────────
    pagos = _diagonal_atual(tri)

    # ── Projetar Ultimates ────────────────────────────────────────────────
    col_por_ano = _coluna_atual(tri)          # Series: ano -> coluna
    cdf_por_ano = col_por_ano.map(cdfs)       # Series: ano -> CDF escalar
    ultimates   = pagos * cdf_por_ano
    ultimates.name = "ultimate_cl"

    ibnr = ultimates - pagos
    ibnr.name = "ibnr_cl"

    pct_dev = (pagos / ultimates).clip(0, 1)
    pct_dev.name = "pct_desenvolvido"

    logger.info("Chain Ladder — IBNR total: $%.0f", ibnr.sum())

    return ResultadoIBNR(
        metodo="Chain Ladder",
        lob="",
        link_ratios=link_ratios,
        tail_factor=tail_factor,
        cdfs=cdfs,
        ultimates=ultimates,
        pagos=pagos,
        ibnr=ibnr,
        ibnr_total=float(ibnr.sum()),
        pct_desenvolvido=pct_dev,
    )


def bornhuetter_ferguson(
    triangle: pd.DataFrame,
    premio: pd.Series,
    elr: float = 0.70,
    tail_factor: float | None = None,
) -> ResultadoIBNR:
    """
    Método Bornhuetter-Ferguson.

    Ancora a projeção dos anos imaturos na taxa de perda esperada (ELR),
    reduzindo a instabilidade do Chain Ladder para anos de ocorrência recentes.

    Ultimate_BF = Pago + (1 - 1/CDF) × ELR × Prêmio
    """
    # Primeiro rodar CL para obter CDFs e link-ratios
    cl = chain_ladder(triangle, tail_factor)

    # % não desenvolvido = 1 - 1/CDF
    pct_nao_dev = 1.0 - 1.0 / cl.cdfs[_coluna_atual(triangle)]
    pct_nao_dev = pct_nao_dev.clip(0, 1)

    # Perda esperada não reportada
    perda_esperada_nr = pct_nao_dev * elr * premio

    ultimates_bf = cl.pagos + perda_esperada_nr
    ultimates_bf.name = "ultimate_bf"

    ibnr_bf = ultimates_bf - cl.pagos
    ibnr_bf.name = "ibnr_bf"

    pct_dev = (cl.pagos / ultimates_bf).clip(0, 1)

    logger.info("Bornhuetter-Ferguson (ELR=%.2f) — IBNR total: $%.0f", elr, ibnr_bf.sum())

    return ResultadoIBNR(
        metodo=f"Bornhuetter-Ferguson (ELR={elr:.0%})",
        lob="",
        link_ratios=cl.link_ratios,
        tail_factor=cl.tail_factor,
        cdfs=cl.cdfs,
        ultimates=ultimates_bf,
        pagos=cl.pagos,
        ibnr=ibnr_bf,
        ibnr_total=float(ibnr_bf.sum()),
        pct_desenvolvido=pct_dev,
    )


# ─────────────────────────────────────────────
# Helpers internos
# ─────────────────────────────────────────────

def _diagonal_atual(tri: pd.DataFrame) -> pd.Series:
    """Retorna o valor mais recente disponível (diagonal) para cada ano."""
    result = {}
    for ano in tri.index:
        row = tri.loc[ano].dropna()
        result[ano] = row.iloc[-1] if not row.empty else np.nan
    return pd.Series(result, name="pago")


def _coluna_atual(tri: pd.DataFrame) -> pd.Index:
    """Retorna a coluna (período de desenvolvimento) mais recente por ano."""
    result = {}
    for ano in tri.index:
        row = tri.loc[ano].dropna()
        result[ano] = row.index[-1] if not row.empty else tri.columns[-1]
    return pd.Series(result)


def _estimar_tail(link_ratios: np.ndarray, min_lr: float = 1.001) -> float:
    """
    Estima fator de cauda via regressão exponencial nos últimos link-ratios.

    Ajusta ln(LR - 1) = a + b*t e extrapola para t → ∞, depois integra
    para obter o produto cumulativo da cauda.
    """
    lrs = link_ratios[link_ratios > min_lr]
    if len(lrs) < 3:
        return 1.005

    y = np.log(lrs - 1.0)
    x = np.arange(len(y), dtype=float)

    # Regressão linear simples em ln(LR-1) ~ a + b*t
    b, a = np.polyfit(x, y, 1)

    if b >= 0:  # curva não convergente — usar média simples dos últimos 2
        return float(np.mean(lrs[-2:]) ** 0.5)

    # Somar projeção de mais 20 períodos
    tail = 1.0
    for k in range(1, 21):
        t = len(lrs) + k - 1
        lr_proj = 1.0 + np.exp(a + b * t)
        tail *= lr_proj

    return float(max(tail, 1.0005))
