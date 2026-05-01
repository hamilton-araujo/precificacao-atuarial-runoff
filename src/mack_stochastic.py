"""
Mack Stochastic Chain Ladder (Mack 1993).

Por que existe:
    Chain Ladder determinístico devolve UM número de IBNR. Mas reservas
    técnicas em seguros são distribuições, não pontos. O regulador
    (Susep / Solvência II) exige Best Estimate + Mean Squared Error
    of Prediction (MSEP) para calibrar a Risk Margin.

    Mack (1993) deriva analiticamente o MSEP do Chain Ladder sem
    assumir distribuição paramétrica — apenas martingale-like behavior
    nos link-ratios. O resultado: σ²_k por período de desenvolvimento
    e MSEP total agregado.

Fórmulas:
    σ²_k = (1/(I-k-1)) Σ_i C_{i,k} · (C_{i,k+1}/C_{i,k} - f_k)²
    MSEP(C_{i,J}) = C_{i,J}² · Σ_k [σ²_k/f_k² · (1/C_{i,k} + 1/Σ C_{i,k})]
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class MackResultado:
    link_ratios:  np.ndarray
    sigmas2:      np.ndarray   # σ²_k por período
    ultimates:    pd.Series    # mesmo do CL determinístico
    msep_por_ano: pd.Series    # MSE de previsão por ano de ocorrência
    msep_total:   float        # MSE agregado
    cv_por_ano:   pd.Series    # coeficiente de variação por ano
    cv_total:     float        # CV do IBNR total
    quantil_75:   float        # IBNR no percentil 75 (lognormal aproximado)
    quantil_99_5: float        # IBNR no percentil 99.5 (Solvência II)


def mack_chain_ladder(triangle: pd.DataFrame) -> MackResultado:
    """
    Chain Ladder estocástico de Mack — link-ratios determinísticos +
    estimativa fechada do erro de previsão.
    """
    tri = triangle.copy()
    cols = tri.columns.tolist()
    I, J = tri.shape

    # ── Link-ratios ponderados ────────────────────────────────────────
    fs = np.zeros(J - 1)
    sigmas2 = np.zeros(J - 1)

    for k in range(J - 1):
        c_k = tri.iloc[:, k].values
        c_k1 = tri.iloc[:, k + 1].values
        mask = ~np.isnan(c_k) & ~np.isnan(c_k1)
        if mask.sum() < 2:
            fs[k] = 1.0
            sigmas2[k] = 0.0
            continue

        soma_k = c_k[mask].sum()
        soma_k1 = c_k1[mask].sum()
        fs[k] = soma_k1 / soma_k

        # σ²_k = (1/(I-k-1)) Σ C_{i,k} (C_{i,k+1}/C_{i,k} - f_k)²
        ratios = c_k1[mask] / c_k[mask]
        n_obs = mask.sum()
        if n_obs > 1:
            sigmas2[k] = (1.0 / (n_obs - 1)) * np.sum(c_k[mask] * (ratios - fs[k]) ** 2)
        else:
            sigmas2[k] = 0.0

    # ── Ultimates ─────────────────────────────────────────────────────
    diagonal = []
    col_atual = []
    for ano in tri.index:
        row = tri.loc[ano].dropna()
        diagonal.append(row.iloc[-1] if len(row) > 0 else np.nan)
        col_atual.append(len(row) - 1 if len(row) > 0 else 0)

    diagonal = np.array(diagonal)
    col_atual = np.array(col_atual)

    ultimates = np.zeros(I)
    for i in range(I):
        u = diagonal[i]
        for k in range(col_atual[i], J - 1):
            u *= fs[k]
        ultimates[i] = u

    # ── MSEP por ano (Mack 1993) ───────────────────────────────────────
    msep = np.zeros(I)
    for i in range(I):
        if col_atual[i] >= J - 1:
            msep[i] = 0.0
            continue
        soma = 0.0
        for k in range(col_atual[i], J - 1):
            if fs[k] == 0:
                continue
            soma_col_k = np.nansum(tri.iloc[:I - k - 1, k].values)
            c_ik_proj = diagonal[i] * np.prod(fs[col_atual[i]:k]) if k > col_atual[i] else diagonal[i]
            if c_ik_proj <= 0 or soma_col_k <= 0:
                continue
            soma += (sigmas2[k] / fs[k] ** 2) * (1.0 / c_ik_proj + 1.0 / soma_col_k)
        msep[i] = ultimates[i] ** 2 * soma

    msep_total = float(np.sum(msep))

    cv_por_ano = pd.Series(
        np.where(ultimates > 0, np.sqrt(msep) / ultimates, 0),
        index=tri.index,
    )

    ibnr_total = (ultimates - diagonal).sum()
    cv_total = float(np.sqrt(msep_total) / ibnr_total) if ibnr_total > 0 else 0.0

    # ── Aproximação log-normal para quantis ────────────────────────────
    if ibnr_total > 0 and cv_total > 0:
        sigma_ln = np.sqrt(np.log(1 + cv_total ** 2))
        mu_ln = np.log(ibnr_total) - 0.5 * sigma_ln ** 2
        q75 = float(np.exp(mu_ln + sigma_ln * stats.norm.ppf(0.75)))
        q995 = float(np.exp(mu_ln + sigma_ln * stats.norm.ppf(0.995)))
    else:
        q75, q995 = ibnr_total, ibnr_total

    return MackResultado(
        link_ratios=fs,
        sigmas2=sigmas2,
        ultimates=pd.Series(ultimates, index=tri.index, name="ultimate_mack"),
        msep_por_ano=pd.Series(msep, index=tri.index, name="msep"),
        msep_total=msep_total,
        cv_por_ano=cv_por_ano,
        cv_total=cv_total,
        quantil_75=q75,
        quantil_99_5=q995,
    )
