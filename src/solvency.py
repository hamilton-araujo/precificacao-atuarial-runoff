"""
Solvência II / Susep — Best Estimate, Risk Margin e SCR.

Por que existe:
    O regulador (Susep no Brasil, EIOPA na UE) exige que toda seguradora
    demonstre solvência sob choque adverso 99.5% (1-em-200 anos).
    Resultado contábil = Best Estimate + Risk Margin. Se o Capital Próprio
    < SCR, a seguradora é intervindida.

Fórmulas (simplificadas):
    Best Estimate = E[IBNR] = Mack ultimate
    Risk Margin   = CoC × Σ_t SCR(t) / (1+r)^t
    SCR_reservas  = q99.5(IBNR) - E[IBNR]   (via aproximação log-normal de Mack)
    SCR_premio    = σ_premio × Volume × 3.0 (USP padrão)
    SCR_total     = √(SCR_res² + SCR_pre² + 2·ρ·SCR_res·SCR_pre)
    SCR_ratio     = Capital_Proprio / SCR_total
"""

from dataclasses import dataclass

import numpy as np
from scipy import stats

from src.mack_stochastic import MackResultado

COST_OF_CAPITAL = 0.06    # CoC regulatório
TAXA_DESCONTO = 0.10      # taxa livre de risco (Selic Brasil ~10.5%)
CORRELACAO_RES_PREM = 0.50  # padrão Susep
SIGMA_PREMIUM = 0.10      # USP padrão (Premium Risk)


@dataclass
class SolvencyResultado:
    best_estimate:      float
    risk_margin:        float
    scr_reservas:       float
    scr_premio:         float
    scr_total:          float
    capital_proprio:    float
    scr_ratio:          float       # CP / SCR (≥ 1.0 = solvente)
    classificacao:      str         # SOLVENTE_FORTE / SOLVENTE / SOB_INTERVENCAO


def calcular(
    mack: MackResultado,
    premio_anual: float,
    capital_proprio: float,
    ibnr_total: float,
    horizonte_runoff_anos: int = 5,
) -> SolvencyResultado:
    """
    Calcula SCR completo e classifica a seguradora.

    Args:
        mack:           resultado Mack com quantis 75/99.5
        premio_anual:   prêmio anual de referência
        capital_proprio: capital próprio elegível
        ibnr_total:     IBNR best estimate (E[IBNR])
    """
    be = float(ibnr_total)
    # Quantil 99.5 do IBNR via aproximação log-normal usando CV total de Mack
    if be > 0 and mack.cv_total > 0:
        sigma_ln = np.sqrt(np.log(1 + mack.cv_total ** 2))
        mu_ln = np.log(be) - 0.5 * sigma_ln ** 2
        ibnr_q995 = float(np.exp(mu_ln + sigma_ln * stats.norm.ppf(0.995)))
    else:
        ibnr_q995 = be * 1.5
    scr_res = max(ibnr_q995 - be, 0.0)

    # SCR prêmio (USP simplificado)
    scr_pre = SIGMA_PREMIUM * premio_anual * 3.0

    # Agregação SCR (Susep correlation)
    scr_total = (scr_res ** 2 + scr_pre ** 2 +
                 2 * CORRELACAO_RES_PREM * scr_res * scr_pre) ** 0.5

    # Risk Margin via Cost of Capital approach (escada de runoff)
    runoff_factor = sum((1 - t / horizonte_runoff_anos) /
                        (1 + TAXA_DESCONTO) ** t
                        for t in range(horizonte_runoff_anos + 1))
    risk_margin = COST_OF_CAPITAL * scr_total * runoff_factor / horizonte_runoff_anos

    scr_ratio = capital_proprio / scr_total if scr_total > 0 else float("inf")

    if scr_ratio >= 2.0:
        classif = "SOLVENTE_FORTE"
    elif scr_ratio >= 1.0:
        classif = "SOLVENTE"
    else:
        classif = "SOB_INTERVENCAO"

    return SolvencyResultado(
        best_estimate=be,
        risk_margin=risk_margin,
        scr_reservas=scr_res,
        scr_premio=scr_pre,
        scr_total=scr_total,
        capital_proprio=capital_proprio,
        scr_ratio=scr_ratio,
        classificacao=classif,
    )
