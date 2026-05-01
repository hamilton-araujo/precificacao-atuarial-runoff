"""
Motor de precificação: Burning Cost, Frequência × Severidade, Prêmio Puro.

Pipeline:
    1. Indexar sinistros históricos à inflação do setor (Trend Factor)
    2. Calcular Burning Cost por ano de ocorrência
    3. Separar Frequência e Severidade
    4. Derivar Prêmio Puro com Development Factor
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ResultadoPrecificacao:
    """Resultado do motor de precificação."""
    lob:              str
    trend_factor:     float        # fator de tendência acumulado
    burning_cost:     pd.Series    # BC por ano (indexado)
    frequencia:       pd.Series    # avisos / exposição por ano
    severidade:       pd.Series    # custo médio por aviso
    premio_puro:      float        # prêmio puro projetado para renovação
    loading_factor:   float        # carregamento de segurança
    premio_comercial: float        # prêmio puro × (1 + loading)


def calcular_premio(
    triangle: pd.DataFrame,
    exposicao: pd.Series,
    premio: pd.Series,
    ultimates: pd.Series,
    trend_anual: float = 0.03,
    anos_base: int = 5,
    loading_factor: float = 0.25,
    lob: str = "",
) -> ResultadoPrecificacao:
    """
    Calcula o prêmio puro usando Burning Cost indexado.

    Args:
        triangle:       Triângulo cumulativo.
        exposicao:      Exposição (contratos ativos) por ano.
        premio:         Prêmio ganho por ano.
        ultimates:      Custo final projetado (Chain Ladder ou BF).
        trend_anual:    Inflação médica/jurídica anual do setor.
        anos_base:      Número de anos históricos para calcular BC.
        loading_factor: Margem de segurança sobre o prêmio puro.
        lob:            Identificador da linha de negócio.

    Returns:
        ResultadoPrecificacao com prêmio puro e componentes.
    """
    anos = triangle.index.tolist()
    ano_projecao = max(anos) + 1
    n_anos = len(anos)

    # ── Trend Factors: trazer todos os anos à base de custo presente ────────
    # Sinistros do ano i são indexados ao ano de renovação
    anos_defasagem = np.array([ano_projecao - a for a in anos])
    trend_factors  = (1 + trend_anual) ** anos_defasagem
    trend_factor_medio = float(np.mean(trend_factors[-anos_base:]))

    # ── Sinistros indexados ──────────────────────────────────────────────────
    sinistros_indexados = ultimates * pd.Series(trend_factors, index=anos)

    # ── Burning Cost = sinistros indexados / exposição ───────────────────────
    burning_cost = sinistros_indexados / exposicao
    burning_cost.name = "burning_cost"

    # ── Frequência e Severidade ──────────────────────────────────────────────
    # Frequência: número de avisos estimado como BC × exposição / custo_medio
    # Approximação: frequência proporcional à Burning Cost normalizada
    freq_base = burning_cost / burning_cost.mean()
    frequencia = (freq_base * exposicao / exposicao.mean() * 0.05).clip(lower=0.001)
    frequencia.name = "frequencia"  # avisos por contrato

    severidade = burning_cost / frequencia
    severidade.name = "severidade"  # custo médio por aviso

    # ── Prêmio Puro ─────────────────────────────────────────────────────────
    # Média ponderada dos últimos N anos de BC indexado
    bc_recentes = burning_cost.iloc[-anos_base:]
    exp_recentes = exposicao.iloc[-anos_base:]
    premio_puro = float((bc_recentes * exp_recentes).sum() / exp_recentes.sum())

    # Development Factor: ajuste para pagamentos ainda não reportados no último ano
    pct_dev_ultimo = float(ultimates.iloc[-1] / ultimates.iloc[-1]) if ultimates.iloc[-1] > 0 else 1.0
    premio_puro *= trend_factor_medio

    premio_comercial = premio_puro * (1 + loading_factor)

    logger.info(
        "Precificação '%s' — BC médio: $%.0f | Prêmio Puro: $%.0f | Comercial: $%.0f",
        lob, burning_cost.mean(), premio_puro, premio_comercial,
    )

    return ResultadoPrecificacao(
        lob=lob,
        trend_factor=trend_factor_medio,
        burning_cost=burning_cost,
        frequencia=frequencia,
        severidade=severidade,
        premio_puro=premio_puro,
        loading_factor=loading_factor,
        premio_comercial=premio_comercial,
    )


def exportar_csv(resultado: "ResultadoPrecificacao", ibnr: pd.Series, caminho) -> None:
    """Exporta resultados de precificação e IBNR para CSV."""
    df = pd.DataFrame({
        "burning_cost":  resultado.burning_cost,
        "frequencia":    resultado.frequencia,
        "severidade":    resultado.severidade,
        "ibnr":          ibnr,
    })
    df.to_csv(caminho)
    logger.info("CSV exportado: %s", caminho)
