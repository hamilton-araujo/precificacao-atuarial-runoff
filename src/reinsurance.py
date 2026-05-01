"""
Pricing de tratado de Resseguro Excess of Loss (XoL).

Por que existe:
    Eventos catastróficos (large losses) violam premissas do Chain Ladder.
    Para cobrir o tail risk, seguradoras compram resseguro XoL: o
    ressegurador paga toda perda acima de um limite (priority/retention)
    até um teto (limit). O atuário precisa precificar este tratado.

Modelo:
    Frequência de large losses ~ Poisson(λ)
    Severidade dos large losses ~ LogNormal(μ, σ)  (Pareto também válido)

    Prêmio Puro XoL = E[Σ min(max(X_i - prio, 0), limit)]
                    = λ · E[min(max(X - prio, 0), limit)]

Cálculo Monte Carlo:
    1. Simula 50.000 anos
    2. Para cada ano: nº sinistros ~ Poisson(λ), severidades ~ LogNormal
    3. Aplica camada XoL (priority, limit) e calcula custo do ressegurador
    4. Devolve Prêmio Puro + CV + máximo histórico simulado
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class TratadoXoL:
    priority:    float    # ponto de retenção (R$)
    limit:       float    # limite da camada (R$)
    lambda_freq: float    # frequência anual de large losses
    mu_ln:       float    # μ do log-severidade
    sigma_ln:    float    # σ do log-severidade


@dataclass
class PrecificacaoXoL:
    premio_puro_resseguro: float    # R$/ano
    rate_on_line:          float    # premio / limite (% do limite)
    cv_anual:              float    # coeficiente de variação do custo anual
    expected_freq_layer:   float    # nº médio de sinistros que entram na camada
    max_simulado:          float    # pior ano nos 50k simulados
    distribuicao:          np.ndarray   # custo anual simulado (50k pontos)


def precificar(
    tratado: TratadoXoL,
    n_anos: int = 50_000,
    seed: int = 42,
) -> PrecificacaoXoL:
    """Monte Carlo do prêmio puro XoL."""
    rng = np.random.default_rng(seed)

    custos = np.zeros(n_anos)
    n_layer_total = 0

    for i in range(n_anos):
        n_claims = rng.poisson(tratado.lambda_freq)
        if n_claims == 0:
            continue
        sevs = rng.lognormal(tratado.mu_ln, tratado.sigma_ln, n_claims)
        layer = np.minimum(np.maximum(sevs - tratado.priority, 0), tratado.limit)
        custos[i] = layer.sum()
        n_layer_total += (layer > 0).sum()

    premio_puro = float(custos.mean())
    cv = float(custos.std() / premio_puro) if premio_puro > 0 else 0.0

    return PrecificacaoXoL(
        premio_puro_resseguro=premio_puro,
        rate_on_line=premio_puro / tratado.limit,
        cv_anual=cv,
        expected_freq_layer=n_layer_total / n_anos,
        max_simulado=float(custos.max()),
        distribuicao=custos,
    )


def calibrar_de_triangulo(
    valores_sinistros: np.ndarray,
    limiar_large_loss_quantil: float = 0.95,
) -> tuple[float, float, float]:
    """
    Calibra λ, μ, σ a partir de dados históricos de sinistros.

    Returns:
        (lambda_anual, mu_ln, sigma_ln)
    """
    valores = valores_sinistros[valores_sinistros > 0]
    threshold = np.quantile(valores, limiar_large_loss_quantil)
    large = valores[valores >= threshold]

    if len(large) < 5:
        return 0.0, 0.0, 0.0

    log_large = np.log(large)
    mu = float(log_large.mean())
    sigma = float(log_large.std())

    n_anos_hist = 5  # assume janela de 5 anos
    lambda_anual = len(large) / n_anos_hist

    return lambda_anual, mu, sigma
