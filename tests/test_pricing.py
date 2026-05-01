"""Testes do motor de precificação — Burning Cost e Prêmio Puro."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pricing import calcular_premio
from ingest import carregar
from triangle import chain_ladder


def _dados_simples():
    anos = list(range(2015, 2025))
    data = {}
    for j, col in enumerate(range(2015, 2020)):
        col_data = []
        for i in range(10):
            if i + j < 10:
                col_data.append(1000.0 * (i + 1) * (j + 1) * 0.8)
            else:
                col_data.append(np.nan)
        data[col] = col_data

    tri = pd.DataFrame(data, index=anos)
    tri.index.name = "ano_ocorrencia"
    exposicao = pd.Series([10000 + i * 200 for i in range(10)], index=anos)
    premio    = pd.Series([8000  + i * 150 for i in range(10)], index=anos)
    return tri, exposicao, premio


class TestBurningCost:
    def test_premio_puro_positivo(self):
        tri, exp, prem = _dados_simples()
        res_cl = chain_ladder(tri)
        preco = calcular_premio(tri, exp, prem, res_cl.ultimates, lob="auto")
        assert preco.premio_puro > 0

    def test_premio_comercial_maior_que_puro(self):
        tri, exp, prem = _dados_simples()
        res_cl = chain_ladder(tri)
        preco = calcular_premio(tri, exp, prem, res_cl.ultimates,
                                loading_factor=0.25, lob="auto")
        assert preco.premio_comercial > preco.premio_puro

    def test_burning_cost_serie_completa(self):
        tri, exp, prem = _dados_simples()
        res_cl = chain_ladder(tri)
        preco = calcular_premio(tri, exp, prem, res_cl.ultimates, lob="auto")
        assert len(preco.burning_cost) == len(tri.index)
        assert (preco.burning_cost > 0).all()

    def test_trend_maior_aumenta_premio(self):
        tri, exp, prem = _dados_simples()
        res_cl = chain_ladder(tri)
        p_low  = calcular_premio(tri, exp, prem, res_cl.ultimates, trend_anual=0.01)
        p_high = calcular_premio(tri, exp, prem, res_cl.ultimates, trend_anual=0.08)
        assert p_high.premio_puro >= p_low.premio_puro

    def test_loading_zero_comercial_igual_puro(self):
        tri, exp, prem = _dados_simples()
        res_cl = chain_ladder(tri)
        preco = calcular_premio(tri, exp, prem, res_cl.ultimates, loading_factor=0.0)
        assert abs(preco.premio_comercial - preco.premio_puro) < 1e-6

    def test_frequencia_positiva(self):
        tri, exp, prem = _dados_simples()
        res_cl = chain_ladder(tri)
        preco = calcular_premio(tri, exp, prem, res_cl.ultimates, lob="auto")
        assert (preco.frequencia > 0).all()

    def test_severidade_positiva(self):
        tri, exp, prem = _dados_simples()
        res_cl = chain_ladder(tri)
        preco = calcular_premio(tri, exp, prem, res_cl.ultimates, lob="auto")
        assert (preco.severidade > 0).all()

    @pytest.mark.parametrize("lob", ["auto", "liability", "property"])
    def test_premio_por_lob(self, lob):
        dados = carregar(lob)
        res = chain_ladder(dados["triangle"])
        preco = calcular_premio(
            dados["triangle"], dados["exposicao"], dados["premio"],
            res.ultimates, lob=lob,
        )
        assert preco.premio_puro > 0
        assert preco.premio_comercial > preco.premio_puro
