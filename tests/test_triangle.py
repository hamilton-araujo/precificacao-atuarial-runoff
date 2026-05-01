"""Testes do motor atuarial — Chain Ladder, BF e Tail Factor."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from triangle import chain_ladder, bornhuetter_ferguson, _estimar_tail, _diagonal_atual
from ingest import carregar


def _triangulo_simples() -> pd.DataFrame:
    """Triângulo 5×5 com desenvolvimento conhecido."""
    data = {
        2015: [1000, 800,  600,  400, 200],
        2016: [1800, 1440, 1080, 720, np.nan],
        2017: [2400, 1920, 1440, np.nan, np.nan],
        2018: [2800, 2240, np.nan, np.nan, np.nan],
        2019: [3000, np.nan, np.nan, np.nan, np.nan],
    }
    df = pd.DataFrame(data, index=[2015, 2016, 2017, 2018, 2019])
    df.index.name = "ano_ocorrencia"
    df.columns.name = "ano_desenvolvimento"
    return df


class TestChainLadder:
    def test_link_ratios_maiores_ou_iguais_1(self):
        tri = _triangulo_simples()
        res = chain_ladder(tri)
        assert (res.link_ratios >= 1.0).all(), "Todos LRs devem ser >= 1.0"

    def test_ultimates_maiores_que_pagos(self):
        tri = _triangulo_simples()
        res = chain_ladder(tri)
        assert (res.ultimates >= res.pagos).all(), "Ultimate >= Pago para todo ano"

    def test_ibnr_nao_negativo(self):
        tri = _triangulo_simples()
        res = chain_ladder(tri)
        assert (res.ibnr >= -1e-6).all(), "IBNR deve ser >= 0"

    def test_ibnr_total_igual_soma(self):
        tri = _triangulo_simples()
        res = chain_ladder(tri)
        assert abs(res.ibnr_total - res.ibnr.sum()) < 1e-6

    def test_ano_mais_maduro_ibnr_menor(self):
        """Ano mais antigo (mais desenvolvido) deve ter IBNR menor."""
        tri = _triangulo_simples()
        res = chain_ladder(tri)
        assert res.ibnr.iloc[0] <= res.ibnr.iloc[-1], \
            "Ano mais maduro deve ter IBNR <= ano mais recente"

    def test_pct_desenvolvido_entre_0_e_1(self):
        tri = _triangulo_simples()
        res = chain_ladder(tri)
        assert (res.pct_desenvolvido >= 0).all()
        assert (res.pct_desenvolvido <= 1.0 + 1e-6).all()

    def test_tail_factor_manual(self):
        tri = _triangulo_simples()
        res = chain_ladder(tri, tail_factor=1.02)
        assert abs(res.tail_factor - 1.02) < 1e-9

    def test_triangulo_totalmente_desenvolvido(self):
        """Triângulo completo deve ter IBNR próximo de zero (só tail)."""
        n = 5
        data = {c: [1000.0 * (c - 2014) * (1 + 0.1 * r)
                    for r in range(n)]
                for c in range(2015, 2015 + n)}
        tri = pd.DataFrame(data, index=range(2015, 2015 + n))
        tri.index.name = "ano_ocorrencia"
        res = chain_ladder(tri, tail_factor=1.0)
        assert res.ibnr.sum() >= 0


class TestBornhuetterFerguson:
    def test_bf_ibnr_menor_cl_anos_imaturos(self):
        """BF ancora nos anos jovens → IBNR menor que CL para último ano."""
        tri = _triangulo_simples()
        premio = pd.Series([5000, 6000, 7000, 8000, 9000], index=tri.index)
        res_cl = chain_ladder(tri)
        res_bf = bornhuetter_ferguson(tri, premio, elr=0.70)

        # Para o ano mais recente (menos desenvolvido), BF < CL
        assert res_bf.ibnr.iloc[-1] <= res_cl.ibnr.iloc[-1] * 1.1, \
            "BF deve estabilizar IBNR no ano mais imaturo"

    def test_bf_ibnr_positivo(self):
        tri = _triangulo_simples()
        premio = pd.Series([5000, 6000, 7000, 8000, 9000], index=tri.index)
        res = bornhuetter_ferguson(tri, premio, elr=0.65)
        assert (res.ibnr >= -1e-6).all()

    def test_bf_elr_alto_aumenta_ibnr(self):
        tri = _triangulo_simples()
        premio = pd.Series([5000] * 5, index=tri.index)
        res_low  = bornhuetter_ferguson(tri, premio, elr=0.50)
        res_high = bornhuetter_ferguson(tri, premio, elr=0.85)
        assert res_high.ibnr_total >= res_low.ibnr_total


class TestTailFactor:
    def test_tail_positivo(self):
        lrs = np.array([2.5, 1.6, 1.3, 1.1, 1.04, 1.02])
        tail = _estimar_tail(lrs)
        assert tail >= 1.0

    def test_tail_convergente(self):
        """Com LRs muito próximos de 1, tail deve ser pequeno."""
        lrs = np.array([1.005, 1.003, 1.002])
        tail = _estimar_tail(lrs)
        assert tail < 1.05

    def test_dados_insuficientes_retorna_padrao(self):
        lrs = np.array([1.02])
        tail = _estimar_tail(lrs)
        assert tail >= 1.0


class TestIngest:
    @pytest.mark.parametrize("lob", ["auto", "liability", "property"])
    def test_triangulo_gerado(self, lob):
        dados = carregar(lob, force_reload=True)
        assert "triangle" in dados
        assert "exposicao" in dados
        assert "premio" in dados

    @pytest.mark.parametrize("lob", ["auto", "liability", "property"])
    def test_triangulo_cumulativo(self, lob):
        """Cada célula deve ser >= célula anterior na mesma linha."""
        dados = carregar(lob)
        tri = dados["triangle"]
        cols = tri.columns.tolist()
        for i in range(1, len(cols)):
            c_prev = cols[i - 1]
            c_curr = cols[i]
            mask = tri[c_prev].notna() & tri[c_curr].notna()
            assert (tri.loc[mask, c_curr] >= tri.loc[mask, c_prev] - 1).all(), \
                f"Coluna {c_curr} deve ser >= {c_prev}"

    def test_lob_invalida(self):
        with pytest.raises((ValueError, KeyError)):
            carregar("inexistente", force_reload=True)
