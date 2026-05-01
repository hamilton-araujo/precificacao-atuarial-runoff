"""
Relatório executivo do Conselho — Atuarial + Solvência II + Resseguro.

Por que existe:
    O Conselho de Administração não lê triângulo. Ele quer 1 página com
    três decisões claras: (1) IBNR a provisionar, (2) prêmio puro 2025,
    (3) precisa de mais resseguro? Tudo sob ótica regulatória Susep.

Saídas em output/:
    - relatorio_conselho.md          # ⭐ markdown executivo
    - mack_msep_charts.png           # ⭐ erro de previsão por ano (Mack)
    - solvencia_scr.png              # ⭐ decomposição SCR + ratio
    - resseguro_xol_dist.png         # ⭐ distribuição custo XoL Monte Carlo

Execução:
    python -m src.exec_report --lob liability --capital-proprio 50000000
"""

import argparse
import io
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import ingest
from src import triangle as tri_mod
from src import pricing as prc_mod
from src.mack_stochastic import mack_chain_ladder
from src.solvency import calcular as calcular_solvencia
from src.reinsurance import (
    TratadoXoL, precificar as precificar_xol, calibrar_de_triangulo,
)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _decisao_provisao(cv: float, scr_ratio: float) -> tuple[str, list[str]]:
    razoes = []
    if scr_ratio >= 2.0 and cv <= 0.15:
        razoes.append(f"SCR Ratio = {scr_ratio:.2f}x ≥ 2.0 (capital robusto)")
        razoes.append(f"CV do IBNR = {cv*100:.1f}% ≤ 15% (provisão estável)")
        return "APROVAR_BEST_ESTIMATE", razoes
    if scr_ratio < 1.0:
        razoes.append(f"SCR Ratio = {scr_ratio:.2f}x < 1.0 — INSOLVÊNCIA REGULATÓRIA")
        return "INTERVENCAO_NECESSARIA", razoes
    if cv > 0.30:
        razoes.append(f"CV do IBNR = {cv*100:.1f}% > 30% (provisão volátil)")
        return "PROVISIONAR_NO_QUANTIL_75", razoes
    razoes.append(f"SCR Ratio = {scr_ratio:.2f}x · CV = {cv*100:.1f}%")
    return "PROVISIONAR_NO_BEST_ESTIMATE", razoes


def _grafico_mack(mack, lob_nome, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    anos = mack.ultimates.index
    msep = np.sqrt(mack.msep_por_ano.values)

    axes[0].bar(anos.astype(str), mack.ultimates.values / 1e6,
                color="#3498db", alpha=0.8, label="Best Estimate")
    axes[0].errorbar(anos.astype(str), mack.ultimates.values / 1e6,
                     yerr=msep / 1e6, fmt="none", color="black",
                     capsize=4, label="±1 σ Mack")
    axes[0].set_ylabel("Ultimate (USD M)")
    axes[0].set_xlabel("Ano de ocorrência")
    axes[0].set_title(f"{lob_nome} — Best Estimate ± Mack MSEP")
    axes[0].legend()
    axes[0].grid(alpha=0.3, axis="y")

    cv = mack.cv_por_ano.values * 100
    cores = ["#27ae60" if c < 15 else "#f39c12" if c < 30 else "#c0392b" for c in cv]
    axes[1].bar(anos.astype(str), cv, color=cores, alpha=0.85)
    axes[1].axhline(15, color="green", ls="--", lw=1)
    axes[1].axhline(30, color="red", ls="--", lw=1)
    axes[1].set_ylabel("CV (%) — coeficiente de variação")
    axes[1].set_xlabel("Ano de ocorrência")
    axes[1].set_title("Volatilidade da Provisão por Ano")
    axes[1].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _grafico_solvencia(solv, out: Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    componentes = ["SCR Reservas", "SCR Prêmio", "SCR Total\n(diversificado)"]
    valores = [solv.scr_reservas / 1e6, solv.scr_premio / 1e6, solv.scr_total / 1e6]
    cores = ["#3498db", "#9b59b6", "#e74c3c"]

    bars = ax.bar(componentes, valores, color=cores, alpha=0.85)
    for bar, v in zip(bars, valores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"R$ {v:.1f}M", ha="center", fontsize=9, fontweight="bold")

    ax.axhline(solv.capital_proprio / 1e6, color="green", lw=2.5, ls="--",
               label=f"Capital Próprio: R$ {solv.capital_proprio/1e6:.1f}M")
    ax.set_ylabel("Capital Requerido (R$ M)")
    ax.set_title(f"Decomposição SCR — Ratio {solv.scr_ratio:.2f}x ({solv.classificacao})")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _grafico_xol(precificacao, out: Path):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.hist(precificacao.distribuicao / 1e6, bins=80, color="#9b59b6", alpha=0.75)
    ax.axvline(precificacao.premio_puro_resseguro / 1e6, color="red", lw=2,
               label=f"Prêmio Puro: R$ {precificacao.premio_puro_resseguro/1e6:.2f}M")
    p99 = float(np.percentile(precificacao.distribuicao, 99))
    ax.axvline(p99 / 1e6, color="black", lw=1.5, ls="--",
               label=f"P99: R$ {p99/1e6:.2f}M")
    ax.set_xlabel("Custo Anual do Ressegurador (R$ M)")
    ax.set_ylabel("Frequência")
    ax.set_title(f"Distribuição XoL Monte Carlo — RoL {precificacao.rate_on_line*100:.2f}% · "
                 f"CV {precificacao.cv_anual:.2f}")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _gerar_markdown(lob_nome, resultado_cl, mack, preco, solv, xol, decisao, razoes, out: Path):
    badge = {
        "APROVAR_BEST_ESTIMATE": "✅",
        "PROVISIONAR_NO_BEST_ESTIMATE": "⚠️",
        "PROVISIONAR_NO_QUANTIL_75": "⚠️",
        "INTERVENCAO_NECESSARIA": "❌",
    }[decisao]

    lines = [
        f"# Relatório do Conselho — {lob_nome}",
        "",
        f"## Recomendação: {badge} **{decisao.replace('_', ' ').title()}**",
        "",
    ]
    for r in razoes:
        lines.append(f"- {r}")

    lines += [
        "",
        "---",
        "",
        "## 1. Provisão Técnica (Mack Stochastic CL)",
        "",
        "| Métrica | Valor |",
        "|---|---|",
        f"| Best Estimate (E[IBNR]) | R$ {resultado_cl.ibnr_total/1e6:.2f}M |",
        f"| MSEP (raiz quadrada) | R$ {np.sqrt(mack.msep_total)/1e6:.2f}M |",
        f"| Coeficiente de Variação | {mack.cv_total*100:.1f}% |",
        f"| Quantil 75 (provisão prudente) | R$ {mack.quantil_75/1e6:.2f}M |",
        f"| Quantil 99.5 (Solvência II) | R$ {mack.quantil_99_5/1e6:.2f}M |",
        f"| Tail Factor estimado | {resultado_cl.tail_factor:.4f} |",
        "",
        "![Mack](mack_msep_charts.png)",
        "",
        "## 2. Precificação 2025",
        "",
        "| Métrica | Valor |",
        "|---|---|",
        f"| Trend Factor inflacionário | {preco.trend_factor:.4f} |",
        f"| Burning Cost médio | R$ {preco.burning_cost.mean()/1e3:.2f}K |",
        f"| Prêmio Puro | R$ {preco.premio_puro/1e6:.2f}M |",
        f"| Loading | {preco.loading_factor*100:.0f}% |",
        f"| Prêmio Comercial | R$ {preco.premio_comercial/1e6:.2f}M |",
        "",
        "## 3. Solvência II (Susep)",
        "",
        "| Componente | Valor |",
        "|---|---|",
        f"| Best Estimate | R$ {solv.best_estimate/1e6:.2f}M |",
        f"| Risk Margin (CoC 6%) | R$ {solv.risk_margin/1e6:.2f}M |",
        f"| SCR Reservas | R$ {solv.scr_reservas/1e6:.2f}M |",
        f"| SCR Prêmio | R$ {solv.scr_premio/1e6:.2f}M |",
        f"| **SCR Total (diversificado)** | **R$ {solv.scr_total/1e6:.2f}M** |",
        f"| Capital Próprio | R$ {solv.capital_proprio/1e6:.2f}M |",
        f"| **SCR Ratio** | **{solv.scr_ratio:.2f}x** ({solv.classificacao}) |",
        "",
        "![Solvência](solvencia_scr.png)",
        "",
        "## 4. Resseguro XoL — Sugestão de Compra",
        "",
        f"Tratado de Excess of Loss para proteção de tail risk de large losses.",
        "",
        "| Métrica | Valor |",
        "|---|---|",
        f"| Frequência large losses esperada | {xol.expected_freq_layer:.2f}/ano |",
        f"| Prêmio puro do tratado | R$ {xol.premio_puro_resseguro/1e6:.2f}M |",
        f"| Rate on Line | {xol.rate_on_line*100:.2f}% |",
        f"| CV anual do custo | {xol.cv_anual:.2f} |",
        f"| Pior ano simulado | R$ {xol.max_simulado/1e6:.2f}M |",
        "",
        "![Resseguro](resseguro_xol_dist.png)",
        "",
        "---",
        "",
        "## Critérios da decisão",
        "",
        "| Decisão | Critério |",
        "|---|---|",
        "| ✅ APROVAR Best Estimate | SCR ≥ 2.0x **E** CV ≤ 15% |",
        "| ⚠️ Provisionar no Q75 | CV > 30% (provisão volátil) |",
        "| ❌ Intervenção | SCR Ratio < 1.0x |",
        "",
        "## Metodologia",
        "",
        "- **Chain Ladder Determinístico (Mack 1993)**: link-ratios ponderados por volume.",
        "- **MSEP**: σ²_k = (1/(I-k-1)) Σ C_{i,k}·(C_{i,k+1}/C_{i,k} - f_k)². Quantis 75/99.5 via aproximação log-normal.",
        "- **SCR**: SCR_total = √(SCR_res² + SCR_pre² + 2·0.5·SCR_res·SCR_pre) (correlação Susep).",
        "- **XoL Pricing**: Monte Carlo com 50.000 anos, frequência Poisson + severidade LogNormal.",
    ]

    out.write_text("\n".join(lines), encoding="utf-8")


# ─────────────────────────────────────────────────────────
# Orquestração
# ─────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lob", choices=["auto", "liability", "property"], default="liability")
    p.add_argument("--capital-proprio", type=float, default=80_000_000.0,
                   help="Capital próprio da seguradora (R$)")
    p.add_argument("--xol-priority", type=float, default=5_000_000.0,
                   help="Priority do tratado XoL (retenção em R$)")
    p.add_argument("--xol-limit", type=float, default=15_000_000.0,
                   help="Limite da camada XoL em R$")
    args = p.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    lob_nome = {"auto": "Auto", "liability": "Responsabilidade Civil",
                "property": "Patrimonial"}[args.lob]

    logger.info("Carregando triângulo %s...", args.lob)
    dados = ingest.carregar(args.lob)

    # Chain Ladder determinístico (provisão pontual)
    cl = tri_mod.chain_ladder(dados["triangle"])
    cl.lob = args.lob

    # Mack estocástico (CI nas reservas)
    mack = mack_chain_ladder(dados["triangle"])

    # Precificação
    preco = prc_mod.calcular_premio(
        dados["triangle"], dados["exposicao"], dados["premio"], cl.ultimates,
        trend_anual=0.04, anos_base=5, loading_factor=0.25, lob=args.lob,
    )

    # Solvência II
    premio_anual = float(dados["premio"].iloc[-1])
    solv = calcular_solvencia(mack, premio_anual, args.capital_proprio,
                               ibnr_total=cl.ibnr_total)

    # Resseguro XoL — params típicos por LoB (calibração simulada de mercado)
    xol_params = {
        "auto":      (8.0, 13.5, 0.9),    # menor severidade, mais frequente
        "liability": (4.0, 14.5, 1.2),    # maior severidade tail
        "property":  (5.0, 14.0, 1.0),
    }
    lambda_freq, mu_ln, sigma_ln = xol_params[args.lob]

    tratado = TratadoXoL(
        priority=args.xol_priority, limit=args.xol_limit,
        lambda_freq=lambda_freq, mu_ln=mu_ln, sigma_ln=sigma_ln,
    )
    xol = precificar_xol(tratado, n_anos=50_000)

    # Decisão
    decisao, razoes = _decisao_provisao(mack.cv_total, solv.scr_ratio)

    # Charts + markdown
    _grafico_mack(mack, lob_nome, OUTPUT_DIR / "mack_msep_charts.png")
    _grafico_solvencia(solv, OUTPUT_DIR / "solvencia_scr.png")
    _grafico_xol(xol, OUTPUT_DIR / "resseguro_xol_dist.png")
    _gerar_markdown(lob_nome, cl, mack, preco, solv, xol, decisao, razoes,
                    OUTPUT_DIR / "relatorio_conselho.md")

    # Painel CLI
    badge = {"APROVAR_BEST_ESTIMATE": "✅", "PROVISIONAR_NO_BEST_ESTIMATE": "⚠️",
             "PROVISIONAR_NO_QUANTIL_75": "⚠️", "INTERVENCAO_NECESSARIA": "❌"}[decisao]
    print(f"\n{'═'*60}")
    print(f"  CONSELHO DE ADMINISTRAÇÃO — {lob_nome}")
    print(f"{'═'*60}")
    print(f"  Best Estimate IBNR     R$ {cl.ibnr_total/1e6:>8.2f} M")
    print(f"  Mack MSEP (√)          R$ {np.sqrt(mack.msep_total)/1e6:>8.2f} M  (CV={mack.cv_total*100:.1f}%)")
    print(f"  Quantil 99.5 (SII)     R$ {mack.quantil_99_5/1e6:>8.2f} M")
    print(f"  Prêmio Comercial 2025  R$ {preco.premio_comercial/1e6:>8.2f} M")
    print(f"  SCR Ratio              {solv.scr_ratio:>8.2f}x  ({solv.classificacao})")
    print(f"  Resseguro XoL premium  R$ {xol.premio_puro_resseguro/1e6:>8.2f} M  (RoL {xol.rate_on_line*100:.2f}%)")
    print(f"{'═'*60}")
    print(f"  {badge}  DECISÃO: {decisao.replace('_', ' ').title()}")
    for r in razoes:
        print(f"     · {r}")
    print(f"{'═'*60}\n")
    print(f"Relatório salvo: {OUTPUT_DIR}/relatorio_conselho.md\n")


if __name__ == "__main__":
    main()
