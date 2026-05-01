"""
CLI — Precificação Atuarial e Triângulos de Run-off.

Exemplos:
    python src/main.py --lob auto
    python src/main.py --lob liability --method bf --elr 0.72
    python src/main.py --lob property --trend 2.5 --loading 30
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Motor Atuarial — Chain Ladder + BF + Burning Cost",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--lob",     choices=["auto", "liability", "property"],
                   default="auto", help="Linha de negócio.")
    p.add_argument("--method",  choices=["chain_ladder", "bf"],
                   default="chain_ladder", help="Método atuarial de projeção.")
    p.add_argument("--elr",     type=float, default=0.70,
                   help="Taxa de Perda Esperada (apenas BF).")
    p.add_argument("--trend",   type=float, default=3.0,
                   help="Tendência inflacionária anual (%%a.a.).")
    p.add_argument("--loading", type=float, default=25.0,
                   help="Loading de segurança sobre o prêmio puro (%%)")
    p.add_argument("--tail",    type=float, default=None,
                   help="Fator de cauda manual (padrão: estimado).")
    p.add_argument("--anos-bc", type=int, default=5,
                   help="Anos históricos para cálculo do Burning Cost.")
    p.add_argument("--no-charts", action="store_true",
                   help="Suprimir geração de gráficos HTML.")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def _validar(args: argparse.Namespace) -> None:
    if not 0 < args.elr < 1:
        raise ValueError(f"--elr deve estar em (0, 1). Recebido: {args.elr}")
    if args.trend < 0:
        raise ValueError(f"--trend deve ser >= 0. Recebido: {args.trend}")
    if args.loading < 0:
        raise ValueError(f"--loading deve ser >= 0. Recebido: {args.loading}")
    if args.tail is not None and args.tail < 1.0:
        raise ValueError(f"--tail deve ser >= 1.0. Recebido: {args.tail}")


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        _validar(args)
    except ValueError as e:
        print(f"Erro: {e}", file=sys.stderr)
        sys.exit(1)

    import ingest
    import triangle as tri_mod
    import pricing as prc_mod
    import charts

    from pathlib import Path
    OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
    OUTPUT_DIR.mkdir(exist_ok=True)

    lob_nome = {"auto": "Auto", "liability": "Responsabilidade Civil",
                "property": "Patrimonial"}[args.lob]

    print(f"\n{'━'*55}")
    print(f"  Motor Atuarial — {lob_nome}")
    print(f"  Método: {args.method.replace('_',' ').title()}  |  "
          f"Trend: {args.trend:.1f}%  |  Loading: {args.loading:.0f}%")
    print(f"{'━'*55}")

    # 1. Dados
    print("\n[1/4] Carregando triângulo...")
    dados = ingest.carregar(args.lob)
    ingest.resumo(dados)

    # 2. IBNR
    print("[2/4] Projetando IBNR...")
    if args.method == "chain_ladder":
        resultado = tri_mod.chain_ladder(dados["triangle"], tail_factor=args.tail)
    else:
        resultado = tri_mod.bornhuetter_ferguson(
            dados["triangle"], dados["premio"],
            elr=args.elr, tail_factor=args.tail,
        )
    resultado.lob = args.lob

    # 3. Precificação
    print("[3/4] Calculando Burning Cost e Prêmio Puro...")
    preco = prc_mod.calcular_premio(
        dados["triangle"], dados["exposicao"], dados["premio"],
        resultado.ultimates,
        trend_anual=args.trend / 100,
        anos_base=args.anos_bc,
        loading_factor=args.loading / 100,
        lob=args.lob,
    )

    # 4. Painel
    _imprimir_painel(resultado, preco)

    # Exportar CSV
    csv_path = OUTPUT_DIR / f"{args.lob}_{args.method}_resultados.csv"
    prc_mod.exportar_csv(preco, resultado.ibnr, csv_path)
    print(f"[OK] CSV exportado: {csv_path.name}")

    # Gráficos
    if not args.no_charts:
        print("[4/4] Gerando gráficos...")
        figs = {
            "triangulo":   charts.heatmap_triangulo(dados["triangle"], lob_nome),
            "fatores":     charts.grafico_fatores(resultado.link_ratios, lob_nome),
            "ibnr":        charts.grafico_ibnr(resultado.ibnr, resultado.ultimates,
                                               resultado.pagos, lob_nome),
            "burning_cost": charts.grafico_burning_cost(preco.burning_cost, lob_nome),
        }
        charts.salvar_todos(figs, prefixo=f"{args.lob}_{args.method}")
        print(f"[OK] Graficos salvos em output/")
    else:
        print("[4/4] Graficos suprimidos (--no-charts).")


def _imprimir_painel(resultado, preco) -> None:
    _SEP = "=" * 55
    print(f"\n{_SEP}")
    print(f"  RESULTADO ATUARIAL — {resultado.metodo.upper()}")
    print(_SEP)
    print(f"  {'Ano':<6}  {'Pago':>12}  {'Ultimate':>12}  {'IBNR':>12}  {'%Dev':>6}")
    print(f"  {'-'*6}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*6}")
    for ano in resultado.pagos.index:
        pago  = resultado.pagos[ano]
        ult   = resultado.ultimates[ano]
        ibnr  = resultado.ibnr[ano]
        pdev  = resultado.pct_desenvolvido[ano]
        print(f"  {ano:<6}  ${pago:>11,.0f}  ${ult:>11,.0f}  ${ibnr:>11,.0f}  {pdev:>5.1%}")
    print(_SEP)
    print(f"  IBNR Total         ${resultado.ibnr_total:>12,.0f}")
    print(f"  Tail Factor        {resultado.tail_factor:>12.4f}")
    print(_SEP)
    print(f"\n  PRECIFICACAO")
    print(f"  {'─'*53}")
    print(f"  Trend Factor       {preco.trend_factor:>12.4f}")
    print(f"  Burning Cost medio ${preco.burning_cost.mean():>12,.0f}")
    print(f"  Premio Puro        ${preco.premio_puro:>12,.0f}")
    print(f"  Loading            {preco.loading_factor:>12.1%}")
    print(f"  Premio Comercial   ${preco.premio_comercial:>12,.0f}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
