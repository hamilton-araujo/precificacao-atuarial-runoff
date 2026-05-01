# TODO — Precificação Atuarial e Triângulos de Run-off

## Fase 1 — Dados e Triângulo
- [ ] Criar `src/ingest.py` — gerar triângulo cumulativo sintético com exposição e prêmio por LoB
- [ ] Suportar múltiplas Linhas de Negócio (Auto, Liability, Property)
- [ ] Calcular prêmios ganhos pro-rata temporis (regra dos 24 avos)

## Fase 2 — Motor Atuarial
- [ ] Criar `src/triangle.py` — Chain Ladder com fatores ponderados por volume
- [ ] Implementar Tail Factor via curva exponencial
- [ ] Implementar Bornhuetter-Ferguson ancorado em ELR configurável
- [ ] Calcular IBNR = Ultimate - Paid para cada ano de ocorrência

## Fase 3 — Precificação
- [ ] Criar `src/pricing.py` — Burning Cost com indexação inflacionária (Trend Factors)
- [ ] Separar frequência (avisos/exposição) e severidade (custo/aviso)
- [ ] Derivar prêmio puro final por LoB

## Fase 4 — Visualizações
- [ ] Criar `src/charts.py` — heatmap interativo do triângulo (Plotly)
- [ ] Gráfico de fatores de desenvolvimento por ano
- [ ] Gráfico de IBNR acumulado por LoB
- [ ] Gráfico de Burning Cost histórico

## Fase 5 — Dashboard Streamlit
- [ ] Criar `src/app.py` — sidebar com filtros LoB e método
- [ ] @st.cache_data para evitar reprocessamento
- [ ] Painel: triângulo + fatores + IBNR + prêmio puro

## Fase 6 — CLI
- [ ] Criar `src/main.py` com argparse
- [ ] Parâmetros: `--lob`, `--method`, `--elr`, `--trend`, `--tail`
- [ ] Exportar resultados em CSV

## Fase 7 — Testes
- [ ] Testar Chain Ladder: fatores devem ser >= 1.0
- [ ] Testar BF: IBNR_BF < IBNR_CL para anos maduros
- [ ] Testar Tail Factor: convergência para 1.0 com mais colunas
- [ ] Testar Burning Cost: prêmio > 0 para qualquer LoB

## Fase 8 — GitHub
- [ ] Criar `.gitignore`
- [ ] `git init` e primeiro commit
- [ ] Criar repositório público e fazer push
