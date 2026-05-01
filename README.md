# Atuarial End-to-End — Run-Off + Solvência II + Resseguro XoL

> **A pergunta do Conselho:** *Qual IBNR provisionar? Qual prêmio para 2025? Precisamos de mais resseguro? Sob Solvência II, o capital próprio cobre o pior cenário 99.5%?*

Motor atuarial integrado que responde **as quatro perguntas regulatórias** do Conselho de uma seguradora — não com triângulos isolados, com decisão executiva fundamentada.

---

## Por que existe

Atuários frequentemente entregam três artefatos desconectados: triângulo CL, planilha de prêmios, slide de resseguro. O Conselho precisa de **uma decisão integrada**:

| Questão | Sinal técnico |
|---|---|
| Quanto provisionar de IBNR? | Mack Stochastic CL → CV e quantis |
| Estamos solventes 99.5%? | SCR Ratio = Capital / SCR_total |
| Qual prêmio cobrar em 2025? | Burning Cost × Trend × Loading |
| Comprar mais resseguro XoL? | Monte Carlo da camada (priority/limit) |

Este projeto consolida tudo num pipeline standalone que entrega **markdown executivo + 7 charts + decisão final** com um único comando.

---

## A história em três atos

### Ato 1 — A reunião marcada
Quinta-feira, 19h. O Conselho da seguradora se reúne na próxima terça. Pauta: ajustar o balanço FY2024 e definir tarifa 2025. Você roda:

```bash
python -m src.exec_report --lob liability --capital-proprio 800000000
```

### Ato 2 — A decisão integrada
30 segundos depois, o motor entrega `output/relatorio_conselho.md`:

```
Best Estimate IBNR        R$ 712.95 M
Mack MSEP (raiz)          R$  61.47 M  (CV = 9.8%)
Quantil 99.5 (Solvência)  R$ 800.63 M
Prêmio Comercial 2025     R$   X.XX M
SCR Ratio                 3.99x  (SOLVENTE_FORTE)
Resseguro XoL prêmio      R$   4.82 M  (RoL 32.13%)

✅ DECISÃO: APROVAR_BEST_ESTIMATE
   · SCR Ratio = 3.99x ≥ 2.0 (capital robusto)
   · CV do IBNR = 9.8% ≤ 15% (provisão estável)
```

### Ato 3 — A defesa regulatória
Susep audita. *"Como vocês chegaram nesse IBNR?"* Você abre `mack_msep_charts.png` e mostra: **Mack 1993** com error bars ±1σ por ano de ocorrência, CV total 9.8%, quantil 99.5% via aproximação log-normal. Tudo documentado, reprodutível, defensável.

---

## Modelos

### 1. Chain Ladder Determinístico
Link-ratios ponderados por volume + Tail Factor estimado por regressão exponencial:
```
f_k = Σ C_{i,k+1} / Σ C_{i,k}
Ultimate_i = C_{i,J} × Π_k f_k × tail
```

### 2. Bornhuetter-Ferguson
Ancora anos imaturos na taxa de perda esperada (ELR), reduzindo volatilidade:
```
Ultimate_BF = Pago + (1 − 1/CDF) × ELR × Prêmio
```

### 3. Mack Stochastic CL (1993)
Sem assumir distribuição paramétrica, deriva analiticamente o erro de previsão:
```
σ²_k = (1/(I−k−1)) Σ_i C_{i,k} · (C_{i,k+1}/C_{i,k} − f_k)²
MSEP(C_{i,J}) = C²_{i,J} · Σ_k [σ²_k/f²_k · (1/C_{i,k} + 1/Σ C_{i,k})]
```

Quantis 75% e 99.5% via aproximação log-normal sobre o CV total — diretamente alimenta SCR de reservas.

### 4. Solvência II / Susep
```
SCR_reservas = q99.5(IBNR) − E[IBNR]                       (via Mack log-normal)
SCR_premio   = σ_premium × Volume × 3.0                    (USP padrão)
SCR_total    = √(SCR_res² + SCR_pre² + 2·0.5·SCR_res·SCR_pre)  (correlação Susep)
Risk Margin  = CoC(6%) × Σ_t SCR(t)/(1+r)^t                (Cost-of-Capital)
SCR Ratio    = Capital_Proprio / SCR_total
```

| Classificação | SCR Ratio |
|---|---|
| SOLVENTE_FORTE | ≥ 2.0x |
| SOLVENTE | 1.0–2.0x |
| SOB_INTERVENÇÃO | < 1.0x |

### 5. Resseguro XoL — Monte Carlo
Para precificar tratado de Excess of Loss `(priority, limit)`:
```
Frequência ~ Poisson(λ)
Severidade ~ LogNormal(μ, σ)
Custo XoL  = Σ_i min(max(X_i − priority, 0), limit)
Prêmio Puro = E[Custo XoL]    (50.000 simulações)
```

---

## Decisão integrada

| Decisão | Critério |
|---|---|
| ✅ APROVAR Best Estimate | SCR ≥ 2.0x **E** CV ≤ 15% |
| ⚠️ Provisionar no Q75 | CV > 30% (provisão volátil) |
| ❌ Intervenção | SCR Ratio < 1.0x |

---

## Stack

| Camada | Tecnologia |
|---|---|
| Triângulos | NumPy · Pandas · SQL Window Functions |
| Atuarial | Chain Ladder · BF · Mack 1993 · Tail exponencial |
| Solvência | Susep correlation · CoC 6% · USP padrão |
| Resseguro | Monte Carlo Poisson + LogNormal · 50k anos |
| Visualização | Plotly (heatmap interativo) · matplotlib (charts estáticos) |
| Dashboard | Streamlit |

---

## Como rodar

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Relatório executivo do Conselho (recomendado)
python -m src.exec_report --lob liability --capital-proprio 800000000

# Motor atuarial via CLI tradicional
python -m src.main --lob auto --method chain_ladder
python -m src.main --lob liability --method bf --elr 0.72

# Dashboard interativo
streamlit run src/app.py
```

---

## Outputs

```
output/
├── relatorio_conselho.md            # ⭐ Briefing Conselho com decisão
├── mack_msep_charts.png             # ⭐ BE ± Mack MSEP + CV por ano
├── solvencia_scr.png                # ⭐ Decomposição SCR + Capital
├── resseguro_xol_dist.png           # ⭐ Distribuição custo XoL Monte Carlo
├── liability_chain_ladder_*.html    # Heatmap Plotly do triângulo
├── liability_chain_ladder_*.csv     # Resultados Chain Ladder
└── ... (BF + Property + Auto)
```

⭐ = adicionado nesta versão (`exec_report`).

---

## Estrutura

```
├── src/
│   ├── exec_report.py        # ⭐ Pipeline standalone para o Conselho
│   ├── mack_stochastic.py    # ⭐ Mack 1993 CL com MSEP
│   ├── solvency.py           # ⭐ SCR Susep / Solvência II
│   ├── reinsurance.py        # ⭐ XoL Monte Carlo
│   ├── triangle.py           # Chain Ladder + BF + Tail Factor
│   ├── pricing.py            # Burning Cost + Trend + Loading
│   ├── ingest.py             # Triângulos por LoB (auto/liability/property)
│   ├── app.py                # Dashboard Streamlit
│   ├── charts.py             # Plotly + matplotlib
│   └── main.py               # CLI tradicional
├── data/
├── output/
├── tests/
├── requirements.txt
└── README.md
```

⭐ = adicionado nesta versão.
