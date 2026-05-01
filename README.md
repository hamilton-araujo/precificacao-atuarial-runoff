# Precificação Atuarial e Triângulos de Run-off Dinâmicos

Motor atuarial end-to-end para precificação de seguros de responsabilidade civil e projeção de provisões técnicas (IBNR), encapsulado num dashboard Streamlit interativo.

## Dataset

**Insurance Claims Dataset** (Kaggle) — dados sintéticos de apólices e sinistros com exposição, prêmio, pagamentos incrementais e reservas por ano de ocorrência e desenvolvimento.

## Stack

| Camada | Tecnologia |
|---|---|
| Dados | Pandas · SQL Window Functions · NumPy |
| Atuarial | Chain Ladder · Bornhuetter-Ferguson · Tail Factor |
| Precificação | Burning Cost · Frequência × Severidade · Trend Factors |
| Dashboard | Streamlit · Plotly (heatmap interativo) |

## Como rodar

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Dashboard interativo
streamlit run src/app.py

# Motor atuarial via CLI
python src/main.py --lob auto --method chain_ladder
python src/main.py --lob liability --method bf --elr 0.65
```

## Métodos Implementados

### Chain Ladder (Determinístico)
Calcula fatores age-to-age (link ratios) com média ponderada por volume:
```
f_k = Σ C_{i,k+1} / Σ C_{i,k}   (para i onde desenvolvimento k+1 existe)
```

### Bornhuetter-Ferguson
Ancora a projeção dos anos imaturos na taxa de perda esperada (ELR):
```
Ultimate_BF = Paid + (1 - 1/CDF) × ELR × Premium
```

### Tail Factor
Interpola um fator de cauda via curva exponencial para capturar desenvolvimento além da última coluna observada.

### Burning Cost
```
Burning Cost = Σ Sinistros Indexados / Σ Exposição
Prêmio Puro  = Burning Cost × Trend Factor × Development Factor
```

## Dashboard Streamlit

```
┌─────────────────────────────────────────────────────┐
│  Linha de Negócio: [Auto ▼]  Método: [Chain Ladder ▼]│
├─────────────────────────────────────────────────────┤
│  Triângulo de Run-off (heatmap interativo Plotly)    │
│  Fatores de Desenvolvimento  │  Projeções IBNR       │
│  Burning Cost por Ano        │  Prêmio Puro Final    │
└─────────────────────────────────────────────────────┘
```

## Estrutura

```
├── src/
│   ├── app.py            # Dashboard Streamlit
│   ├── main.py           # CLI atuarial
│   ├── ingest.py         # Geração/carga do triângulo
│   ├── triangle.py       # Lógica Chain Ladder + BF + Tail
│   ├── pricing.py        # Burning Cost e prêmio puro
│   └── charts.py         # Heatmap Plotly e gráficos
├── data/
├── output/
├── requirements.txt
└── README.md
```
