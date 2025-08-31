# README — Projetos de Regressão e Classificação Facial (TC1 e TC2)

Este repositório contém dois scripts principais desenvolvidos para as disciplinas do PPGETI 2025.1, cobrindo atividades de **regressão** (Trabalho Computacional 1 — TC1) e **classificação facial com PCA** (Trabalho Computacional 2 — TC2).

---

## 1) regression_model.py — Benchmark de Regressão

### Visão geral
Script completo para benchmark em regressão no conjunto **Real estate valuation data set**, incluindo:
- Análise exploratória (EDA).
- Cenários **Baseline** e **Sem outliers no treino** (remoção via IQR).
- Conjunto de modelos clássicos (Regressão Linear, SVR, LSSVR, MLP etc.), além de implementações manuais (MLP 1H/2H e PL por GD).
- Comparação com métricas de artigo (RM_Art, NN_Art).
- Validação cruzada (K-fold) para os melhores modelos.
- Geração de planilhas Excel com métricas e gráficos de diagnóstico.

### Execução
```bash
python regression_model.py   --data "/caminho/Real estate valuation data set.xlsx"   --outdir "./outputs_tc1"   --test_size 0.30   --seed 42   --iqr_k 1.5
```

### Saídas
- **Excel**: `regression_results_TC1.xlsx` (métricas, CV, predições, resíduos, campeões).
- **Gráficos**: boxplots, histogramas, correlação, resíduos, QQ, dispersão predito vs real.
- **Console**: resumo dos melhores modelos por RMSE e campeões por critério composto.

### Observações
- Inclui transformações em X (padronização, Yeo-Johnson) e y (via `TransformedTargetRegressor`).
- Suporte opcional a **XGBoost** e **LightGBM**.
- Implementações manuais em NumPy: `ManualMLPRegressor1H/2H` e `PLRegressorGD`.

---

## 2) face_preprocessing_column.py — Classificação Facial com PCA (TC2)

### Visão geral
Script que cobre as 8 atividades do **Trabalho Computacional 2 (TC2)**, incluindo:
- Pré-processamento de imagens de rosto (cinza, redimensionamento, vetor coluna).
- PCA (rotação completa, seleção de componentes para variância-alvo).
- Normalizações (z-score, Box-Cox + z-score).
- Classificadores: MQ, PL, k-NN, LDA, QDA, SVM, MLP (sklearn), MLP manual, QDA manual.
- **Atividade 8 (controle de acesso)**:
  - Binária (autorizado vs intruso).
  - Open-set (somente autorizados, decisão por limiar em `p_max`).

### Estrutura de pastas
```
faces/         # imagens dos sujeitos (autorizados)
intruder/      # (opcional) imagens de intrusos
out_faces_tc2/ # saídas organizadas por atividade/classificador
```

### Execução
```bash
python face_preprocessing_column.py
```
> Parâmetros como `SIZE`, `RUNS`, `TRAIN_RATIO`, `VAR_TGT` e normalizações devem ser ajustados no cabeçalho do script.

### Saídas
- **CSV**: resultados de grid, melhores classificadores, sumários, métricas da A8.
- **PNG**: variância explicada PCA, matrizes de confusão, curvas de acurácia/perda (MLP).
- **Console**: logs de cada repetição, melhor desempenho por atividade.

### Observações
- Atividade 8 suporta dois modos de limiar: quantil (`thr_q`) e k-desvios (`thr_kstd`).
- Se a pasta `intruder/` não existir, A8 é ignorada.
- Resultados médios e desvios são salvos em `controle_acesso_summary.csv`.

---

## Requisitos Comuns
- Python 3.10+
- Bibliotecas: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `scipy`, `Pillow`, `tqdm`
- O script de regressão pode usar `xgboost` e `lightgbm` (opcionais).

---

## Créditos
Scripts preparados para as disciplinas do **PPGETI 2025.1 — UFC**.

- **TC1 (regressão)**: baseado no conjunto *Real estate valuation data set* (UCI).
- **TC2 (faces)**: baseado no conjunto *ORL Faces* (adaptado).

Autor: Polycarpo Neto
