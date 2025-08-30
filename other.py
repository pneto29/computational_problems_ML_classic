# -*- coding: utf-8 -*-
"""
TC1 — Runner local (lê dados na MESMA PASTA do script) + Comparação com o Artigo

O script procura automaticamente um arquivo Excel/CSV na mesma pasta
com as colunas do dataset de "Real estate valuation (UCI)".

Saídas em ./outputs_tc1_local/
"""

import os
import sys
import math
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, jarque_bera

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# === Valores do ARTIGO (fixos; Tabelas 6 e 7) ===
ARTICLE_TABLE6 = pd.DataFrame({
    "Model": ["MLR", "MLP (1 hidden layer)", "MLP (2 hidden layers)", "SVR", "LSSVR"],
    "RMSE_mean": [7.066, 6.607, 6.482, 6.769, 6.461],
    "RMSE_std":  [0.326, 0.288, 0.289, 0.313, 0.288],
    "MAE_mean":  [5.302, 4.873, 4.744, 4.987, 4.740],
    "MAE_std":   [0.248, 0.222, 0.220, 0.230, 0.218],
    "R2_mean":   [0.631, 0.670, 0.678, 0.653, 0.680],
    "R2_std":    [0.022, 0.020, 0.019, 0.021, 0.019],
})
ARTICLE_TABLE7 = pd.DataFrame({
    "Model": ["MLR", "MLP (1 hidden layer)", "MLP (2 hidden layers)", "SVR", "LSSVR"],
    "RMSE_Rank": [4.92, 3.28, 2.50, 3.84, 2.46],
    "MAE_Rank":  [4.92, 3.26, 2.48, 3.84, 2.50],
    "R2_Rank":   [4.92, 3.26, 2.48, 3.84, 2.50],
    "Avg_Rank":  [4.92, 3.27, 2.49, 3.84, 2.49]
})

DATA_COLS_MAP = {
    'X1 transaction date': 'trans_date',
    'X2 house age': 'house_age',
    'X3 distance to the nearest MRT station': 'dist_mrt',
    'X4 number of convenience stores': 'n_stores',
    'X5 latitude': 'lat',
    'X6 longitude': 'lon',
    'Y house price of unit area': 'price_unit'
}

FEATURES = ['trans_date','house_age','dist_mrt','n_stores','lat','lon']

def find_local_dataset(folder: Path) -> Path:
    candidates = []
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in {".xlsx", ".xls", ".csv"}:
            candidates.append(p)
    # Heurística: preferir nomes que contenham "real" e "estate"
    candidates.sort(key=lambda p: (("real" in p.name.lower()) + ("estate" in p.name.lower())), reverse=True)
    if not candidates:
        raise FileNotFoundError("Nenhum arquivo .xlsx/.xls/.csv encontrado na pasta do script.")
    return candidates[0]

def load_df_auto(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    return df

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Renomeia colunas conhecidas; tenta achar a coluna alvo se vier com outra grafia
    df2 = df.rename(columns=DATA_COLS_MAP)
    if 'price_unit' not in df2.columns:
        cands = [c for c in df2.columns if 'price' in c.lower() and 'unit' in c.lower()]
        if cands:
            df2 = df2.rename(columns={cands[0]: 'price_unit'})
    return df2

def validate_columns(df: pd.DataFrame):
    missing = [c for c in FEATURES + ['price_unit'] if c not in df.columns]
    if missing:
        msg = "Faltam colunas esperadas: " + ", ".join(missing)
        raise ValueError(msg)

def corr(a, b) -> float:
    try:
        r, _ = pearsonr(a, b); return float(r)
    except Exception:
        return float("nan")

def compute_ranks(df_summary: pd.DataFrame) -> pd.DataFrame:
    # aceita 'model' ou 'Model'
    model_col = "model" if "model" in df_summary.columns else ("Model" if "Model" in df_summary.columns else None)
    if not model_col:
        raise ValueError("Coluna de modelo não encontrada (esperava 'model' ou 'Model').")
    df = df_summary.copy()
    if "rmse_mean" not in df.columns or "mae_mean" not in df.columns or "r2_mean" not in df.columns:
        raise ValueError("Colunas de média não encontradas em df_summary.")
    df["RMSE_Rank"] = df["rmse_mean"].rank(method="average", ascending=True)
    df["MAE_Rank"]  = df["mae_mean"].rank(method="average", ascending=True)
    df["R2_Rank"]   = df["r2_mean"].rank(method="average", ascending=False)
    df["Avg_Rank"]  = df[["RMSE_Rank","MAE_Rank","R2_Rank"]].mean(axis=1)
    return df[[model_col,"RMSE_Rank","MAE_Rank","R2_Rank","Avg_Rank"]].sort_values("Avg_Rank")

def main():
    here = Path(__file__).resolve().parent
    out_dir = here / "outputs_tc1_local"
    out_dir.mkdir(exist_ok=True)

    data_path = find_local_dataset(here)
    print(f"[INFO] Lendo dados de: {data_path.name}")
    df_raw = load_df_auto(data_path)
    print(f"[INFO] Colunas originais: {list(df_raw.columns)}")

    df = normalize_columns(df_raw)
    print(f"[INFO] Colunas após normalização: {list(df.columns)}")
    validate_columns(df)

    X = df[FEATURES].copy().dropna()
    y = df.loc[X.index, 'price_unit'].copy()

    # Modelos
    models = {
        "MLR": Pipeline([("linreg", LinearRegression())]),
        "MLP_1h": Pipeline([("scaler", StandardScaler()), ("mlp", MLPRegressor(hidden_layer_sizes=(32,), activation="tanh", solver="adam", learning_rate="adaptive", max_iter=2000, early_stopping=True, n_iter_no_change=25, random_state=0))]),
        "MLP_2h": Pipeline([("scaler", StandardScaler()), ("mlp", MLPRegressor(hidden_layer_sizes=(64,32), activation="tanh", solver="adam", learning_rate="adaptive", max_iter=2000, early_stopping=True, n_iter_no_change=25, random_state=0))]),
        "SVR": Pipeline([("scaler", StandardScaler()), ("svr", SVR(kernel="rbf", C=10.0, gamma="scale", epsilon=0.2))]),
        "LSSVR": Pipeline([("scaler", StandardScaler()), ("krr", KernelRidge(kernel="rbf", alpha=1.0, gamma=None))]),
    }

    # Rodada única (gráficos + métricas)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=2025)
    single_rows = []
    for name, mdl in models.items():
        mdl.fit(X_tr.values, y_tr.values)
        y_tr_hat = mdl.predict(X_tr.values)
        y_te_hat = mdl.predict(X_te.values)
        resid = y_tr.values - y_tr_hat
        jb_stat, jb_p = jarque_bera(resid)
        single_rows.append({
            "model": name,
            "corr_train": corr(y_tr.values, y_tr_hat),
            "corr_test": corr(y_te.values, y_te_hat),
            "rmse_train": math.sqrt(mean_squared_error(y_tr.values, y_tr_hat)),
            "rmse_test": math.sqrt(mean_squared_error(y_te.values, y_te_hat)),
            "mae_train": mean_absolute_error(y_tr.values, y_tr_hat),
            "mae_test": mean_absolute_error(y_te.values, y_te_hat),
            "r2_train": r2_score(y_tr.values, y_tr_hat),
            "r2_test": r2_score(y_te.values, y_te_hat),
            "jb_stat": float(jb_stat), "jb_p": float(jb_p)
        })

        # Plots para a rodada única
        # 1) Histograma de resíduos
        plt.figure()
        plt.hist(resid, bins=30)
        plt.title(f"{name} — Histograma de resíduos (treino) [single]")
        plt.xlabel("Resíduo (y_true - y_pred)"); plt.ylabel("Frequência")
        plt.tight_layout()
        plt.savefig(out_dir / f"single_{name}_hist_residuos_treino.png", dpi=150)
        plt.close()

        # 2) Scatter treino
        plt.figure()
        plt.scatter(y_tr.values, y_tr_hat, s=12)
        lims = [min(y_tr.min(), y_tr_hat.min()), max(y_tr.max(), y_tr_hat.max())]
        plt.plot(lims, lims)
        plt.title(f"{name} — Dispersão (treino) [single]")
        plt.xlabel("y medido (treino)"); plt.ylabel("y predito (treino)")
        plt.tight_layout()
        plt.savefig(out_dir / f"single_{name}_scatter_treino.png", dpi=150)
        plt.close()

        # 3) Scatter teste
        plt.figure()
        plt.scatter(y_te.values, y_te_hat, s=12)
        lims = [min(y_te.min(), y_te_hat.min()), max(y_te.max(), y_te_hat.max())]
        plt.plot(lims, lims)
        plt.title(f"{name} — Dispersão (teste) [single]")
        plt.xlabel("y medido (teste)"); plt.ylabel("y predito (teste)")
        plt.tight_layout()
        plt.savefig(out_dir / f"single_{name}_scatter_teste.png", dpi=150)
        plt.close()

    df_single = pd.DataFrame(single_rows)
    df_single.to_csv(out_dir / "metrics_single_run.csv", index=False)

    # 50 repetições
    rows = []
    rng = np.random.RandomState(42)
    for r in range(50):
        rs = int(rng.randint(0, 1_000_000))
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=rs)
        for name, mdl in models.items():
            mdl.fit(Xtr.values, ytr.values)
            ypred = mdl.predict(Xte.values)
            rows.append({
                "repeat": r+1, "model": name,
                "rmse": math.sqrt(mean_squared_error(yte.values, ypred)),
                "mae": mean_absolute_error(yte.values, ypred),
                "r2": r2_score(yte.values, ypred)
            })
    df_rep = pd.DataFrame(rows)
    df_rep.to_csv(out_dir / "metrics_repeats_raw_50x.csv", index=False)

    summary = df_rep.groupby("model").agg(
        rmse_mean=("rmse","mean"), rmse_std=("rmse","std"),
        mae_mean=("mae","mean"), mae_std=("mae","std"),
        r2_mean=("r2","mean"),   r2_std=("r2","std")
    ).reset_index()
    summary.to_csv(out_dir / "metrics_repeats_summary_50x.csv", index=False)

    # Comparação com Artigo — Tabela 6
    name_map = {"MLR":"MLR","MLP_1h":"MLP (1 hidden layer)","MLP_2h":"MLP (2 hidden layers)","SVR":"SVR","LSSVR":"LSSVR"}
    sum_named = summary.copy(); sum_named["Model"] = sum_named["model"].map(name_map)
    comp6 = pd.merge(ARTICLE_TABLE6, sum_named, on="Model", how="left")
    comp6 = comp6[[
        "Model",
        "RMSE_mean","RMSE_std","rmse_mean","rmse_std",
        "MAE_mean","MAE_std","mae_mean","mae_std",
        "R2_mean","R2_std","r2_mean","r2_std"
    ]]
    comp6.to_csv(out_dir / "compare_table6_article_vs_mine.csv", index=False)

    # Comparação com Artigo — Tabela 7 (ranks)
    my_ranks = compute_ranks(summary)
    my_ranks = my_ranks.rename(columns={"model":"Model"})
    # ajustar nomes
    my_ranks["Model"] = my_ranks["Model"].map(name_map)
    comp7 = pd.merge(ARTICLE_TABLE7, my_ranks, on="Model", how="left", suffixes=("_Artigo","_Meu"))
    comp7.to_csv(out_dir / "compare_table7_article_vs_mine.csv", index=False)

    # HTML simples consolidando
    html = []
    def add_table(title, dfp):
        html.append(f"<h2>{title}</h2>")
        html.append(dfp.to_html(index=False))

    html.append("<h1>TC1 — Resultados locais & Comparação com Artigo</h1>")
    html.append("<p><b>Nota:</b> Valores do artigo são de referência e estão fixos no código.</p>")
    add_table("Métricas (rodada única)", df_single)
    add_table("Resumo 50 repetições (média±std)", summary)
    add_table("Comparação Tabela 6 (Artigo vs Meu)", comp6)
    add_table("Comparação Tabela 7 (Artigo vs Meu)", comp7)
    with open(out_dir / "report_tc1_local.html", "w", encoding="utf-8") as f:
        f.write("\n".join(html))

    print("[OK] Concluído.")
    print("Saídas em:", out_dir)
    for fn in sorted(os.listdir(out_dir)):
        print("-", fn)


if __name__ == "__main__":
    main()