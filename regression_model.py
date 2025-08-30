# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# TC1 – Benchmark no 'Real estate valuation data set'
# + Comparação com artigo (RM_Art, NN_Art)
# + Gráficos (linha ideal + tendência + equação + R²)
# + Campeão (critério composto)
# + CENÁRIOS: Baseline vs Remoção de Outliers (IQR em y, apenas no TREINO)
# + EDA: Boxplots/histograma (quartis/IQR), correlação, describe
#
# Execução:
#   python regression_model.py --data "/caminho/Real estate valuation data set.xlsx" --outdir "./outputs_tc1"
# """
#
# import argparse, os, warnings, re
# from typing import Dict, Tuple
# import numpy as np, pandas as pd, matplotlib.pyplot as plt
#
# from sklearn import __version__ as skl_version
# from sklearn.model_selection import train_test_split, KFold, cross_val_score
# from sklearn.preprocessing import StandardScaler, PowerTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.compose import TransformedTargetRegressor
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
# from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
# from sklearn.svm import SVR
# from sklearn.kernel_ridge import KernelRidge
# from sklearn.neural_network import MLPRegressor
# from sklearn.base import BaseEstimator, RegressorMixin
#
# # --- árvores e boosting
# from sklearn.ensemble import RandomForestRegressor
# try:
#     from xgboost import XGBRegressor
#     HAS_XGB = True
# except Exception:
#     HAS_XGB = False
# try:
#     from lightgbm import LGBMRegressor
#     HAS_LGBM = True
# except Exception:
#     HAS_LGBM = False
#
# import openpyxl  # engine Excel
#
# warnings.filterwarnings("ignore", category=UserWarning)
# warnings.filterwarnings("ignore", category=FutureWarning)
# warnings.filterwarnings("ignore", category=RuntimeWarning)
#
# # ========================== Compatibilidade sklearn ============================
# def rmse(y_true, y_pred) -> float:
#     try:
#         return mean_squared_error(y_true, y_pred, squared=False)
#     except TypeError:
#         return np.sqrt(mean_squared_error(y_true, y_pred))
#
# def robust_pearsonr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
#     yt = np.asarray(y_true).ravel()
#     yp = np.asarray(y_pred).ravel()
#     if yt.std() == 0 or yp.std() == 0:
#         return 0.0
#     return float(np.corrcoef(yt, yp)[0, 1])
#
# RMSE_SCORER = make_scorer(lambda yt, yp: -rmse(yt, yp), greater_is_better=True)
#
# # ================================ Utils =======================================
# def detect_target_column(df: pd.DataFrame) -> str:
#     exact = [c for c in df.columns if c.strip().lower() == "y house price of unit area"]
#     if exact: return exact[0]
#     candidates = [c for c in df.columns if "price" in c.lower()]
#     return candidates[0] if candidates else df.columns[-1]
#
# def make_scaled_regressor(base_estimator):
#     """Escala/transforma X e padroniza y (TransformedTargetRegressor)."""
#     pipe = Pipeline([
#         ("scaler", StandardScaler()),
#         ("pt", PowerTransformer(method="yeo-johnson")),
#         ("model", base_estimator),
#     ])
#     return TransformedTargetRegressor(
#         regressor=pipe,
#         transformer=StandardScaler(with_mean=True, with_std=True)
#     )
#
# def make_scaled_X_only(base_estimator):
#     """Escala/transforma apenas X (para estimadores que já cuidam de y)."""
#     return Pipeline([
#         ("scaler", StandardScaler()),
#         ("pt", PowerTransformer(method="yeo-johnson")),
#         ("model", base_estimator),
#     ])
#
# def hit_rate(y_true: np.ndarray, y_pred: np.ndarray, pct: float) -> float:
#     yt = np.asarray(y_true).ravel()
#     yp = np.asarray(y_pred).ravel()
#     rel = np.abs(yp - yt) / np.maximum(np.abs(yt), 1e-8)
#     return 100.0 * float(np.mean(rel <= pct))
#
# # -------- Outliers (IQR em y, apenas no treino) --------
# def iqr_mask(y: np.ndarray, k: float = 1.5) -> np.ndarray:
#     q1, q3 = np.percentile(y, 25), np.percentile(y, 75)
#     iqr = q3 - q1
#     lo, hi = q1 - k*iqr, q3 + k*iqr
#     return (y >= lo) & (y <= hi)
#
# # ============================== MLPs MANUAIS ==================================
# def _tanh(z): return np.tanh(z)
# def _dtanh(a): return 1.0 - a*a  # a = tanh(z)
# def _xavier_limit(fan_in, fan_out): return np.sqrt(6.0 / (fan_in + fan_out))
#
# def _init_wb(rng, fan_in, fan_out):
#     lim = _xavier_limit(fan_in, fan_out)
#     W = rng.uniform(-lim, lim, size=(fan_in, fan_out))
#     b = np.zeros((fan_out,), dtype=float)
#     return W, b
#
# def _clip_inplace(arrs, max_norm: float):
#     if max_norm is None or max_norm <= 0: return
#     total = np.sqrt(sum(float(np.sum(a*a)) for a in arrs))
#     if total > max_norm:
#         scale = max_norm / (total + 1e-12)
#         for a in arrs: a *= scale
#
# class ManualMLPRegressor1H(BaseEstimator, RegressorMixin):
#     def __init__(self,
#                  hidden_size=18, lr=0.1, momentum=0.9, epochs=200, batch_size=32,
#                  weight_decay=1e-4, early_stopping=True, patience=20,
#                  lr_decay=True, lr_decay_factor=0.5, lr_patience=8,
#                  clip_grad_norm=5.0, random_state=42, verbose=False):
#         self.hidden_size = hidden_size
#         self.lr = lr
#         self.momentum = momentum
#         self.epochs = epochs
#         self.batch_size = batch_size
#         self.weight_decay = weight_decay
#         self.early_stopping = early_stopping
#         self.patience = patience
#         self.lr_decay = lr_decay
#         self.lr_decay_factor = lr_decay_factor
#         self.lr_patience = lr_patience
#         self.clip_grad_norm = clip_grad_norm
#         self.random_state = random_state
#         self.verbose = verbose
#
#     def fit(self, X, y):
#         rng = np.random.RandomState(self.random_state)
#         X = np.asarray(X, dtype=float)
#         y = np.asarray(y, dtype=float).reshape(-1, 1)
#
#         n, d = X.shape; H = self.hidden_size
#         W1, b1 = _init_wb(rng, d, H)
#         W2, b2 = _init_wb(rng, H, 1)
#         vW1 = np.zeros_like(W1); vb1 = np.zeros_like(b1)
#         vW2 = np.zeros_like(W2); vb2 = np.zeros_like(b2)
#
#         best_val = np.inf; best_params = (W1.copy(), b1.copy(), W2.copy(), b2.copy())
#         wait_es = 0; wait_lr = 0; lr = float(self.lr)
#
#         if self.early_stopping:
#             idx = rng.permutation(n); vsz = max(1, int(0.1*n))
#             val_idx = idx[:vsz]; tr_idx = idx[vsz:]
#             Xtr, ytr = X[tr_idx], y[tr_idx]; Xval, yval = X[val_idx], y[val_idx]
#         else:
#             Xtr, ytr = X, y; Xval, yval = None, None
#
#         for ep in range(self.epochs):
#             order = rng.permutation(len(Xtr))
#             for start in range(0, len(Xtr), self.batch_size):
#                 sl = order[start:start+self.batch_size]; xb = Xtr[sl]; yb = ytr[sl]
#                 a1 = _tanh(xb @ W1 + b1)
#                 yhat = a1 @ W2 + b2
#                 diff = (yhat - yb); grad_yhat = (2.0 / max(1, len(xb))) * diff
#                 dW2 = a1.T @ grad_yhat + self.weight_decay * W2
#                 db2 = grad_yhat.sum(axis=0)
#                 da1 = grad_yhat @ W2.T; dz1 = da1 * _dtanh(a1)
#                 dW1 = xb.T @ dz1 + self.weight_decay * W1
#                 db1 = dz1.sum(axis=0)
#                 _clip_inplace([dW1, db1, dW2, db2], self.clip_grad_norm)
#                 vW2 = self.momentum * vW2 - lr * dW2; vb2 = self.momentum * vb2 - lr * db2
#                 vW1 = self.momentum * vW1 - lr * dW1; vb1 = self.momentum * vb1 - lr * db1
#                 W2 += vW2; b2 += vb2; W1 += vW1; b1 += vb1
#
#             if self.early_stopping:
#                 a1v = _tanh(Xval @ W1 + b1); yhat_val = a1v @ W2 + b2
#                 val_rmse = rmse(yval, yhat_val)
#                 if val_rmse + 1e-12 < best_val:
#                     best_val = val_rmse; best_params = (W1.copy(), b1.copy(), W2.copy(), b2.copy())
#                     wait_es = 0; wait_lr = 0
#                 else:
#                     wait_es += 1; wait_lr += 1
#                 if self.lr_decay and wait_lr >= self.lr_patience:
#                     lr *= self.lr_decay_factor; wait_lr = 0
#                 if self.verbose:
#                     print(f"[1H][epoch {ep+1}] val_RMSE={val_rmse:.4f} best={best_val:.4f} lr={lr:.5f} wait={wait_es}")
#                 if wait_es >= self.patience: break
#
#         if self.early_stopping:
#             W1, b1, W2, b2 = best_params
#         self.W1_, self.b1_, self.W2_, self.b2_ = W1, b1, W2, b2
#         return self
#
#     def predict(self, X):
#         X = np.asarray(X, dtype=float)
#         a1 = _tanh(X @ self.W1_ + self.b1_)
#         yhat = a1 @ self.W2_ + self.b2_
#         return yhat.ravel()
#
# class ManualMLPRegressor2H(BaseEstimator, RegressorMixin):
#     def __init__(self,
#                  hidden_size1=18, hidden_size2=18, lr=0.1, momentum=0.9, epochs=250, batch_size=32,
#                  weight_decay=1e-4, early_stopping=True, patience=25,
#                  lr_decay=True, lr_decay_factor=0.5, lr_patience=8,
#                  clip_grad_norm=5.0, random_state=42, verbose=False):
#         self.hidden_size1 = hidden_size1; self.hidden_size2 = hidden_size2
#         self.lr = lr; self.momentum = momentum; self.epochs = epochs; self.batch_size = batch_size
#         self.weight_decay = weight_decay; self.early_stopping = early_stopping; self.patience = patience
#         self.lr_decay = lr_decay; self.lr_decay_factor = lr_decay_factor; self.lr_patience = lr_patience
#         self.clip_grad_norm = clip_grad_norm; self.random_state = random_state; self.verbose = verbose
#
#     def fit(self, X, y):
#         rng = np.random.RandomState(self.random_state)
#         X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float).reshape(-1, 1)
#         n, d = X.shape; H1, H2 = self.hidden_size1, self.hidden_size2
#         W1, b1 = _init_wb(rng, d,  H1); W2, b2 = _init_wb(rng, H1, H2); W3, b3 = _init_wb(rng, H2, 1)
#         vW1 = np.zeros_like(W1); vb1 = np.zeros_like(b1)
#         vW2 = np.zeros_like(W2); vb2 = np.zeros_like(b2)
#         vW3 = np.zeros_like(W3); vb3 = np.zeros_like(b3)
#
#         best_val = np.inf; best_params = (W1.copy(), b1.copy(), W2.copy(), b2.copy(), W3.copy(), b3.copy())
#         wait_es = 0; wait_lr = 0; lr = float(self.lr)
#
#         if self.early_stopping:
#             idx = rng.permutation(n); vsz = max(1, int(0.1*n))
#             val_idx = idx[:vsz]; tr_idx = idx[vsz:]
#             Xtr, ytr = X[tr_idx], y[tr_idx]; Xval, yval = X[val_idx], y[val_idx]
#         else:
#             Xtr, ytr = X, y; Xval, yval = None, None
#
#         for ep in range(self.epochs):
#             order = rng.permutation(len(Xtr))
#             for start in range(0, len(Xtr), self.batch_size):
#                 sl = order[start:start+self.batch_size]; xb = Xtr[sl]; yb = ytr[sl]
#                 a1 = _tanh(xb @ W1 + b1); a2 = _tanh(a1 @ W2 + b2); yhat = a2 @ W3 + b3
#                 diff = (yhat - yb); grad_yhat = (2.0 / max(1, len(xb))) * diff
#                 dW3 = a2.T @ grad_yhat + self.weight_decay * W3; db3 = grad_yhat.sum(axis=0)
#                 da2 = grad_yhat @ W3.T; dz2 = da2 * _dtanh(a2)
#                 dW2 = a1.T @ dz2 + self.weight_decay * W2; db2 = dz2.sum(axis=0)
#                 da1 = dz2 @ W2.T; dz1 = da1 * _dtanh(a1)
#                 dW1 = xb.T @ dz1 + self.weight_decay * W1; db1 = dz1.sum(axis=0)
#                 _clip_inplace([dW1, db1, dW2, db2, dW3, db3], self.clip_grad_norm)
#                 vW3 = self.momentum * vW3 - lr * dW3; vb3 = self.momentum * vb3 - lr * db3
#                 vW2 = self.momentum * vW2 - lr * dW2; vb2 = self.momentum * vb2 - lr * db2
#                 vW1 = self.momentum * vW1 - lr * dW1; vb1 = self.momentum * vb1 - lr * db1
#                 W3 += vW3; b3 += vb3; W2 += vW2; b2 += vb2; W1 += vW1; b1 += vb1
#
#             if self.early_stopping:
#                 a1v = _tanh(Xval @ W1 + b1); a2v = _tanh(a1v @ W2 + b2)
#                 yhat_val = a2v @ W3 + b3; val_rmse = rmse(yval, yhat_val)
#                 if val_rmse + 1e-12 < best_val:
#                     best_val = val_rmse; best_params = (W1.copy(), b1.copy(), W2.copy(), b2.copy(), W3.copy(), b3.copy())
#                     wait_es = 0; wait_lr = 0
#                 else:
#                     wait_es += 1; wait_lr += 1
#                 if self.lr_decay and wait_lr >= self.lr_patience:
#                     lr *= self.lr_decay_factor; wait_lr = 0
#                 if self.verbose:
#                     print(f"[2H][epoch {ep+1}] val_RMSE={val_rmse:.4f} best={best_val:.4f} lr={lr:.5f} wait={wait_es}")
#                 if wait_es >= self.patience: break
#
#         if self.early_stopping:
#             W1, b1, W2, b2, W3, b3 = best_params
#         self.W1_, self.b1_, self.W2_, self.b2_, self.W3_, self.b3_ = W1, b1, W2, b2, W3, b3
#         return self
#
#     def predict(self, X):
#         X = np.asarray(X, dtype=float)
#         a1 = _tanh(X @ self.W1_ + self.b1_)
#         a2 = _tanh(a1 @ self.W2_ + self.b2_)
#         yhat = a2 @ self.W3_ + self.b3_
#         return yhat.ravel()
#
# # ========== Perceptron Logístico (Regra Delta Generalizada) ==========
# def _logsig(z): return 1.0 / (1.0 + np.exp(-z))
# def _dlogsig(y): return y * (1.0 - y)
# def _dtanh_from_y(y): return 1.0 - y*y
#
# class PLRegressorGD(BaseEstimator, RegressorMixin):
#     """Perceptron Logístico para REGRESSÃO com Regra Delta Generalizada."""
#     def __init__(self, activation="logsig", lr=0.05, epochs=500, batch_size=32,
#                  momentum=0.9, weight_decay=1e-4, random_state=42, verbose=False,
#                  eps_range=1e-3):
#         assert activation in ("logsig", "tanh")
#         self.activation = activation
#         self.lr = lr; self.epochs = epochs; self.batch_size = batch_size
#         self.momentum = momentum; self.weight_decay = weight_decay
#         self.random_state = random_state; self.verbose = verbose
#         self.eps_range = eps_range
#
#     def _fit_target_scaler(self, y):
#         y = y.ravel().astype(float)
#         self.y_min_ = float(np.min(y)); self.y_max_ = float(np.max(y))
#         rng = self.y_max_ - self.y_min_
#         self.y_rng_ = rng if rng > 0 else 1.0
#
#     def _y_to_act_space(self, y):
#         y = y.ravel().astype(float)
#         y01 = (y - self.y_min_) / self.y_rng_
#         y01 = np.clip(y01, 0.0 + self.eps_range, 1.0 - self.eps_range)
#         if self.activation == "logsig":
#             return y01
#         else:
#             return y01 * (2.0 - 2*self.eps_range) - (1.0 - self.eps_range)
#
#     def _y_from_act_space(self, y_act):
#         y_act = np.asarray(y_act, dtype=float).ravel()
#         if self.activation == "logsig":
#             y01 = np.clip(y_act, 0.0, 1.0)
#         else:
#             y01 = (np.clip(y_act, -1.0, 1.0) + 1.0) / 2.0
#         return y01 * self.y_rng_ + self.y_min_
#
#     def _forward(self, X, w, b):
#         z = X @ w + b
#         return _logsig(z) if self.activation == "logsig" else np.tanh(z)
#
#     def fit(self, X, y):
#         rng = np.random.RandomState(self.random_state)
#         X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float).ravel()
#         n, d = X.shape
#         self._fit_target_scaler(y)
#         yd = self._y_to_act_space(y).reshape(-1, 1)
#
#         w = rng.normal(0.0, 0.01, size=(d,1)); b = 0.0
#         vw = np.zeros_like(w); vb = 0.0
#
#         for ep in range(self.epochs):
#             order = rng.permutation(n)
#             for start in range(0, n, self.batch_size):
#                 sl = order[start:start+self.batch_size]
#                 xb = X[sl]; db = yd[sl]
#                 yb = self._forward(xb, w, b)
#                 e  = db - yb
#                 phi_prime = _dlogsig(yb) if self.activation == "logsig" else _dtanh_from_y(yb)
#                 delta = e * phi_prime
#                 grad_w = xb.T @ delta / max(1, len(sl)) + self.weight_decay * w
#                 grad_b = float(np.sum(delta) / max(1, len(sl)))
#                 # momentum (ascendente, já incorpora lr)
#                 vw = self.momentum * vw + self.lr * grad_w
#                 vb = self.momentum * vb + self.lr * grad_b
#                 w += vw; b += vb
#             if self.verbose and (ep+1) % 50 == 0:
#                 y_tr_pred = self._y_from_act_space(self._forward(X, w, b))
#                 print(f"[PL-{self.activation}][{ep+1}/{self.epochs}] RMSE={rmse(y, y_tr_pred):.4f}")
#
#         self.w_ = w; self.b_ = b
#         return self
#
#     def predict(self, X):
#         X = np.asarray(X, dtype=float)
#         y_act = self._forward(X, self.w_, self.b_)
#         return self._y_from_act_space(y_act).ravel()
#
# # ============================ Modelos & Avaliação =============================
# def build_models(n_features: int) -> Dict[str, object]:
#     gamma_rbf = 1.0 / max(n_features, 1)
#     cv5 = KFold(n_splits=5, shuffle=True, random_state=42)
#
#     models = {
#         # lineares / kernel / MLP (sklearn)
#         "OLS_LinearRegression": Pipeline([
#             ("scaler", StandardScaler()),
#             ("pt", PowerTransformer(method="yeo-johnson")),
#             ("model", LinearRegression())
#         ]),
#         "RidgeCV": Pipeline([
#             ("scaler", StandardScaler()),
#             ("pt", PowerTransformer(method="yeo-johnson")),
#             ("model", RidgeCV(alphas=np.logspace(-3, 3, 25), cv=cv5))
#         ]),
#         "LassoCV": Pipeline([
#             ("scaler", StandardScaler()),
#             ("pt", PowerTransformer(method="yeo-johnson")),
#             ("model", LassoCV(alphas=None, cv=cv5, max_iter=20000))
#         ]),
#         "ElasticNetCV": Pipeline([
#             ("scaler", StandardScaler()),
#             ("pt", PowerTransformer(method="yeo-johnson")),
#             ("model", ElasticNetCV(l1_ratio=[.1,.3,.5,.7,.9,.95,.99,1.0], cv=cv5, max_iter=20000))
#         ]),
#         "KernelRidge_RBF": make_scaled_regressor(KernelRidge(alpha=1.0, kernel="rbf", gamma=gamma_rbf)),
#         "SVR_RBF": make_scaled_regressor(SVR(C=10.0, epsilon=0.1, kernel="rbf", gamma="scale", max_iter=-1)),
#
#         "MLP_1H_18": make_scaled_regressor(MLPRegressor(
#             hidden_layer_sizes=(18,), activation="relu", solver="adam",
#             learning_rate_init=0.01, max_iter=5000, early_stopping=True, n_iter_no_change=25, random_state=42)),
#         "MLP_2H_18_18": make_scaled_regressor(MLPRegressor(
#             hidden_layer_sizes=(18,18), activation="relu", solver="adam",
#             learning_rate_init=0.01, max_iter=6000, early_stopping=True, n_iter_no_change=25, random_state=42)),
#
#         # MLPs manuais
#         "MLP_manual_1H": make_scaled_regressor(ManualMLPRegressor1H(
#             hidden_size=18, lr=0.1, momentum=0.9, epochs=200, batch_size=32,
#             weight_decay=1e-4, early_stopping=True, patience=20,
#             lr_decay=True, lr_decay_factor=0.5, lr_patience=8,
#             clip_grad_norm=5.0, random_state=42, verbose=False
#         )),
#         "MLP_manual_2H": make_scaled_regressor(ManualMLPRegressor2H(
#             hidden_size1=18, hidden_size2=18, lr=0.1, momentum=0.9, epochs=250, batch_size=32,
#             weight_decay=1e-4, early_stopping=True, patience=25,
#             lr_decay=True, lr_decay_factor=0.5, lr_patience=8,
#             clip_grad_norm=5.0, random_state=42, verbose=False
#         )),
#
#         # Perceptrons Logísticos (Regra Delta Generalizada) – escala apenas X
#         "PL_LogSig_LMS": make_scaled_X_only(PLRegressorGD(
#             activation="logsig", lr=0.05, epochs=500, batch_size=32,
#             momentum=0.9, weight_decay=1e-4, random_state=42, verbose=False
#         )),
#         "PL_Tanh_LMS": make_scaled_X_only(PLRegressorGD(
#             activation="tanh", lr=0.05, epochs=500, batch_size=32,
#             momentum=0.9, weight_decay=1e-4, random_state=42, verbose=False
#         )),
#     }
#
#     # ================= Árvores/Boosting (regressão) =================
#     models["RandomForest"] = RandomForestRegressor(
#         n_estimators=500,
#         max_depth=None,
#         min_samples_leaf=1,
#         n_jobs=-1,
#         random_state=42
#     )
#
#     if HAS_XGB:
#         models["XGBoost"] = XGBRegressor(
#             n_estimators=600,
#             learning_rate=0.05,
#             max_depth=4,
#             subsample=0.9,
#             colsample_bytree=0.9,
#             reg_lambda=1.0,
#             objective="reg:squarederror",
#             n_jobs=-1,
#             random_state=42
#         )
#
#     if HAS_LGBM:
#         models["LightGBM"] = LGBMRegressor(
#             n_estimators=800,
#             learning_rate=0.05,
#             num_leaves=31,
#             max_depth=-1,
#             subsample=0.9,
#             colsample_bytree=0.9,
#             reg_lambda=1.0,
#             random_state=42,
#             n_jobs=-1
#         )
#
#     return models
#
# def evaluate_models(models: Dict[str, object],
#                     X_train: pd.DataFrame, y_train: pd.Series,
#                     X_test: pd.DataFrame, y_test: pd.Series,
#                     scenario: str) -> Tuple[pd.DataFrame, Dict, Dict, Dict]:
#     records = []
#     preds_train, preds_test, residuals_train = {}, {}, {}
#     avg_price_test = float(np.mean(y_test))
#
#     for name, model in models.items():
#         model.fit(X_train, y_train)
#         yhat_tr = model.predict(X_train)
#         yhat_te = model.predict(X_test)
#
#         preds_train[name] = yhat_tr
#         preds_test[name] = yhat_te
#         residuals_train[name] = y_train.values - yhat_tr
#
#         rec = {
#             "Scenario": scenario,
#             "Model": name,
#             "RMSE_train": rmse(y_train, yhat_tr),
#             "RMSE_test":  rmse(y_test,  yhat_te),
#             "MAE_train": mean_absolute_error(y_train, yhat_tr),
#             "MAE_test":  mean_absolute_error(y_test,  yhat_te),
#             "R2_train":  r2_score(y_train, yhat_tr),
#             "R2_test":   r2_score(y_test,  yhat_te),
#             "Corr_train": robust_pearsonr(y_train, yhat_tr),
#             "Corr_test":  robust_pearsonr(y_test,  yhat_te),
#             "HitRate20_test(%)": hit_rate(y_test, yhat_te, 0.20),
#             "HitRate10_test(%)": hit_rate(y_test, yhat_te, 0.10),
#             "AvgPrice_test": avg_price_test,
#             "RMSE/AvgPrice_test": (rmse(y_test, yhat_te) / avg_price_test) if avg_price_test != 0 else np.nan,
#         }
#         records.append(rec)
#
#     metrics_df = pd.DataFrame(records).sort_values(by=["Scenario","RMSE_test"]).reset_index(drop=True)
#     return metrics_df, preds_train, preds_test, residuals_train
#
# def crossval_top4(models: Dict[str, object], metrics_df: pd.DataFrame,
#                   X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
#     top_names = metrics_df.sort_values("RMSE_test")["Model"].unique()[:4].tolist()
#     cv = KFold(n_splits=5, shuffle=True, random_state=42)
#     rows = []
#     for name in top_names:
#         est = models[name]
#         scores = cross_val_score(est, X_train, y_train, scoring=RMSE_SCORER, cv=cv)
#         rmse_vals = -scores
#         rows.append({"Model": name, "CV_RMSE_mean": rmse_vals.mean(), "CV_RMSE_std": rmse_vals.std()})
#     return pd.DataFrame(rows)
#
# # ====================== Linhas “mocadas” do ARTIGO ============================
# RM_ART_ROW = {"Scenario":"Baseline","Model":"RM_Art","RMSE_train":np.nan,"RMSE_test":8.04,"MAE_train":np.nan,"MAE_test":np.nan,
#               "R2_train":np.nan,"R2_test":0.392,"Corr_train":np.nan,"Corr_test":np.nan,
#               "HitRate20_test(%)":68.7,"HitRate10_test(%)":39.1,"AvgPrice_test":36.03,"RMSE/AvgPrice_test":0.224}
# NN_ART_ROW = {"Scenario":"Baseline","Model":"NN_Art","RMSE_train":np.nan,"RMSE_test":7.12,"MAE_train":np.nan,"MAE_test":np.nan,
#               "R2_train":np.nan,"R2_test":0.541,"Corr_train":np.nan,"Corr_test":np.nan,
#               "HitRate20_test(%)":76.1,"HitRate10_test(%)":45.9,"AvgPrice_test":36.03,"RMSE/AvgPrice_test":0.200}
#
# # =========================== Gráficos & Campeão ===============================
# def _trendline_and_r2(x: np.ndarray, y: np.ndarray):
#     x = np.asarray(x).ravel(); y = np.asarray(y).ravel()
#     a, b = np.polyfit(x, y, 1)
#     y_fit = a*x + b
#     ss_res = np.sum((y - y_fit)**2); ss_tot = np.sum((y - np.mean(y))**2)
#     r2 = 1.0 - ss_res/ss_tot if ss_tot>0 else np.nan
#     return a, b, r2
#
# def _nice_limits(x, y):
#     mn = float(min(np.min(x), np.min(y))); mx = float(max(np.max(x), np.max(y)))
#     span = mx - mn; pad = 0.05*span if span>0 else 1.0
#     return (mn - pad, mx + pad)
#
# def _scatter_with_identity_and_trend(save_path: str, y_true, y_pred, title: str):
#     y_true = np.asarray(y_true).ravel(); y_pred = np.asarray(y_pred).ravel()
#     a, b, r2 = _trendline_and_r2(y_true, y_pred)
#     plt.figure()
#     plt.scatter(y_true, y_pred, alpha=0.7, label="Pontos")
#     lo, hi = _nice_limits(y_true, y_pred)
#     xs = np.linspace(lo, hi, 200)
#     plt.plot(xs, xs, linewidth=1.0, label="y = x")
#     plt.plot(xs, a*xs + b, linewidth=1.2, label="Tendência")
#     plt.title(title); plt.xlabel("Measured (y)"); plt.ylabel("Predicted (ŷ)")
#     plt.grid(True, linestyle="--", alpha=0.5)
#     plt.xlim(lo, hi); plt.ylim(lo, hi); plt.gca().set_aspect('equal', adjustable='box')
#     eq_text = f"ŷ = {a:.3f}·y + {b:.3f}\nR² = {r2:.4f}"
#     plt.annotate(eq_text, xy=(0.05, 0.95), xycoords='axes fraction',
#                  ha='left', va='top', fontsize=10,
#                  bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.7))
#     plt.legend(loc="lower right")
#     plt.tight_layout(); plt.savefig(save_path); plt.close()
#
# def _choose_champion(metrics_df: pd.DataFrame) -> pd.Series:
#     df = metrics_df.sort_values(
#         by=["RMSE_test","HitRate20_test(%)","MAE_test","R2_test"],
#         ascending=[True, False, True, False],
#         kind="mergesort"
#     ).reset_index(drop=True)
#     return df.iloc[0]
#
# # ============================== EDA (gráficos) =================================
# def _plot_target_boxplot(save_path: str, y: np.ndarray, k: float, title_suffix=""):
#     y = np.asarray(y).ravel()
#     q1, q2, q3 = np.percentile(y, [25,50,75]); iqr = q3 - q1
#     lo, hi = q1 - k*iqr, q3 + k*iqr
#
#     plt.figure(figsize=(5,6))
#     plt.boxplot(y, vert=True, showfliers=True, whis=1.5)
#     for val, ls, lbl in [(q1,'--','Q1'), (q2,'-','Q2/mediana'), (q3,'--','Q3'), (lo,':',f'LB (Q1-{k}·IQR)'), (hi,':',f'UB (Q3+{k}·IQR)')]:
#         plt.axhline(val, linestyle=ls, linewidth=1.0, label=lbl)
#     plt.ylabel("y (house price per unit area)")
#     plt.title(f"Boxplot do alvo {title_suffix}".strip())
#     plt.legend(loc="best", fontsize=8)
#     plt.tight_layout(); plt.savefig(save_path); plt.close()
#
# def _plot_target_hist(save_path: str, y: np.ndarray, k: float, bins=20, title_suffix=""):
#     y = np.asarray(y).ravel()
#     q1, q2, q3 = np.percentile(y, [25,50,75]); iqr = q3 - q1
#     lo, hi = q1 - k*iqr, q3 + k*iqr
#
#     plt.figure(figsize=(7,5))
#     plt.hist(y, bins=bins, edgecolor="black", alpha=0.8)
#     for val, ls, lbl in [(q1,'--','Q1'), (q2,'-','Q2/mediana'), (q3,'--','Q3'), (lo,':',f'LB (Q1-{k}·IQR)'), (hi,':',f'UB (Q3+{k}·IQR)')]:
#         plt.axvline(val, linestyle=ls, linewidth=1.0, label=lbl)
#     plt.xlabel("y (house price per unit area)"); plt.ylabel("Frequência")
#     plt.title(f"Histograma do alvo {title_suffix}".strip())
#     plt.legend(loc="best", fontsize=8)
#     plt.tight_layout(); plt.savefig(save_path); plt.close()
#
# def _plot_features_boxplot(save_path: str, X: pd.DataFrame):
#     num = X.select_dtypes(include=[np.number])
#     plt.figure(figsize=(max(6, 1.2*len(num.columns)), 6))
#     num.boxplot(rot=45, grid=True)
#     plt.title("Boxplots das features (numéricas)")
#     plt.tight_layout(); plt.savefig(save_path); plt.close()
#
# def _plot_corr_matrix(save_path: str, df_num: pd.DataFrame):
#     corr = df_num.corr()
#     plt.figure(figsize=(7,6))
#     im = plt.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
#     plt.colorbar(im, fraction=0.046, pad=0.04)
#     plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right", fontsize=8)
#     plt.yticks(range(len(corr.index)), corr.index, fontsize=8)
#     plt.title("Matriz de correlação (numéricos)")
#     for i in range(corr.shape[0]):
#         for j in range(corr.shape[1]):
#             plt.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center", fontsize=7, color="black")
#     plt.tight_layout(); plt.savefig(save_path); plt.close()
#
# # =============================== Salvamento ===================================
# def save_all(outdir: str,
#              df_raw: pd.DataFrame,
#              base: dict, noout: dict,
#              best_base: str, best_noout: str,
#              champion_base: pd.Series, champion_noout: pd.Series,
#              y_col: str, iqr_k: float):
#     os.makedirs(outdir, exist_ok=True)
#     plots_dir = os.path.join(outdir, "plots"); os.makedirs(plots_dir, exist_ok=True)
#     eda_dir = os.path.join(outdir, "eda"); os.makedirs(eda_dir, exist_ok=True)
#     out_xlsx = os.path.join(outdir, "regression_results_TC1.xlsx")
#
#     def _slug(s: str) -> str:
#         s = re.sub(r"[^0-9a-zA-Z._-]+", "_", str(s))
#         while "__" in s: s = s.replace("__", "_")
#         return s.strip("_")
#
#     # EDA - figuras
#     _plot_target_boxplot(os.path.join(eda_dir, "boxplot_y_full.png"), df_raw[y_col].values, k=iqr_k, title_suffix="(dataset completo)")
#     _plot_target_hist(os.path.join(eda_dir, "hist_y_full.png"), df_raw[y_col].values, k=iqr_k, bins=20, title_suffix="(dataset completo)")
#     _plot_features_boxplot(os.path.join(eda_dir, "boxplots_features.png"), df_raw.drop(columns=[y_col]))
#     _plot_corr_matrix(os.path.join(eda_dir, "corr_matrix.png"), df_raw.select_dtypes(include=[np.number]))
#
#     with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
#         # EDA - tabelas
#         num_desc = df_raw.select_dtypes(include=[np.number]).describe().T
#         num_desc.to_excel(w, sheet_name="eda_numeric_describe")
#
#         y = base["y_train"].values
#         q1, q2, q3 = np.percentile(y, [25,50,75]); iqr = q3 - q1
#         lo, hi = q1 - iqr_k*iqr, q3 + iqr_k*iqr
#         y_stats = pd.DataFrame({
#             "Q1":[q1], "Q2_mediana":[q2], "Q3":[q3], "IQR":[iqr],
#             f"LB_Q1-{iqr_k}*IQR":[lo], f"UB_Q3+{iqr_k}*IQR":[hi],
#             "n_train":[len(y)],
#             "n_outliers_train":[int(np.sum((y<lo)|(y>hi)))],
#             "pct_outliers_train(%)":[100.0*float(np.mean((y<lo)|(y>hi)))]
#         })
#         y_stats.to_excel(w, sheet_name="eda_y_quartis_iqr", index=False)
#
#         df_raw.to_excel(w, sheet_name="raw_data", index=False)
#
#         # Métricas
#         m_base = base["metrics_df"].copy()
#         m_no = noout["metrics_df"].copy()
#         for m in (m_base, m_no):
#             nums = m.select_dtypes(include=[np.number]).columns
#             m[nums] = m[nums].round(4)
#         m_base.to_excel(w, sheet_name="metrics_baseline", index=False)
#         m_no.to_excel(w, sheet_name="metrics_no_outliers", index=False)
#         m_comb = pd.concat([m_base, m_no], ignore_index=True)
#         m_comb.to_excel(w, sheet_name="metrics_combined", index=False)
#
#         # CV
#         cvb = base["cv_df"].copy(); cvn = noout["cv_df"].copy()
#         for c in (cvb, cvn):
#             nums = c.select_dtypes(include=[np.number]).columns
#             c[nums] = c[nums].round(4)
#         cvb.to_excel(w, sheet_name="cv_summary_baseline", index=False)
#         cvn.to_excel(w, sheet_name="cv_summary_no_outliers", index=False)
#
#         # Predições do melhor de cada cenário
#         pd.DataFrame({
#             "y_train": base["y_train"].values,
#             "yhat_train": base["preds_train"][best_base],
#             "resid_train": base["y_train"].values - base["preds_train"][best_base]
#         }).to_excel(w, sheet_name=f"{best_base}_train_pred_BASE", index=False)
#         pd.DataFrame({
#             "y_test": base["y_test"].values,
#             "yhat_test": base["preds_test"][best_base]
#         }).to_excel(w, sheet_name=f"{best_base}_test_pred_BASE", index=False)
#
#         pd.DataFrame({
#             "y_train": noout["y_train"].values,
#             "yhat_train": noout["preds_train"][best_noout],
#             "resid_train": noout["y_train"].values - noout["preds_train"][best_noout]
#         }).to_excel(w, sheet_name=f"{best_noout}_train_pred_NOOUT", index=False)
#         pd.DataFrame({
#             "y_test": noout["y_test"].values,
#             "yhat_test": noout["preds_test"][best_noout]
#         }).to_excel(w, sheet_name=f"{best_noout}_test_pred_NOOUT", index=False)
#
#         # Campeões
#         cb = champion_base.to_dict();  cb["Scenario"] = "Baseline"
#         cn = champion_noout.to_dict(); cn["Scenario"] = "TrainNoOutliers"
#         champ_df = pd.DataFrame([cb, cn])
#         nums = champ_df.select_dtypes(include=[np.number]).columns
#         champ_df[nums] = champ_df[nums].round(4)
#         champ_df.to_excel(w, sheet_name="champion_summary", index=False)
#
#     # ======== Plots – TODOS os modelos (regressão e resíduos; train e test) ========
#     def _plot_all_for(bundle: dict, tag: str):
#         """
#         bundle: dict com keys ["y_train","y_test","preds_train","preds_test","residuals_train"]
#         tag: "BASE" ou "NOOUT" para sufixo nos arquivos
#         """
#         ytr = bundle["y_train"].values if isinstance(bundle["y_train"], pd.Series) else np.asarray(bundle["y_train"])
#         yte = bundle["y_test"].values if isinstance(bundle["y_test"], pd.Series) else np.asarray(bundle["y_test"])
#
#         for name in bundle["preds_train"].keys():
#             safe = _slug(name)
#
#             # Scatter y vs ŷ (Train/Test)
#             _scatter_with_identity_and_trend(
#                 os.path.join(plots_dir, f"scatter_train_{safe}_{tag}.png"),
#                 ytr, bundle["preds_train"][name],
#                 f"Measured vs Predicted (Train) – {name} [{tag}]"
#             )
#             _scatter_with_identity_and_trend(
#                 os.path.join(plots_dir, f"scatter_test_{safe}_{tag}.png"),
#                 yte, bundle["preds_test"][name],
#                 f"Measured vs Predicted (Test) – {name} [{tag}]"
#             )
#
#             # Resíduos (Train) – hist
#             resid_tr = bundle["residuals_train"][name]
#             plt.figure(); plt.hist(resid_tr, bins=20, edgecolor="black")
#             plt.title(f"Residuals Histogram (Train) – {name} [{tag}]")
#             plt.xlabel("Residual"); plt.ylabel("Frequency")
#             plt.tight_layout(); plt.savefig(os.path.join(plots_dir, f"residuals_hist_train_{safe}_{tag}.png")); plt.close()
#
#             # Resíduos (Test) – hist
#             resid_te = yte - bundle["preds_test"][name]
#             plt.figure(); plt.hist(resid_te, bins=20, edgecolor="black")
#             plt.title(f"Residuals Histogram (Test) – {name} [{tag}]")
#             plt.xlabel("Residual"); plt.ylabel("Frequency")
#             plt.tight_layout(); plt.savefig(os.path.join(plots_dir, f"residuals_hist_test_{safe}_{tag}.png")); plt.close()
#
#     _plot_all_for(base,  "BASE")
#     _plot_all_for(noout, "NOOUT")
#
# def print_console(outdir: str, base: dict, noout: dict,
#                   champion_base: pd.Series, champion_noout: pd.Series,
#                   y_train: pd.Series, iqr_k: float):
#     lines = []
#     lines.append(f"scikit-learn version: {skl_version}\n")
#
#     def df_to_str(df):
#         d = df.copy(); nums = d.select_dtypes(include=[np.number]).columns
#         d[nums] = d[nums].round(4); return d.to_string(index=False)
#
#     # EDA
#     q1, q2, q3 = np.percentile(y_train.values, [25,50,75]); iqr = q3 - q1
#     lb, ub = q1 - iqr_k*iqr, q3 + iqr_k*iqr
#     out_mask = (y_train.values < lb) | (y_train.values > ub)
#     lines.append("=== EDA (alvo no TREINO) ===")
#     lines.append(f"Q1={q1:.4f} | Q2/mediana={q2:.4f} | Q3={q3:.4f} | IQR={iqr:.4f}")
#     lines.append(f"Limites (k={iqr_k}): LB={lb:.4f}, UB={ub:.4f}")
#     lines.append(f"Outliers no treino: {int(out_mask.sum())} de {len(y_train)} ({100.0*out_mask.mean():.2f}%)\n")
#
#     lines.append("=== Metrics – Baseline (inclui RM_Art e NN_Art) ===")
#     lines.append(df_to_str(base["metrics_df"]))
#     lines.append("\n=== Metrics – TrainNoOutliers (inclui RM_Art e NN_Art) ===")
#     lines.append(df_to_str(noout["metrics_df"]))
#     lines.append("\n=== 5-fold CV – Baseline (RMSE) – top-4 ===")
#     lines.append(df_to_str(base["cv_df"]))
#     lines.append("\n=== 5-fold CV – TrainNoOutliers (RMSE) – top-4 ===")
#     lines.append(df_to_str(noout["cv_df"]))
#
#     lines.append("\n=== Champions ===")
#     cb = pd.DataFrame([champion_base]); cnc = cb.select_dtypes(include=[np.number]).columns
#     cb[cnc] = cb[cnc].round(4); lines.append("Baseline:\n" + cb.to_string(index=False))
#     cn = pd.DataFrame([champion_noout]); cnc2 = cn.select_dtypes(include=[np.number]).columns
#     cn[cnc2] = cn[cnc2].round(4); lines.append("\nTrainNoOutliers:\n" + cn.to_string(index=False))
#
#     text = "\n".join(lines)
#     print(text)
#     os.makedirs(outdir, exist_ok=True)
#     with open(os.path.join(outdir, "console_summary.txt"), "w", encoding="utf-8") as f:
#         f.write(text)
#
# # ================================= Main =======================================
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--data", dest="data_path", type=str, default=None,
#                         help="Caminho para 'Real estate valuation data set.xlsx'.")
#     parser.add_argument("--outdir", type=str, default="./outputs_tc1")
#     parser.add_argument("--test_size", type=float, default=0.30)
#     parser.add_argument("--seed", type=int, default=42)
#     parser.add_argument("--iqr_k", type=float, default=1.5, help="Fator k do IQR (p/ outliers no treino).")
#     args = parser.parse_args()
#
#     default_guess = os.path.join(os.getcwd(), "Real estate valuation data set.xlsx")
#     data_path = args.data_path or default_guess
#     if not os.path.isfile(data_path):
#         raise FileNotFoundError(f"Não encontrei o arquivo: {data_path}")
#
#     # Dados
#     df = pd.read_excel(data_path)
#     df.columns = [c.strip() for c in df.columns]
#     df = df.drop(columns=[c for c in df.columns if c.lower() in ["no", "id"]], errors="ignore")
#     y_col = detect_target_column(df)
#     X = df.drop(columns=[y_col]); y = df[y_col].astype(float)
#
#     # Split único
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.seed)
#
#     # --------- CENÁRIO 1: Baseline ---------
#     models = build_models(n_features=X_train.shape[1])
#     metrics_base, preds_tr_b, preds_te_b, resid_tr_b = evaluate_models(models, X_train, y_train, X_test, y_test, "Baseline")
#     # artigo
#     metrics_base = pd.concat([metrics_base, pd.DataFrame([RM_ART_ROW, NN_ART_ROW])], ignore_index=True)
#     # CV (top-4 dos SEUS modelos)
#     cv_base = crossval_top4(models, metrics_base[~metrics_base["Model"].isin(["RM_Art","NN_Art"])], X_train, y_train)
#     own_base = metrics_base[~metrics_base["Model"].isin(["RM_Art","NN_Art"])]
#     best_base_model = own_base.sort_values(by="RMSE_test").iloc[0]["Model"]
#     champion_base = _choose_champion(own_base)
#
#     base = {"metrics_df":metrics_base, "cv_df":cv_base, "preds_train":preds_tr_b, "preds_test":preds_te_b,
#             "residuals_train":resid_tr_b, "y_train":y_train, "y_test":y_test}
#
#     # --------- CENÁRIO 2: TrainNoOutliers ---------
#     mask = iqr_mask(y_train.values, k=args.iqr_k)
#     Xtr_no, ytr_no = X_train[mask], y_train[mask]
#     models_no = build_models(n_features=Xtr_no.shape[1])
#     metrics_no, preds_tr_n, preds_te_n, resid_tr_n = evaluate_models(models_no, Xtr_no, ytr_no, X_test, y_test, "TrainNoOutliers")
#     rm_art_no = RM_ART_ROW.copy(); rm_art_no["Scenario"]="TrainNoOutliers"
#     nn_art_no = NN_ART_ROW.copy(); nn_art_no["Scenario"]="TrainNoOutliers"
#     metrics_no = pd.concat([metrics_no, pd.DataFrame([rm_art_no, nn_art_no])], ignore_index=True)
#     cv_no = crossval_top4(models_no, metrics_no[~metrics_no["Model"].isin(["RM_Art","NN_Art"])], Xtr_no, ytr_no)
#     own_no = metrics_no[~metrics_no["Model"].isin(["RM_Art","NN_Art"])]
#     best_no_model = own_no.sort_values(by="RMSE_test").iloc[0]["Model"]
#     champion_noout = _choose_champion(own_no)
#
#     noout = {"metrics_df":metrics_no, "cv_df":cv_no, "preds_train":preds_tr_n, "preds_test":preds_te_n,
#              "residuals_train":resid_tr_n, "y_train":ytr_no, "y_test":y_test}
#
#     # --------- Salvar tudo ---------
#     save_all(args.outdir, df, base, noout, best_base_model, best_no_model,
#              champion_base, champion_noout, y_col, args.iqr_k)
#
#     # --------- Console ---------
#     print_console(args.outdir, base, noout, champion_base, champion_noout, y_train=y_train, iqr_k=args.iqr_k)
#
#     print("\nArquivos gerados em:", os.path.abspath(args.outdir))
#     print(" - Excel: regression_results_TC1.xlsx (abas: eda_*, raw_data, metrics_*, cv_summary_*, champion_summary, *_pred_*)")
#     print(" - Plots EDA: outputs_tc1/eda/ (boxplot_y_full.png, hist_y_full.png, boxplots_features.png, corr_matrix.png)")
#     print(" - Plots Modelos: outputs_tc1/plots/ (residuals_*_BASE/NOOUT, scatter_*_BASE/NOOUT)")
#     print(f"Melhor por RMSE (Baseline): {best_base_model}  |  Campeão (Baseline): {champion_base['Model']}")
#     print(f"Melhor por RMSE (NoOut):   {best_no_model}     |  Campeão (NoOut):   {champion_noout['Model']}")
#
# if __name__ == "__main__":
#     main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TC1 – Benchmark no 'Real estate valuation data set'
+ Comparação com artigo (RM_Art, NN_Art)
+ Gráficos (linha ideal + tendência + equação + R²)
+ Campeão (critério composto)
+ CENÁRIOS: Baseline vs Remoção de Outliers (IQR em y, apenas no TREINO)
+ EDA: Boxplots/histograma (quartis/IQR), correlação, describe
+ Diagnóstico de resíduos: hist + Normal/KDE + bandas ±σ + Q–Q separado
  (LEGENDA FORA à direita; PAINEL DE TESTES FORA no rodapé).

Execução:
  python regression_model.py --data "/caminho/Real estate valuation data set.xlsx" --outdir "./outputs_tc1"
"""

import argparse, os, warnings, re
from typing import Dict, Tuple
import numpy as np, pandas as pd, matplotlib.pyplot as plt

from sklearn import __version__ as skl_version
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from sklearn.base import BaseEstimator, RegressorMixin

# --- árvores e boosting
from sklearn.ensemble import RandomForestRegressor
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False
try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

# --- diagnósticos (normalidade)
from scipy import stats

import openpyxl  # engine Excel

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ========================== Compatibilidade sklearn ============================
def rmse(y_true, y_pred) -> float:
    try:
        return mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        return np.sqrt(mean_squared_error(y_true, y_pred))

def robust_pearsonr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    if yt.std() == 0 or yp.std() == 0:
        return 0.0
    return float(np.corrcoef(yt, yp)[0, 1])

RMSE_SCORER = make_scorer(lambda yt, yp: -rmse(yt, yp), greater_is_better=True)

# ================================ Utils =======================================
def detect_target_column(df: pd.DataFrame) -> str:
    exact = [c for c in df.columns if c.strip().lower() == "y house price of unit area"]
    if exact: return exact[0]
    candidates = [c for c in df.columns if "price" in c.lower()]
    return candidates[0] if candidates else df.columns[-1]

def make_scaled_regressor(base_estimator):
    """Escala/transforma X e padroniza y (TransformedTargetRegressor)."""
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pt", PowerTransformer(method="yeo-johnson")),
        ("model", base_estimator),
    ])
    return TransformedTargetRegressor(
        regressor=pipe,
        transformer=StandardScaler(with_mean=True, with_std=True)
    )

def make_scaled_X_only(base_estimator):
    """Escala/transforma apenas X (para estimadores que já cuidam de y)."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pt", PowerTransformer(method="yeo-johnson")),
        ("model", base_estimator),
    ])

def hit_rate(y_true: np.ndarray, y_pred: np.ndarray, pct: float) -> float:
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    rel = np.abs(yp - yt) / np.maximum(np.abs(yt), 1e-8)
    return 100.0 * float(np.mean(rel <= pct))

# -------- Outliers (IQR em y, apenas no treino) --------
def iqr_mask(y: np.ndarray, k: float = 1.5) -> np.ndarray:
    q1, q3 = np.percentile(y, 25), np.percentile(y, 75)
    iqr = q3 - q1
    lo, hi = q1 - k*iqr, q3 + k*iqr
    return (y >= lo) & (y <= hi)

# ============================== MLPs MANUAIS ==================================
def _tanh(z): return np.tanh(z)
def _dtanh(a): return 1.0 - a*a  # a = tanh(z)
def _xavier_limit(fan_in, fan_out): return np.sqrt(6.0 / (fan_in + fan_out))

def _init_wb(rng, fan_in, fan_out):
    lim = _xavier_limit(fan_in, fan_out)
    W = rng.uniform(-lim, lim, size=(fan_in, fan_out))
    b = np.zeros((fan_out,), dtype=float)
    return W, b

def _clip_inplace(arrs, max_norm: float):
    if max_norm is None or max_norm <= 0: return
    total = np.sqrt(sum(float(np.sum(a*a)) for a in arrs))
    if total > max_norm:
        scale = max_norm / (total + 1e-12)
        for a in arrs: a *= scale

class ManualMLPRegressor1H(BaseEstimator, RegressorMixin):
    def __init__(self,
                 hidden_size=18, lr=0.1, momentum=0.9, epochs=200, batch_size=32,
                 weight_decay=1e-4, early_stopping=True, patience=20,
                 lr_decay=True, lr_decay_factor=0.5, lr_patience=8,
                 clip_grad_norm=5.0, random_state=42, verbose=False):
        self.hidden_size = hidden_size
        self.lr = lr
        self.momentum = momentum
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.early_stopping = early_stopping
        self.patience = patience
        self.lr_decay = lr_decay
        self.lr_decay_factor = lr_decay_factor
        self.lr_patience = lr_patience
        self.clip_grad_norm = clip_grad_norm
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)

        n, d = X.shape; H = self.hidden_size
        W1, b1 = _init_wb(rng, d, H)
        W2, b2 = _init_wb(rng, H, 1)
        vW1 = np.zeros_like(W1); vb1 = np.zeros_like(b1)
        vW2 = np.zeros_like(W2); vb2 = np.zeros_like(b2)

        best_val = np.inf; best_params = (W1.copy(), b1.copy(), W2.copy(), b2.copy())
        wait_es = 0; wait_lr = 0; lr = float(self.lr)

        if self.early_stopping:
            idx = rng.permutation(n); vsz = max(1, int(0.1*n))
            val_idx = idx[:vsz]; tr_idx = idx[vsz:]
            Xtr, ytr = X[tr_idx], y[tr_idx]; Xval, yval = X[val_idx], y[val_idx]
        else:
            Xtr, ytr = X, y; Xval, yval = None, None

        for ep in range(self.epochs):
            order = rng.permutation(len(Xtr))
            for start in range(0, len(Xtr), self.batch_size):
                sl = order[start:start+self.batch_size]; xb = Xtr[sl]; yb = ytr[sl]
                a1 = _tanh(xb @ W1 + b1)
                yhat = a1 @ W2 + b2
                diff = (yhat - yb); grad_yhat = (2.0 / max(1, len(xb))) * diff
                dW2 = a1.T @ grad_yhat + self.weight_decay * W2
                db2 = grad_yhat.sum(axis=0)
                da1 = grad_yhat @ W2.T; dz1 = da1 * _dtanh(a1)
                dW1 = xb.T @ dz1 + self.weight_decay * W1
                db1 = dz1.sum(axis=0)
                _clip_inplace([dW1, db1, dW2, db2], self.clip_grad_norm)
                vW2 = self.momentum * vW2 - lr * dW2; vb2 = self.momentum * vb2 - lr * db2
                vW1 = self.momentum * vW1 - lr * dW1; vb1 = self.momentum * vb1 - lr * db1
                W2 += vW2; b2 += vb2; W1 += vW1; b1 += vb1

            if self.early_stopping:
                a1v = _tanh(Xval @ W1 + b1); yhat_val = a1v @ W2 + b2
                val_rmse = rmse(yval, yhat_val)
                if val_rmse + 1e-12 < best_val:
                    best_val = val_rmse; best_params = (W1.copy(), b1.copy(), W2.copy(), b2.copy())
                    wait_es = 0; wait_lr = 0
                else:
                    wait_es += 1; wait_lr += 1
                if self.lr_decay and wait_lr >= self.lr_patience:
                    lr *= self.lr_decay_factor; wait_lr = 0
                if self.verbose:
                    print(f"[1H][epoch {ep+1}] val_RMSE={val_rmse:.4f} best={best_val:.4f} lr={lr:.5f} wait={wait_es}")
                if wait_es >= self.patience: break

        if self.early_stopping:
            W1, b1, W2, b2 = best_params
        self.W1_, self.b1_, self.W2_, self.b2_ = W1, b1, W2, b2
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        a1 = _tanh(X @ self.W1_ + self.b1_)
        yhat = a1 @ self.W2_ + self.b2_
        return yhat.ravel()

class ManualMLPRegressor2H(BaseEstimator, RegressorMixin):
    def __init__(self,
                 hidden_size1=18, hidden_size2=18, lr=0.1, momentum=0.9, epochs=250, batch_size=32,
                 weight_decay=1e-4, early_stopping=True, patience=25,
                 lr_decay=True, lr_decay_factor=0.5, lr_patience=8,
                 clip_grad_norm=5.0, random_state=42, verbose=False):
        self.hidden_size1 = hidden_size1; self.hidden_size2 = hidden_size2
        self.lr = lr; self.momentum = momentum; self.epochs = epochs; self.batch_size = batch_size
        self.weight_decay = weight_decay; self.early_stopping = early_stopping; self.patience = patience
        self.lr_decay = lr_decay; self.lr_decay_factor = lr_decay_factor; self.lr_patience = lr_patience
        self.clip_grad_norm = clip_grad_norm; self.random_state = random_state; self.verbose = verbose

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float).reshape(-1, 1)
        n, d = X.shape; H1, H2 = self.hidden_size1, self.hidden_size2
        W1, b1 = _init_wb(rng, d,  H1); W2, b2 = _init_wb(rng, H1, H2); W3, b3 = _init_wb(rng, H2, 1)
        vW1 = np.zeros_like(W1); vb1 = np.zeros_like(b1)
        vW2 = np.zeros_like(W2); vb2 = np.zeros_like(b2)
        vW3 = np.zeros_like(W3); vb3 = np.zeros_like(b3)

        best_val = np.inf; best_params = (W1.copy(), b1.copy(), W2.copy(), b2.copy(), W3.copy(), b3.copy())
        wait_es = 0; wait_lr = 0; lr = float(self.lr)

        if self.early_stopping:
            idx = rng.permutation(n); vsz = max(1, int(0.1*n))
            val_idx = idx[:vsz]; tr_idx = idx[vsz:]
            Xtr, ytr = X[tr_idx], y[tr_idx]; Xval, yval = X[val_idx], y[val_idx]
        else:
            Xtr, ytr = X, y; Xval, yval = None, None

        for ep in range(self.epochs):
            order = rng.permutation(len(Xtr))
            for start in range(0, len(Xtr), self.batch_size):
                sl = order[start:start+self.batch_size]; xb = Xtr[sl]; yb = ytr[sl]
                a1 = _tanh(xb @ W1 + b1); a2 = _tanh(a1 @ W2 + b2); yhat = a2 @ W3 + b3
                diff = (yhat - yb); grad_yhat = (2.0 / max(1, len(xb))) * diff
                dW3 = a2.T @ grad_yhat + self.weight_decay * W3; db3 = grad_yhat.sum(axis=0)
                da2 = grad_yhat @ W3.T; dz2 = da2 * _dtanh(a2)
                dW2 = a1.T @ dz2 + self.weight_decay * W2; db2 = dz2.sum(axis=0)
                da1 = dz2 @ W2.T; dz1 = da1 * _dtanh(a1)
                dW1 = xb.T @ dz1 + self.weight_decay * W1; db1 = dz1.sum(axis=0)
                _clip_inplace([dW1, db1, dW2, db2, dW3, db3], self.clip_grad_norm)
                vW3 = self.momentum * vW3 - lr * dW3; vb3 = self.momentum * vb3 - lr * db3
                vW2 = self.momentum * vW2 - lr * dW2; vb2 = self.momentum * vb2 - lr * db2
                vW1 = self.momentum * vW1 - lr * dW1; vb1 = self.momentum * vb1 - lr * db1
                W3 += vW3; b3 += vb3; W2 += vW2; b2 += vb2; W1 += vW1; b1 += vb1

            if self.early_stopping:
                a1v = _tanh(Xval @ W1 + b1); a2v = _tanh(a1v @ W2 + b2)
                yhat_val = a2v @ W3 + b3; val_rmse = rmse(yval, yhat_val)
                if val_rmse + 1e-12 < best_val:
                    best_val = val_rmse; best_params = (W1.copy(), b1.copy(), W2.copy(), b2.copy(), W3.copy(), b3.copy())
                    wait_es = 0; wait_lr = 0
                else:
                    wait_es += 1; wait_lr += 1
                if self.lr_decay and wait_lr >= self.lr_patience:
                    lr *= self.lr_decay_factor; wait_lr = 0
                if self.verbose:
                    print(f"[2H][epoch {ep+1}] val_RMSE={val_rmse:.4f} best={best_val:.4f} lr={lr:.5f} wait={wait_es}")
                if wait_es >= self.patience: break

        if self.early_stopping:
            W1, b1, W2, b2, W3, b3 = best_params
        self.W1_, self.b1_, self.W2_, self.b2_, self.W3_, self.b3_ = W1, b1, W2, b2, W3, b3
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        a1 = _tanh(X @ self.W1_ + self.b1_)
        a2 = _tanh(a1 @ self.W2_)
        yhat = a2 @ self.W3_ + self.b3_
        return yhat.ravel()

# ========== Perceptron Logístico (Regra Delta Generalizada) ==========
def _logsig(z): return 1.0 / (1.0 + np.exp(-z))
def _dlogsig(y): return y * (1.0 - y)
def _dtanh_from_y(y): return 1.0 - y*y

class PLRegressorGD(BaseEstimator, RegressorMixin):
    """Perceptron Logístico para REGRESSÃO com Regra Delta Generalizada."""
    def __init__(self, activation="logsig", lr=0.05, epochs=500, batch_size=32,
                 momentum=0.9, weight_decay=1e-4, random_state=42, verbose=False,
                 eps_range=1e-3):
        assert activation in ("logsig", "tanh")
        self.activation = activation
        self.lr = lr; self.epochs = epochs; self.batch_size = batch_size
        self.momentum = momentum; self.weight_decay = weight_decay
        self.random_state = random_state; self.verbose = verbose
        self.eps_range = eps_range

    def _fit_target_scaler(self, y):
        y = y.ravel().astype(float)
        self.y_min_ = float(np.min(y)); self.y_max_ = float(np.max(y))
        rng = self.y_max_ - self.y_min_
        self.y_rng_ = rng if rng > 0 else 1.0

    def _y_to_act_space(self, y):
        y = y.ravel().astype(float)
        y01 = (y - self.y_min_) / self.y_rng_
        y01 = np.clip(y01, 0.0 + self.eps_range, 1.0 - self.eps_range)
        if self.activation == "logsig":
            return y01
        else:
            return y01 * (2.0 - 2*self.eps_range) - (1.0 - self.eps_range)

    def _y_from_act_space(self, y_act):
        y_act = np.asarray(y_act, dtype=float).ravel()
        if self.activation == "logsig":
            y01 = np.clip(y_act, 0.0, 1.0)
        else:
            y01 = (np.clip(y_act, -1.0, 1.0) + 1.0) / 2.0
        return y01 * self.y_rng_ + self.y_min_

    def _forward(self, X, w, b):
        z = X @ w + b
        return _logsig(z) if self.activation == "logsig" else np.tanh(z)

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float).ravel()
        n, d = X.shape
        self._fit_target_scaler(y)
        yd = self._y_to_act_space(y).reshape(-1, 1)

        w = rng.normal(0.0, 0.01, size=(d,1)); b = 0.0
        vw = np.zeros_like(w); vb = 0.0

        for ep in range(self.epochs):
            order = rng.permutation(n)
            for start in range(0, n, self.batch_size):
                sl = order[start:start+self.batch_size]
                xb = X[sl]; db = yd[sl]
                yb = self._forward(xb, w, b)
                e  = db - yb
                phi_prime = _dlogsig(yb) if self.activation == "logsig" else _dtanh_from_y(yb)
                delta = e * phi_prime
                grad_w = xb.T @ delta / max(1, len(sl)) + self.weight_decay * w
                grad_b = float(np.sum(delta) / max(1, len(sl)))
                vw = self.momentum * vw + self.lr * grad_w
                vb = self.momentum * vb + self.lr * grad_b
                w += vw; b += vb
            if self.verbose and (ep+1) % 50 == 0:
                y_tr_pred = self._y_from_act_space(self._forward(X, w, b))
                print(f"[PL-{self.activation}][{ep+1}/{self.epochs}] RMSE={rmse(y, y_tr_pred):.4f}")

        self.w_ = w; self.b_ = b
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        y_act = self._forward(X, self.w_, self.b_)
        return self._y_from_act_space(y_act).ravel()

# ============================ Modelos & Avaliação =============================
def build_models(n_features: int) -> Dict[str, object]:
    gamma_rbf = 1.0 / max(n_features, 1)
    cv5 = KFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "OLS_LinearRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("pt", PowerTransformer(method="yeo-johnson")),
            ("model", LinearRegression())
        ]),
        "RidgeCV": Pipeline([
            ("scaler", StandardScaler()),
            ("pt", PowerTransformer(method="yeo-johnson")),
            ("model", RidgeCV(alphas=np.logspace(-3, 3, 25), cv=cv5))
        ]),
        "LassoCV": Pipeline([
            ("scaler", StandardScaler()),
            ("pt", PowerTransformer(method="yeo-johnson")),
            ("model", LassoCV(alphas=None, cv=cv5, max_iter=20000))
        ]),
        "ElasticNetCV": Pipeline([
            ("scaler", StandardScaler()),
            ("pt", PowerTransformer(method="yeo-johnson")),
            ("model", ElasticNetCV(l1_ratio=[.1,.3,.5,.7,.9,.95,.99,1.0], cv=cv5, max_iter=20000))
        ]),
        "KernelRidge_RBF": make_scaled_regressor(KernelRidge(alpha=1.0, kernel="rbf", gamma=gamma_rbf)),
        "SVR_RBF": make_scaled_regressor(SVR(C=10.0, epsilon=0.1, kernel="rbf", gamma="scale", max_iter=-1)),
        "MLP_1H_18": make_scaled_regressor(MLPRegressor(
            hidden_layer_sizes=(18,), activation="relu", solver="adam",
            learning_rate_init=0.01, max_iter=5000, early_stopping=True, n_iter_no_change=25, random_state=42)),
        "MLP_2H_18_18": make_scaled_regressor(MLPRegressor(
            hidden_layer_sizes=(18,18), activation="relu", solver="adam",
            learning_rate_init=0.01, max_iter=6000, early_stopping=True, n_iter_no_change=25, random_state=42)),
        "MLP_manual_1H": make_scaled_regressor(ManualMLPRegressor1H(
            hidden_size=18, lr=0.1, momentum=0.9, epochs=200, batch_size=32,
            weight_decay=1e-4, early_stopping=True, patience=20,
            lr_decay=True, lr_decay_factor=0.5, lr_patience=8,
            clip_grad_norm=5.0, random_state=42, verbose=False)),
        "MLP_manual_2H": make_scaled_regressor(ManualMLPRegressor2H(
            hidden_size1=18, hidden_size2=18, lr=0.1, momentum=0.9, epochs=250, batch_size=32,
            weight_decay=1e-4, early_stopping=True, patience=25,
            lr_decay=True, lr_decay_factor=0.5, lr_patience=8,
            clip_grad_norm=5.0, random_state=42, verbose=False)),
        "PL_LogSig_LMS": make_scaled_X_only(PLRegressorGD(
            activation="logsig", lr=0.05, epochs=500, batch_size=32,
            momentum=0.9, weight_decay=1e-4, random_state=42, verbose=False)),
        "PL_Tanh_LMS": make_scaled_X_only(PLRegressorGD(
            activation="tanh", lr=0.05, epochs=500, batch_size=32,
            momentum=0.9, weight_decay=1e-4, random_state=42, verbose=False)),
    }

    models["RandomForest"] = RandomForestRegressor(
        n_estimators=500, max_depth=None, min_samples_leaf=1, n_jobs=-1, random_state=42
    )

    if HAS_XGB:
        models["XGBoost"] = XGBRegressor(
            n_estimators=600, learning_rate=0.05, max_depth=4,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            objective="reg:squarederror", n_jobs=-1, random_state=42
        )

    if HAS_LGBM:
        models["LightGBM"] = LGBMRegressor(
            n_estimators=800, learning_rate=0.05, num_leaves=31, max_depth=-1,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, random_state=42, n_jobs=-1
        )

    return models

def evaluate_models(models: Dict[str, object],
                    X_train: pd.DataFrame, y_train: pd.Series,
                    X_test: pd.DataFrame, y_test: pd.Series,
                    scenario: str) -> Tuple[pd.DataFrame, Dict, Dict, Dict]:
    records = []
    preds_train, preds_test, residuals_train = {}, {}, {}
    avg_price_test = float(np.mean(y_test))

    for name, model in models.items():
        model.fit(X_train, y_train)
        yhat_tr = model.predict(X_train)
        yhat_te = model.predict(X_test)

        preds_train[name] = yhat_tr
        preds_test[name] = yhat_te
        residuals_train[name] = y_train.values - yhat_tr

        rec = {
            "Scenario": scenario,
            "Model": name,
            "RMSE_train": rmse(y_train, yhat_tr),
            "RMSE_test":  rmse(y_test,  yhat_te),
            "MAE_train": mean_absolute_error(y_train, yhat_tr),
            "MAE_test":  mean_absolute_error(y_test,  yhat_te),
            "R2_train":  r2_score(y_train, yhat_tr),
            "R2_test":   r2_score(y_test,  yhat_te),
            "Corr_train": robust_pearsonr(y_train, yhat_tr),
            "Corr_test":  robust_pearsonr(y_test,  yhat_te),
            "HitRate20_test(%)": hit_rate(y_test, yhat_te, 0.20),
            "HitRate10_test(%)": hit_rate(y_test, yhat_te, 0.10),
            "AvgPrice_test": avg_price_test,
            "RMSE/AvgPrice_test": (rmse(y_test, yhat_te) / avg_price_test) if avg_price_test != 0 else np.nan,
        }
        records.append(rec)

    metrics_df = pd.DataFrame(records).sort_values(by=["Scenario","RMSE_test"]).reset_index(drop=True)
    return metrics_df, preds_train, preds_test, residuals_train

def crossval_top4(models: Dict[str, object], metrics_df: pd.DataFrame,
                  X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    top_names = metrics_df.sort_values("RMSE_test")["Model"].unique()[:4].tolist()
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    rows = []
    for name in top_names:
        est = models[name]
        scores = cross_val_score(est, X_train, y_train, scoring=RMSE_SCORER, cv=cv)
        rmse_vals = -scores
        rows.append({"Model": name, "CV_RMSE_mean": rmse_vals.mean(), "CV_RMSE_std": rmse_vals.std()})
    return pd.DataFrame(rows)

# ====================== Linhas “mocadas” do ARTIGO ============================
RM_ART_ROW = {"Scenario":"Baseline","Model":"RM_Art","RMSE_train":np.nan,"RMSE_test":8.04,"MAE_train":np.nan,"MAE_test":np.nan,
              "R2_train":np.nan,"R2_test":0.392,"Corr_train":np.nan,"Corr_test":np.nan,
              "HitRate20_test(%)":68.7,"HitRate10_test(%)":39.1,"AvgPrice_test":36.03,"RMSE/AvgPrice_test":0.224}
NN_ART_ROW = {"Scenario":"Baseline","Model":"NN_Art","RMSE_train":np.nan,"RMSE_test":7.12,"MAE_train":np.nan,"MAE_test":np.nan,
              "R2_train":np.nan,"R2_test":0.541,"Corr_train":np.nan,"Corr_test":np.nan,
              "HitRate20_test(%)":76.1,"HitRate10_test(%)":45.9,"AvgPrice_test":36.03,"RMSE/AvgPrice_test":0.200}

# =========================== Gráficos & Campeão ===============================
def _trendline_and_r2(x: np.ndarray, y: np.ndarray):
    x = np.asarray(x).ravel(); y = np.asarray(y).ravel()
    a, b = np.polyfit(x, y, 1)
    y_fit = a*x + b
    ss_res = np.sum((y - y_fit)**2); ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1.0 - ss_res/ss_tot if ss_tot>0 else np.nan
    return a, b, r2

def _nice_limits(x, y):
    mn = float(min(np.min(x), np.min(y))); mx = float(max(np.max(x), np.max(y)))
    span = mx - mn; pad = 0.05*span if span>0 else 1.0
    return (mn - pad, mx + pad)

def _scatter_with_identity_and_trend(save_path: str, y_true, y_pred, title: str):
    y_true = np.asarray(y_true).ravel(); y_pred = np.asarray(y_pred).ravel()
    a, b, r2 = _trendline_and_r2(y_true, y_pred)
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.7, label="Pontos")
    lo, hi = _nice_limits(y_true, y_pred)
    xs = np.linspace(lo, hi, 200)
    plt.plot(xs, xs, linewidth=1.0, label="y = x")
    plt.plot(xs, a*xs + b, linewidth=1.2, label="Tendência")
    plt.title(title); plt.xlabel("Measured (y)"); plt.ylabel("Predicted (ŷ)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xlim(lo, hi); plt.ylim(lo, hi); plt.gca().set_aspect('equal', adjustable='box')
    eq_text = f"ŷ = {a:.3f}·y + {b:.3f}\nR² = {r2:.4f}"
    plt.annotate(eq_text, xy=(0.05, 0.95), xycoords='axes fraction',
                 ha='left', va='top', fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.7))
    plt.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(save_path); plt.close()

def _choose_champion(metrics_df: pd.DataFrame) -> pd.Series:
    df = metrics_df.sort_values(
        by=["RMSE_test","HitRate20_test(%)","MAE_test","R2_test"],
        ascending=[True, False, True, False],
        kind="mergesort"
    ).reset_index(drop=True)
    return df.iloc[0]

# ============================== EDA (gráficos) =================================
def _plot_target_boxplot(save_path: str, y: np.ndarray, k: float, title_suffix=""):
    y = np.asarray(y).ravel()
    q1, q2, q3 = np.percentile(y, [25,50,75]); iqr = q3 - q1
    lo, hi = q1 - k*iqr, q3 + k*iqr

    plt.figure(figsize=(5,6))
    plt.boxplot(y, vert=True, showfliers=True, whis=1.5)
    for val, ls, lbl in [(q1,'--','Q1'), (q2,'-','Q2/mediana'), (q3,'--','Q3'), (lo,':',f'LB (Q1-{k}·IQR)'), (hi,':',f'UB (Q3+{k}·IQR)')]:
        plt.axhline(val, linestyle=ls, linewidth=1.0, label=lbl)
    plt.ylabel("y (house price per unit area)")
    plt.title(f"Boxplot do alvo {title_suffix}".strip())
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout(); plt.savefig(save_path); plt.close()

def _plot_target_hist(save_path: str, y: np.ndarray, k: float, bins=20, title_suffix=""):
    y = np.asarray(y).ravel()
    q1, q2, q3 = np.percentile(y, [25,50,75]); iqr = q3 - q1
    lo, hi = q1 - k*iqr, q3 + k*iqr

    plt.figure(figsize=(7,5))
    plt.hist(y, bins=bins, edgecolor="black", alpha=0.8)
    for val, ls, lbl in [(q1,'--','Q1'), (q2,'-','Q2/mediana'), (q3,'--','Q3'), (lo,':',f'LB (Q1-{k}·IQR)'), (hi,':',f'UB (Q3+{k}·IQR)')]:
        plt.axvline(val, linestyle=ls, linewidth=1.0, label=lbl)
    plt.xlabel("y (house price per unit area)"); plt.ylabel("Frequência")
    plt.title(f"Histograma do alvo {title_suffix}".strip())
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout(); plt.savefig(save_path); plt.close()

def _plot_features_boxplot(save_path: str, X: pd.DataFrame):
    num = X.select_dtypes(include=[np.number])
    plt.figure(figsize=(max(6, 1.2*len(num.columns)), 6))
    num.boxplot(rot=45, grid=True)
    plt.title("Boxplots das features (numéricas)")
    plt.tight_layout(); plt.savefig(save_path); plt.close()

def _plot_corr_matrix(save_path: str, df_num: pd.DataFrame):
    corr = df_num.corr()
    plt.figure(figsize=(7,6))
    im = plt.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right", fontsize=8)
    plt.yticks(range(len(corr.index)), corr.index, fontsize=8)
    plt.title("Matriz de correlação (numéricos)")
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            plt.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center", fontsize=7, color="black")
    plt.tight_layout(); plt.savefig(save_path); plt.close()

# ====================== Diagnóstico de resíduos (hist + testes + QQ) ======================
def _fd_bins(x, min_bins=10, max_bins=50) -> int:
    """Número de bins pela regra de Freedman–Diaconis (robusta a outliers)."""
    x = np.asarray(x).ravel()
    if x.size == 0:
        return 20
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    if iqr <= 0:
        return 20
    bw = 2 * iqr * (x.size ** (-1/3))
    span = float(np.max(x) - np.min(x))
    bins = int(np.clip(span / max(bw, 1e-12), min_bins, max_bins))
    return bins

def residuals_diagnostics_plot(y_true, y_pred, save_path: str, title: str) -> dict:
    """
    Histograma dos resíduos com:
      - Normal(μ̂,σ̂²) (linha vermillion)
      - KDE (linha verde tracejada)
      - Faixas ±1σ/±2σ (sombras suaves)
      - LINHA pontilhada em μ̂
    LEGENDA: fora (direita). PAINEL DE TESTES: fora (rodapé).
    Q–Q plot é salvo à parte: <save_path>_qq.png
    """
    # Paleta Okabe–Ito (acessível daltônicos)
    color_hist = "#0072B2"     # azul
    color_norm = "#D55E00"     # vermillion
    color_kde  = "#009E73"     # verde
    color_mean = "#CC79A7"     # roxo

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    res = y_true - y_pred
    n = res.size
    mu = float(np.mean(res))
    sigma = float(np.std(res, ddof=1))
    bins = _fd_bins(res)

    # --- Figura (reservar espaço p/ legenda à direita e painel no rodapé) ---
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    fig.subplots_adjust(right=0.78, bottom=0.24, top=0.90)

    # Histograma (densidade)
    ax.hist(res, bins=bins, density=True, alpha=0.70, edgecolor="#2f2f2f",
            color=color_hist, label="Resíduos")

    # Curva Normal ajustada
    xs = np.linspace(mu - 4 * sigma if sigma > 0 else np.min(res),
                     mu + 4 * sigma if sigma > 0 else np.max(res), 400)
    if sigma > 0:
        ax.plot(xs, stats.norm.pdf(xs, mu, sigma), lw=2.2, color=color_norm,
                label="Normal(μ̂,σ̂²)")

    # KDE
    try:
        kde = stats.gaussian_kde(res)
        ax.plot(xs, kde(xs), lw=2.2, linestyle="--", color=color_kde, label="KDE")
    except Exception:
        pass

    # Bandas ±σ e ±2σ (mesma cor do histograma, baixa opacidade)
    for k in (2, 1):
        ax.axvspan(mu - k * sigma, mu + k * sigma,
                   color=color_hist, alpha=0.06 if k == 2 else 0.12, lw=0)

    # Linha da média
    ax.axvline(mu, lw=1.6, linestyle=":", color=color_mean, label="μ̂")

    # Eixos, grade, título
    ax.set_title(title, pad=8)
    ax.set_xlabel("Residual")
    ax.set_ylabel("Density")
    ax.grid(True, linestyle=":", alpha=0.35)

    # LEGENDA — fora (direita)
    ax.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", borderaxespad=0.0, fontsize=9)

    # --- Testes/medidas de forma ---
    try:
        if n > 5000:
            rng = np.random.default_rng(0)
            sample = rng.choice(res, size=5000, replace=False)
            sh_W, sh_p = stats.shapiro(sample)
        else:
            sh_W, sh_p = stats.shapiro(res)
    except Exception:
        sh_W, sh_p = (np.nan, np.nan)

    try:
        if n >= 8:
            k2_stat, k2_p = stats.normaltest(res)
        else:
            k2_stat, k2_p = (np.nan, np.nan)
    except Exception:
        k2_stat, k2_p = (np.nan, np.nan)

    try:
        jb_stat, jb_p = stats.jarque_bera(res)
    except Exception:
        jb_stat, jb_p = (np.nan, np.nan)

    try:
        skew = float(stats.skew(res, bias=False))
        exkurt = float(stats.kurtosis(res, fisher=True, bias=False))
    except Exception:
        skew, exkurt = (np.nan, np.nan)

    # Painel de testes FORA (rodapé da figura, à esquerda)
    txt = (
        f"n={n} | μ̂={mu:.2f}  σ̂={sigma:.2f} | "
        f"skew={skew:.2f}  ex.kurt={exkurt:.2f} | "
        f"Shapiro p={np.nan if pd.isna(sh_p) else float(sh_p):.3g} | "
        f"D’Agostino p={np.nan if pd.isna(k2_p) else float(k2_p):.3g} | "
        f"Jarque–Bera p={np.nan if pd.isna(jb_p) else float(jb_p):.3g}"
    )
    fig.text(0.02, 0.04, txt,
             ha="left", va="bottom", fontsize=9.5,
             bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#999999", alpha=0.95))

    # salvar com bbox 'tight' para incluir legenda e rodapé
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Q–Q plot (arquivo separado) ---
    qq_path = save_path.replace(".png", "_qq.png")
    fig2, ax2 = plt.subplots(figsize=(4.8, 4.8))
    stats.probplot(res, dist="norm", plot=ax2)
    ax2.set_title(f"Normal Q–Q — {title}", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.35)
    fig2.tight_layout()
    fig2.savefig(qq_path, dpi=150)
    plt.close(fig2)

    return {
        "Model": None, "Split": None,
        "n": int(n), "mean": mu, "std": sigma,
        "skew": skew, "excess_kurtosis": exkurt,
        "shapiro_W": float(sh_W) if not pd.isna(sh_W) else np.nan,
        "shapiro_p": float(sh_p) if not pd.isna(sh_p) else np.nan,
        "dagostino_K2": float(k2_stat) if not pd.isna(k2_stat) else np.nan,
        "dagostino_p": float(k2_p) if not pd.isna(k2_p) else np.nan,
        "jarque_bera": float(jb_stat) if not pd.isna(jb_stat) else np.nan,
        "jarque_bera_p": float(jb_p) if not pd.isna(jb_p) else np.nan,
        "image_path": save_path, "qqplot_path": qq_path,
    }

# =============================== Salvamento ===================================
def save_all(outdir: str,
             df_raw: pd.DataFrame,
             base: dict, noout: dict,
             best_base: str, best_noout: str,
             champion_base: pd.Series, champion_noout: pd.Series,
             y_col: str, iqr_k: float):
    os.makedirs(outdir, exist_ok=True)
    plots_dir = os.path.join(outdir, "plots"); os.makedirs(plots_dir, exist_ok=True)
    eda_dir = os.path.join(outdir, "eda"); os.makedirs(eda_dir, exist_ok=True)
    out_xlsx = os.path.join(outdir, "regression_results_TC1.xlsx")

    def _slug(s: str) -> str:
        s = re.sub(r"[^0-9a-zA-Z._-]+", "_", str(s))
        while "__" in s: s = s.replace("__", "_")
        return s.strip("_")

    # EDA - figuras
    _plot_target_boxplot(os.path.join(eda_dir, "boxplot_y_full.png"), df_raw[y_col].values, k=iqr_k, title_suffix="(dataset completo)")
    _plot_target_hist(os.path.join(eda_dir, "hist_y_full.png"), df_raw[y_col].values, k=iqr_k, bins=20, title_suffix="(dataset completo)")
    _plot_features_boxplot(os.path.join(eda_dir, "boxplots_features.png"), df_raw.drop(columns=[y_col]))
    _plot_corr_matrix(os.path.join(eda_dir, "corr_matrix.png"), df_raw.select_dtypes(include=[np.number]))

    # ======== Plots – TODOS os modelos (regressão e resíduos; train e test) ========
    def _plot_all_for(bundle: dict, tag: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        ytr = bundle["y_train"].values if isinstance(bundle["y_train"], pd.Series) else np.asarray(bundle["y_train"])
        yte = bundle["y_test"].values if isinstance(bundle["y_test"], pd.Series) else np.asarray(bundle["y_test"])

        diag_rows_tr, diag_rows_te = [], []

        for name in bundle["preds_train"].keys():
            safe = _slug(name)

            # Scatter y vs ŷ
            _scatter_with_identity_and_trend(
                os.path.join(plots_dir, f"scatter_train_{safe}_{tag}.png"),
                ytr, bundle["preds_train"][name],
                f"Measured vs Predicted (Train) – {name} [{tag}]"
            )
            _scatter_with_identity_and_trend(
                os.path.join(plots_dir, f"scatter_test_{safe}_{tag}.png"),
                yte, bundle["preds_test"][name],
                f"Measured vs Predicted (Test) – {name} [{tag}]"
            )

            # Resíduos (Train)
            p_train = os.path.join(plots_dir, f"residuals_train_{safe}_{tag}.png")
            dtr = residuals_diagnostics_plot(
                ytr, bundle["preds_train"][name],
                save_path=p_train,
                title=f"Residuals (Train) – {name} [{tag}]"
            )
            dtr["Model"] = name; dtr["Split"] = "Train"
            diag_rows_tr.append(dtr)

            # Resíduos (Test)
            p_test = os.path.join(plots_dir, f"residuals_test_{safe}_{tag}.png")
            dte = residuals_diagnostics_plot(
                yte, bundle["preds_test"][name],
                save_path=p_test,
                title=f"Residuals (Test) – {name} [{tag}]"
            )
            dte["Model"] = name; dte["Split"] = "Test"
            diag_rows_te.append(dte)

        df_tr = pd.DataFrame(diag_rows_tr)
        df_te = pd.DataFrame(diag_rows_te)
        return df_tr, df_te

    diag_tr_base, diag_te_base   = _plot_all_for(base,  "BASE")
    diag_tr_no,   diag_te_no     = _plot_all_for(noout, "NOOUT")

    # CSVs de diagnósticos
    diag_tr_base.to_csv(os.path.join(outdir, "residuals_diag_train_BASE.csv"), index=False)
    diag_te_base.to_csv(os.path.join(outdir, "residuals_diag_test_BASE.csv"), index=False)
    diag_tr_no.to_csv(  os.path.join(outdir, "residuals_diag_train_NOOUT.csv"), index=False)
    diag_te_no.to_csv(  os.path.join(outdir, "residuals_diag_test_NOOUT.csv"), index=False)

    # ========================= Excel com tudo =========================
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        # EDA - tabelas
        num_desc = df_raw.select_dtypes(include=[np.number]).describe().T
        num_desc.to_excel(w, sheet_name="eda_numeric_describe")

        y = base["y_train"].values
        q1, q2, q3 = np.percentile(y, [25,50,75]); iqr = q3 - q1
        lo, hi = q1 - iqr_k*iqr, q3 + iqr_k*iqr
        y_stats = pd.DataFrame({
            "Q1":[q1], "Q2_mediana":[q2], "Q3":[q3], "IQR":[iqr],
            f"LB_Q1-{iqr_k}*IQR":[lo], f"UB_Q3+{iqr_k}*IQR":[hi],
            "n_train":[len(y)],
            "n_outliers_train":[int(np.sum((y<lo)|(y>hi)))],
            "pct_outliers_train(%)":[100.0*float(np.mean((y<lo)|(y>hi)))]
        })
        y_stats.to_excel(w, sheet_name="eda_y_quartis_iqr", index=False)

        df_raw.to_excel(w, sheet_name="raw_data", index=False)

        # Métricas
        m_base = base["metrics_df"].copy()
        m_no = noout["metrics_df"].copy()
        for m in (m_base, m_no):
            nums = m.select_dtypes(include=[np.number]).columns
            m[nums] = m[nums].round(4)
        m_base.to_excel(w, sheet_name="metrics_baseline", index=False)
        m_no.to_excel(w, sheet_name="metrics_no_outliers", index=False)
        m_comb = pd.concat([m_base, m_no], ignore_index=True)
        m_comb.to_excel(w, sheet_name="metrics_combined", index=False)

        # CV
        cvb = base["cv_df"].copy(); cvn = noout["cv_df"].copy()
        for c in (cvb, cvn):
            nums = c.select_dtypes(include=[np.number]).columns
            c[nums] = c[nums].round(4)
        cvb.to_excel(w, sheet_name="cv_summary_baseline", index=False)
        cvn.to_excel(w, sheet_name="cv_summary_no_outliers", index=False)

        # Predições do melhor de cada cenário
        pd.DataFrame({
            "y_train": base["y_train"].values,
            "yhat_train": base["preds_train"][best_base],
            "resid_train": base["y_train"].values - base["preds_train"][best_base]
        }).to_excel(w, sheet_name=f"{best_base}_train_pred_BASE", index=False)
        pd.DataFrame({
            "y_test": base["y_test"].values,
            "yhat_test": base["preds_test"][best_base]
        }).to_excel(w, sheet_name=f"{best_base}_test_pred_BASE", index=False)

        pd.DataFrame({
            "y_train": noout["y_train"].values,
            "yhat_train": noout["preds_train"][best_noout],
            "resid_train": noout["y_train"].values - noout["preds_train"][best_noout]
        }).to_excel(w, sheet_name=f"{best_noout}_train_pred_NOOUT", index=False)
        pd.DataFrame({
            "y_test": noout["y_test"].values,
            "yhat_test": noout["preds_test"][best_noout]
        }).to_excel(w, sheet_name=f"{best_noout}_test_pred_NOOUT", index=False)

        # Campeões
        cb = champion_base.to_dict();  cb["Scenario"] = "Baseline"
        cn = champion_noout.to_dict(); cn["Scenario"] = "TrainNoOutliers"
        champ_df = pd.DataFrame([cb, cn])
        nums = champ_df.select_dtypes(include=[np.number]).columns
        champ_df[nums] = champ_df[nums].round(4)
        champ_df.to_excel(w, sheet_name="champion_summary", index=False)

        # Diagnósticos de resíduos (tabelas)
        for df_tab, sh in [
            (diag_tr_base, "residuals_diag_tr_BASE"),
            (diag_te_base, "residuals_diag_te_BASE"),
            (diag_tr_no,   "residuals_diag_tr_NOOUT"),
            (diag_te_no,   "residuals_diag_te_NOOUT"),
        ]:
            d = df_tab.copy()
            numcols = d.select_dtypes(include=[np.number]).columns
            d[numcols] = d[numcols].round(6)
            d.to_excel(w, sheet_name=sh, index=False)

def print_console(outdir: str, base: dict, noout: dict,
                  champion_base: pd.Series, champion_noout: pd.Series,
                  y_train: pd.Series, iqr_k: float):
    lines = []
    lines.append(f"scikit-learn version: {skl_version}\n")

    def df_to_str(df):
        d = df.copy(); nums = d.select_dtypes(include=[np.number]).columns
        d[nums] = d[nums].round(4); return d.to_string(index=False)

    # EDA
    q1, q2, q3 = np.percentile(y_train.values, [25,50,75]); iqr = q3 - q1
    lb, ub = q1 - iqr_k*iqr, q3 + iqr_k*iqr
    out_mask = (y_train.values < lb) | (y_train.values > ub)
    lines.append("=== EDA (alvo no TREINO) ===")
    lines.append(f"Q1={q1:.4f} | Q2/mediana={q2:.4f} | Q3={q3:.4f} | IQR={iqr:.4f}")
    lines.append(f"Limites (k={iqr_k}): LB={lb:.4f}, UB={ub:.4f}")
    lines.append(f"Outliers no treino: {int(out_mask.sum())} de {len(y_train)} ({100.0*out_mask.mean():.2f}%)\n")

    lines.append("=== Metrics – Baseline (inclui RM_Art e NN_Art) ===")
    lines.append(df_to_str(base["metrics_df"]))
    lines.append("\n=== Metrics – TrainNoOutliers (inclui RM_Art e NN_Art) ===")
    lines.append(df_to_str(noout["metrics_df"]))
    lines.append("\n=== 5-fold CV – Baseline (RMSE) – top-4 ===")
    lines.append(df_to_str(base["cv_df"]))
    lines.append("\n=== 5-fold CV – TrainNoOutliers (RMSE) – top-4 ===")
    lines.append(df_to_str(noout["cv_df"]))

    lines.append("\n=== Champions ===")
    cb = pd.DataFrame([champion_base]); cnc = cb.select_dtypes(include=[np.number]).columns
    cb[cnc] = cb[cnc].round(4); lines.append("Baseline:\n" + cb.to_string(index=False))
    cn = pd.DataFrame([champion_noout]); cnc2 = cn.select_dtypes(include=[np.number]).columns
    cn[cnc2] = cn[cnc2].round(4); lines.append("\nTrainNoOutliers:\n" + cn.to_string(index=False))

    text = "\n".join(lines)
    print(text)
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "console_summary.txt"), "w", encoding="utf-8") as f:
        f.write(text)

# ================================= Main =======================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", dest="data_path", type=str, default=None,
                        help="Caminho para 'Real estate valuation data set.xlsx'.")
    parser.add_argument("--outdir", type=str, default="./outputs_tc1")
    parser.add_argument("--test_size", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iqr_k", type=float, default=1.5, help="Fator k do IQR (p/ outliers no treino).")
    args = parser.parse_args()

    default_guess = os.path.join(os.getcwd(), "Real estate valuation data set.xlsx")
    data_path = args.data_path or default_guess
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"Não encontrei o arquivo: {data_path}")

    # Dados
    df = pd.read_excel(data_path)
    df.columns = [c.strip() for c in df.columns]
    df = df.drop(columns=[c for c in df.columns if c.lower() in ["no", "id"]], errors="ignore")
    y_col = detect_target_column(df)
    X = df.drop(columns=[y_col]); y = df[y_col].astype(float)

    # Split único
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.seed)

    # --------- CENÁRIO 1: Baseline ---------
    models = build_models(n_features=X_train.shape[1])
    metrics_base, preds_tr_b, preds_te_b, resid_tr_b = evaluate_models(models, X_train, y_train, X_test, y_test, "Baseline")
    metrics_base = pd.concat([metrics_base, pd.DataFrame([RM_ART_ROW, NN_ART_ROW])], ignore_index=True)
    cv_base = crossval_top4(models, metrics_base[~metrics_base["Model"].isin(["RM_Art","NN_Art"])], X_train, y_train)
    own_base = metrics_base[~metrics_base["Model"].isin(["RM_Art","NN_Art"])]
    best_base_model = own_base.sort_values(by="RMSE_test").iloc[0]["Model"]
    champion_base = _choose_champion(own_base)

    base = {"metrics_df":metrics_base, "cv_df":cv_base, "preds_train":preds_tr_b, "preds_test":preds_te_b,
            "residuals_train":resid_tr_b, "y_train":y_train, "y_test":y_test}

    # --------- CENÁRIO 2: TrainNoOutliers ---------
    mask = iqr_mask(y_train.values, k=args.iqr_k)
    Xtr_no, ytr_no = X_train[mask], y_train[mask]
    models_no = build_models(n_features=Xtr_no.shape[1])
    metrics_no, preds_tr_n, preds_te_n, resid_tr_n = evaluate_models(models_no, Xtr_no, ytr_no, X_test, y_test, "TrainNoOutliers")
    rm_art_no = RM_ART_ROW.copy(); rm_art_no["Scenario"]="TrainNoOutliers"
    nn_art_no = NN_ART_ROW.copy(); nn_art_no["Scenario"]="TrainNoOutliers"
    metrics_no = pd.concat([metrics_no, pd.DataFrame([rm_art_no, nn_art_no])], ignore_index=True)
    cv_no = crossval_top4(models_no, metrics_no[~metrics_no["Model"].isin(["RM_Art","NN_Art"])], Xtr_no, ytr_no)
    own_no = metrics_no[~metrics_no["Model"].isin(["RM_Art","NN_Art"])]
    best_no_model = own_no.sort_values(by="RMSE_test").iloc[0]["Model"]
    champion_noout = _choose_champion(own_no)

    noout = {"metrics_df":metrics_no, "cv_df":cv_no, "preds_train":preds_tr_n, "preds_test":preds_te_n,
             "residuals_train":resid_tr_n, "y_train":ytr_no, "y_test":y_test}

    # --------- Salvar tudo ---------
    save_all(args.outdir, df, base, noout, best_base_model, best_no_model,
             champion_base, champion_noout, y_col, args.iqr_k)

    # --------- Console ---------
    print_console(args.outdir, base, noout, champion_base, champion_noout, y_train=y_train, iqr_k=args.iqr_k)

    print("\nArquivos gerados em:", os.path.abspath(args.outdir))
    print(" - Excel: regression_results_TC1.xlsx (abas: eda_*, raw_data, metrics_*, cv_summary_*, champion_summary, *_pred_*, residuals_diag_*)")
    print(" - Plots EDA: outputs_tc1/eda/ (boxplot_y_full.png, hist_y_full.png, boxplots_features.png, corr_matrix.png)")
    print(" - Plots Modelos: outputs_tc1/plots/ (residuals_*_BASE/NOOUT [+ _qq], scatter_*_BASE/NOOUT)")
    print(f"Melhor por RMSE (Baseline): {best_base_model}  |  Campeão (Baseline): {champion_base['Model']}")
    print(f"Melhor por RMSE (NoOut):   {best_no_model}     |  Campeão (NoOut):   {champion_noout['Model']}")

if __name__ == "__main__":
    main()
