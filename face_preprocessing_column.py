#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
faces_pipeline_tc2.py  (versão SEM algoritmos de árvore/boosting)

Cobre as 8 ATIVIDADES do PDF (TC2_PPGETI_2025.1) + OBS-2, com resultados
organizados por ATIVIDADE e por CLASSIFICADOR:

- Classificadores:
  * MQ = Mínimos Quadrados (one-vs-rest)
  * PL = Perceptron Logístico (LogisticRegression)
  * k-NN, LDA, QDA, SVM
  * QDA manual (implementação própria)  << NOVO
  * MLP-1H / MLP-2H (64 e 128 neurônios; relu/tanh; solver adam; early_stopping)
  * MLP manual (1H) / MLP manual (2H) — implementação própria em NumPy (AdamW, CE)

- Métricas (teste): acc, precision, recall, F1 macro, AUC ROC macro OvR
- Custo computacional: fit/predict/total e ms por amostra
- Gráficos:
  * Barras: melhor por classificador (valor sobre as barras)
  * Curvas (vencedores): acc_treino, acc_teste, perda_treino (log-loss)
  * (MLP) Curva interna de loss (se disponível)
  * PCA: variância explicada e acumulada (A3 e A5)
  * Matriz de confusão (vencedores): CSV + heatmap com anotações

- Questão 8 (controle de acesso): agora roda TODOS os classificadores e imprime:
  "acurácia"; "taxa de falsos negativos (proporção de pessoas às quais acesso foi permitido incorretamente)";
  "taxa de falsos positivos (pessoas às quais acesso não foi permitido incorretamente)"; "sensibilidade"; "precisão".
  Valores médios e desvios-padrão para 50 rodadas.

Requisitos: numpy, pandas, pillow, scipy, scikit-learn, matplotlib, tqdm
"""

from pathlib import Path
import re
import time
import warnings
from typing import Tuple, List, Dict, Any, Optional

import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    confusion_matrix, log_loss
)
from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm
import matplotlib.pyplot as plt

# ====== Suprimir avisos e manter saída limpa ======
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore")

# ====== Parâmetros ======
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "faces"          # pasta com arquivos "subject*"
INTRUDER_DIR = ROOT / "intruder"   # (opcional) imagens do intruso
OUT_DIR  = ROOT / "out_faces_tc2"

SIZE = 20           # redimensiona para SIZE x SIZE
RUNS = 50           # número de repetições por atividade
TRAIN_RATIO = 0.8   # proporção de treino
VAR_TGT = 98.0      # alvo de variância explicada acumulada (Atividade 5)

# ====== Normalizações (OBS-2) ======
NORM_OPTIONS = {
    "none": lambda: None,
    "zscore": lambda: StandardScaler(),
    "minmax01": lambda: MinMaxScaler(feature_range=(0.0, 1.0)),
    "minmax-11": lambda: MinMaxScaler(feature_range=(-1.0, 1.0)),
}

# ====== Grids ======
# MLPs (adam-only; tamanhos 64/128; relu/tanh)
# MLP1_GRID = [
#     dict(hidden_layer_sizes=(64,),  activation='relu',  solver='adam',
#          early_stopping=True, n_iter_no_change=20, max_iter=3000),
#     dict(hidden_layer_sizes=(64,),  activation='tanh',  solver='adam',
#          early_stopping=True, n_iter_no_change=20, max_iter=3000),
#     dict(hidden_layer_sizes=(128,), activation='relu',  solver='adam',
#          early_stopping=True, n_iter_no_change=20, max_iter=3000),
#     dict(hidden_layer_sizes=(128,), activation='tanh',  solver='adam',
#          early_stopping=True, n_iter_no_change=20, max_iter=3000),
# ]
# MLP2_GRID = [
#     dict(hidden_layer_sizes=(64, 64),   activation='relu', solver='adam',
#          early_stopping=True, n_iter_no_change=20, max_iter=3000),
#     dict(hidden_layer_sizes=(64, 64),   activation='tanh', solver='adam',
#          early_stopping=True, n_iter_no_change=20, max_iter=3000),
#     dict(hidden_layer_sizes=(128, 128), activation='relu',  solver='adam',
#          early_stopping=True, n_iter_no_change=20, max_iter=3000),
#     dict(hidden_layer_sizes=(128, 128), activation='tanh',  solver='adam',
#          early_stopping=True, n_iter_no_change=20, max_iter=3000),
#     dict(hidden_layer_sizes=(128, 64),  activation='relu',  solver='adam',
#          early_stopping=True, n_iter_no_change=20, max_iter=3000),
#     dict(hidden_layer_sizes=(128, 64),  activation='tanh',  solver='adam',
#          early_stopping=True, n_iter_no_change=20, max_iter=3000),
# ]
# ====== Grids ======m
# MLPs (lbfgs; tamanhos 64/128; relu/tanh)
MLP1_GRID = [
    dict(hidden_layer_sizes=(64,),  activation='relu',  solver='lbfgs', max_iter=3000),
    dict(hidden_layer_sizes=(64,),  activation='tanh',  solver='lbfgs', max_iter=3000),
    dict(hidden_layer_sizes=(128,), activation='relu',  solver='lbfgs', max_iter=3000),
    dict(hidden_layer_sizes=(128,), activation='tanh',  solver='lbfgs', max_iter=3000),
]
MLP2_GRID = [
    dict(hidden_layer_sizes=(64, 64),   activation='relu', solver='lbfgs', max_iter=3000),
    dict(hidden_layer_sizes=(64, 64),   activation='tanh', solver='lbfgs', max_iter=3000),
    dict(hidden_layer_sizes=(128, 128), activation='relu', solver='lbfgs', max_iter=3000),
    dict(hidden_layer_sizes=(128, 128), activation='tanh', solver='lbfgs', max_iter=3000),
    dict(hidden_layer_sizes=(128, 64),  activation='relu', solver='lbfgs', max_iter=3000),
    dict(hidden_layer_sizes=(128, 64),  activation='tanh', solver='lbfgs', max_iter=3000),
]


# Clássicos
KNN_GRID = [
    dict(n_neighbors=1, metric='euclidean', weights='uniform'),
    dict(n_neighbors=3, metric='euclidean', weights='uniform'),
    dict(n_neighbors=1, metric='manhattan', weights='uniform'),
    dict(n_neighbors=3, metric='manhattan', weights='uniform'),
    dict(n_neighbors=1, metric='cosine',   weights='uniform', algorithm='brute'),
]

# >>> AJUSTE 3: Grid do LDA com variantes robustas <<<
LDA_GRID = [
    dict(solver='svd'),                     # padrão (agora em float64)
    dict(solver='lsqr', shrinkage='auto'),  # estável em alta dimensão
    dict(solver='eigen', shrinkage='auto'),
]

QDA_GRID = [dict(reg_param=0.0), dict(reg_param=1e-3), dict(reg_param=1e-2)]
SVM_GRID = [
    dict(kernel='linear', C=1.0, probability=True),
    dict(kernel='rbf',    C=1.0, gamma='scale', probability=True),
]

# ====== MQ: Mínimos Quadrados (one-vs-rest) ======
class MQLeastSquaresClassifier:
    """Multi-classe por Mínimos Quadrados (one-vs-rest) com softmax p/ predict_proba."""
    def __init__(self, alpha: float = 0.0):
        self.alpha = float(alpha)
        self.classes_ = None
        self.W_ = None
    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        C = self.classes_.size
        Y = np.zeros((X.shape[0], C), dtype=np.float64)
        for k, c in enumerate(self.classes_):
            Y[:, k] = (y == c).astype(np.float64)
        XT = X.T
        if self.alpha > 0.0:
            A = XT @ X
            A.flat[::A.shape[0]+1] += self.alpha
            self.W_ = np.linalg.solve(A, XT @ Y)
        else:
            self.W_ = np.linalg.pinv(X) @ Y
        return self
    def decision_function(self, X):
        X = np.asarray(X, dtype=np.float64)
        return X @ self.W_
    def predict(self, X):
        S = self.decision_function(X)
        idx = np.argmax(S, axis=1)
        return self.classes_[idx]
    def predict_proba(self, X):
        S = self.decision_function(X)
        S = S - S.max(axis=1, keepdims=True)
        E = np.exp(S)
        return E / E.sum(axis=1, keepdims=True)

# ====== QDA manual (NumPy) ======
class ManualQDAClassifier:
    """
    QDA manual com regularização espectral:
      - Para cada classe k: Σ_k = cov(X_k)  (amostral)
      - Decompõe Σ_k = V diag(λ) V^T, e aplica λ_reg = max(λ, reg_param + min_eig)
      - Usa Σ_k^{-1} = V diag(1/λ_reg) V^T e log|Σ_k| = sum(log λ_reg)
    """
    def __init__(self, reg_param: float = 1e-3, min_eig: float = 1e-8):
        self.reg_param = float(reg_param)
        self.min_eig   = float(min_eig)
        self.classes_ = None
        self.means_ = None        # [C, D]
        self.inv_covs_ = None     # [C, D, D]
        self.logdets_ = None      # [C]
        self.priors_ = None       # [C]

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        C = self.classes_.size
        N, D = X.shape
        self.means_   = np.zeros((C, D), dtype=np.float64)
        self.inv_covs_= np.zeros((C, D, D), dtype=np.float64)
        self.logdets_ = np.zeros((C,), dtype=np.float64)
        self.priors_  = np.zeros((C,), dtype=np.float64)

        for k in range(C):
            Xk = X[y_idx == k]
            nk = Xk.shape[0]
            self.priors_[k] = nk / max(N, 1)
            mu = Xk.mean(axis=0)
            Xc = Xk - mu

            # cov amostral; se nk==1, resulta em zeros — tratamos via regularização espectral
            cov = (Xc.T @ Xc) / max(nk - 1, 1)

            # decomposição espectral + regularização de autovalores
            w, V = np.linalg.eigh(cov)             # w >= 0 (semidefinida)
            w_reg = np.maximum(w, self.reg_param + self.min_eig)

            # Σ^{-1} = V diag(1/w_reg) V^T, implementado de forma eficiente
            inv_cov = (V / w_reg) @ V.T            # divide cada coluna de V por w_reg

            self.means_[k]    = mu
            self.inv_covs_[k] = inv_cov
            self.logdets_[k]  = np.sum(np.log(w_reg))

        return self

    def _log_gauss(self, X):
        X = np.asarray(X, dtype=np.float64)
        N, D = X.shape
        C = self.classes_.size
        out = np.empty((N, C), dtype=np.float64)
        const = D * np.log(2*np.pi)
        for k in range(C):
            invC = self.inv_covs_[k]
            mu   = self.means_[k]
            dif  = X - mu
            # (x-μ)^T Σ^{-1} (x-μ)
            md2  = np.einsum('ni,ij,nj->n', dif, invC, dif, optimize=True)
            out[:, k] = -0.5 * (md2 + self.logdets_[k] + const)
        return out

    def decision_function(self, X):
        log_like = self._log_gauss(X)
        return log_like + np.log(self.priors_[None, :] + 1e-12)

    def predict_proba(self, X):
        S = self.decision_function(X)
        S = S - S.max(axis=1, keepdims=True)
        P = np.exp(S)
        P /= P.sum(axis=1, keepdims=True)
        return P

    def predict(self, X):
        S = self.decision_function(X)
        return self.classes_[np.argmax(S, axis=1)]

# ====== MLP manual (NumPy) — versão OTIMIZADA ======
class ManualMLPClassifier:
    """
    MLP manual otimizada (NumPy, float32) para multi-classe (softmax).
    - activation: 'relu' | 'tanh'
    - Otimizador: AdamW (decoupled weight decay)
    - Early stopping: validação holdout + avaliação a cada 'eval_every' épocas
    - Precisão: float32; prealocação de grads e momentos; batches grandes
    Exibe atributos compatíveis com sklearn: classes_, loss_curve_
    """
    def __init__(self,
                 hidden_layer_sizes=(64,),
                 activation='relu',
                 max_iter=500,
                 batch_size=256,
                 learning_rate=1e-3,
                 weight_decay=0.0,   # AdamW
                 grad_clip=None,     # ex.: 1.0
                 early_stopping=True,
                 n_iter_no_change=10,
                 tol=1e-4,           # melhoria mínima para resetar paciência
                 val_fraction=0.1,   # fração para validação do ES
                 eval_every=5,       # avalia perda/ES a cada k épocas
                 random_state=0):
        self.hidden_layer_sizes = tuple(hidden_layer_sizes)
        self.activation = activation
        self.max_iter = int(max_iter)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.grad_clip = None if grad_clip is None else float(grad_clip)
        self.early_stopping = bool(early_stopping)
        self.n_iter_no_change = int(n_iter_no_change)
        self.tol = float(tol)
        self.val_fraction = float(val_fraction)
        self.eval_every = int(eval_every)
        self.random_state = int(random_state)

        self.classes_ = None
        self.loss_curve_ = []
        self._params = {}
        self._adam_m = {}
        self._adam_v = {}
        self._adam_t = 0

    # ---------- utils ----------
    @staticmethod
    def _softmax(z):
        z = z - np.max(z, axis=1, keepdims=True)
        e = np.exp(z)
        return e / np.sum(e, axis=1, keepdims=True)

    def _act(self, z):
        if self.activation == 'relu':
            return np.maximum(0.0, z)
        return np.tanh(z)

    def _act_deriv(self, z, a):
        if self.activation == 'relu':
            return (z > 0).astype(np.float32)
        return 1.0 - a*a

    def _init_params(self, D, C):
        rng = np.random.default_rng(self.random_state)
        sizes = [D] + list(self.hidden_layer_sizes) + [C]
        self._params.clear(); self._adam_m.clear(); self._adam_v.clear()
        self._adam_t = 0

        def he(fan_in):          return np.sqrt(2.0 / max(1, fan_in))
        def xavier(f_in, f_out): return np.sqrt(6.0 / (f_in + f_out))

        for l in range(len(sizes)-1):
            f_in, f_out = sizes[l], sizes[l+1]
            if l < len(sizes)-2:
                if self.activation == 'relu':
                    W = rng.normal(0.0, he(f_in), size=(f_in, f_out))
                else:
                    bound = xavier(f_in, f_out)
                    W = rng.uniform(-bound, bound, size=(f_in, f_out))
            else:
                bound = xavier(f_in, f_out)
                W = rng.uniform(-bound, bound, size=(f_in, f_out))
            b = np.zeros((1, f_out))
            self._params[f"W{l+1}"] = np.ascontiguousarray(W, dtype=np.float32)
            self._params[f"b{l+1}"] = np.ascontiguousarray(b, dtype=np.float32)
            self._adam_m[f"W{l+1}"] = np.zeros_like(self._params[f"W{l+1}"])
            self._adam_m[f"b{l+1}"] = np.zeros_like(self._params[f"b{l+1}"])
            self._adam_v[f"W{l+1}"] = np.zeros_like(self._params[f"W{l+1}"])
            self._adam_v[f"b{l+1}"] = np.zeros_like(self._params[f"b{l+1}"])

        # buffers de grad prealocados
        self._grads = {}
        for k in self._params:
            self._grads[k] = np.zeros_like(self._params[k])

    def _forward(self, X):
        caches = []
        A = X
        Lh = len(self.hidden_layer_sizes)
        for l in range(1, Lh+1):
            W, b = self._params[f"W{l}"], self._params[f"b{l}"]
            Z = A @ W + b
            A = self._act(Z)
            caches.append((Z, A))
        Wk, bk = self._params[f"W{Lh+1}"], self._params[f"b{Lh+1}"]
        logits = A @ Wk + bk
        P = self._softmax(logits)
        return caches, logits, P

    @staticmethod
    def _cross_entropy(P, Y_onehot):
        eps = 1e-12
        return -np.sum(Y_onehot * np.log(P + eps)) / Y_onehot.shape[0]

    def _adamw_step(self, lr):
        b1, b2, eps = 0.9, 0.999, 1e-8
        self._adam_t += 1
        for k in self._params.keys():
            g = self._grads[k]
            # Adam moments
            self._adam_m[k] = b1*self._adam_m[k] + (1-b1)*g
            self._adam_v[k] = b2*self._adam_v[k] + (1-b2)*(g*g)
            mhat = self._adam_m[k] / (1 - b1**self._adam_t)
            vhat = self._adam_v[k] / (1 - b2**self._adam_t)
            # Decoupled weight decay (apenas pesos)
            if k.startswith('W') and self.weight_decay > 0.0:
                self._params[k] -= lr * self.weight_decay * self._params[k]
            # Update
            self._params[k] -= lr * mhat / (np.sqrt(vhat) + eps)

    def fit(self, X, y):
        X = np.ascontiguousarray(X, dtype=np.float32)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        C = self.classes_.size
        N, D = X.shape

        # one-hot
        class_to_idx = {c:i for i,c in enumerate(self.classes_)}
        Y = np.zeros((N, C), dtype=np.float32)
        for i, yi in enumerate(y):
            Y[i, class_to_idx[yi]] = 1.0

        # split treino/val para early stopping
        Nv = int(max(1, round(N * self.val_fraction))) if self.early_stopping else 0
        if Nv > 0 and Nv < N:
            Xv, Yv = X[:Nv], Y[:Nv]
            Xtr_es, Ytr_es = X[Nv:], Y[Nv:]
        else:
            Xv = Yv = None
            Xtr_es, Ytr_es = X, Y
        Ntr = Xtr_es.shape[0]

        self._init_params(D, C)
        self.loss_curve_ = []
        best = np.inf
        no_change = 0

        rng = np.random.default_rng(self.random_state)
        idx_all = np.arange(Ntr)
        Lh = len(self.hidden_layer_sizes)

        for epoch in range(self.max_iter):
            rng.shuffle(idx_all)
            # mini-batches
            for start in range(0, Ntr, self.batch_size):
                end = min(start + self.batch_size, Ntr)
                mb = idx_all[start:end]
                Xb = np.ascontiguousarray(Xtr_es[mb], dtype=np.float32)
                Yb = Ytr_es[mb]

                # forward
                caches, logits, P = self._forward(Xb)

                # backward (softmax+CE)
                dlogits = (P - Yb) / Xb.shape[0]
                Alast = caches[-1][1] if Lh > 0 else Xb

                # grads saída
                self._grads[f"W{Lh+1}"][...] = Alast.T @ dlogits
                self._grads[f"b{Lh+1}"][...] = np.sum(dlogits, axis=0, keepdims=True)

                # retroprop nas ocultas
                dA = dlogits @ self._params[f"W{Lh+1}"].T
                for l in range(Lh, 0, -1):
                    Zl, Al = caches[l-1]
                    Aprev = caches[l-2][1] if l-2 >= 0 else Xb
                    dZ = dA * self._act_deriv(Zl, Al)
                    self._grads[f"W{l}"][...] = Aprev.T @ dZ
                    self._grads[f"b{l}"][...] = np.sum(dZ, axis=0, keepdims=True)
                    if l > 1:
                        dA = dZ @ self._params[f"W{l}"].T

                # clipping opcional
                if self.grad_clip is not None:
                    total_norm = 0.0
                    for k in range(1, Lh+2):
                        g = self._grads[f"W{k}"]
                        total_norm += float(np.sum(g*g))
                    total_norm = np.sqrt(total_norm)
                    if total_norm > self.grad_clip:
                        scale = self.grad_clip / (total_norm + 1e-12)
                        for k in range(1, Lh+2):
                            self._grads[f"W{k}"][...] *= scale
                            self._grads[f"b{k}"][...] *= scale

                # AdamW update
                self._adamw_step(self.learning_rate)

            # avaliação periódica p/ ES
            if (not self.early_stopping) or (epoch % self.eval_every != 0):
                continue

            if Xv is not None:
                _, _, Pv = self._forward(Xv)
                loss_val = self._cross_entropy(Pv, Yv)
            else:
                _, _, P_all = self._forward(Xtr_es)
                loss_val = self._cross_entropy(P_all, Ytr_es)

            self.loss_curve_.append(float(loss_val))

            if loss_val < best - self.tol:
                best = loss_val
                no_change = 0
            else:
                no_change += 1
                if no_change >= self.n_iter_no_change:
                    break

        return self

    def predict_proba(self, X):
        X = np.ascontiguousarray(X, dtype=np.float32)
        _, _, P = self._forward(X)
        return P

    def decision_function(self, X):
        X = np.ascontiguousarray(X, dtype=np.float32)
        _, logits, _ = self._forward(X)
        return logits

    def predict(self, X):
        P = self.predict_proba(X)
        return self.classes_[np.argmax(P, axis=1)]

# ====== IO de imagens ======
def list_subject_files(data_dir: Path) -> List[Path]:
    files = []
    for p in sorted(data_dir.glob("subject*")):
        if p.is_file():
            files.append(p)
    if not files:
        raise FileNotFoundError(f"Nenhum arquivo 'subject*' encontrado em {data_dir}")
    return files

def detect_format(path: Path) -> str:
    with open(path, 'rb') as f:
        sig = f.read(8)
    if len(sig) >= 6 and (sig[:6] == b'GIF87a' or sig[:6] == b'GIF89a'):
        return 'gif'
    if len(sig) >= 8 and sig[:8] == bytes([137,80,78,71,13,10,26,10]):
        return 'png'
    if len(sig) >= 2 and sig[0] == 0xFF and sig[1] == 0xD8:
        return 'jpg'
    if len(sig) >= 3 and ((sig[0:2] == b'II' and sig[2] == 42) or (sig[0:2] == b'MM' and sig[2] == 42)):
        return 'tiff'
    if len(sig) >= 2 and sig[0:1] == b'P' and sig[1:2] in (b'2', b'5'):
        return 'pgm'
    return ''

def parse_subject_id(stem: str) -> int:
    m = re.match(r"^subject\s*0*([0-9]+)", stem)
    if not m:
        raise ValueError(f"Nome inesperado: {stem}")
    return int(m.group(1))

def load_face_vector(path: Path, size: int) -> np.ndarray:
    fmt = detect_format(path)
    im = Image.open(path)
    if fmt != 'pgm':
        im = im.convert("L")
    im = im.resize((size, size), Image.BILINEAR)
    arr = np.asarray(im, dtype=np.float32)
    vec = arr.reshape(-1, order='F')  # empilha por colunas (estilo MATLAB)
    return vec

def build_xy(data_dir: Path, size: int) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    files = list_subject_files(data_dir)
    X_list, y_list, names = [], [], []
    for p in files:
        sid = parse_subject_id(p.name)
        vec = load_face_vector(p, size=size)
        X_list.append(vec); y_list.append(sid); names.append(p.name)
    X = np.vstack(X_list).astype(np.float32)  # N x D
    y = np.asarray(y_list, dtype=np.int32)
    return X, y, names

# ====== PCA ======
def pca_full_rotation(X: np.ndarray):
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    N, D = X.shape
    if D > N:
        S = (Xc @ Xc.T) / max(N-1, 1)
        eigvals_small, U = np.linalg.eigh(S)
        idx = np.argsort(eigvals_small)[::-1]
        eigvals_small = eigvals_small[idx]; U = U[:, idx]
        eps = 1e-12
        V = (Xc.T @ U) / np.sqrt(np.maximum(eigvals_small[None, :], eps))
        V = V / (np.linalg.norm(V, axis=0, keepdims=True) + 1e-12)
        eigvals = eigvals_small
    else:
        C = (Xc.T @ Xc) / max(N-1, 1)
        eigvals, V = np.linalg.eigh(C)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]; V = V[:, idx]
    Xrot = (Xc @ V)  # N x D
    return Xrot.astype(np.float32), eigvals.astype(np.float32), V.astype(np.float32), mu.astype(np.float32)

def choose_q_for_variance(eigvals: np.ndarray, var_tgt: float) -> int:
    ve = 100.0 * eigvals / max(float(eigvals.sum()), 1e-12)
    vea = np.cumsum(ve)
    q = int(np.searchsorted(vea, var_tgt) + 1)
    q = max(1, min(q, len(eigvals)))
    return q

def plot_pca_variance(eigvals: np.ndarray, q: int, out_png: Path, title: str):
    ve = 100.0 * eigvals / max(float(eigvals.sum()), 1e-12)
    vea = np.cumsum(ve)
    x = np.arange(1, len(ve)+1)
    plt.figure(figsize=(10, 5))
    l1, = plt.plot(x, ve, label='Variância explicada (%)', linewidth=2)
    l2, = plt.plot(x, vea, label='Variância acumulada (%)', linewidth=2, linestyle='--')
    plt.axvline(q, color='black', linestyle=':', linewidth=1.5)
    plt.text(q+0.5, min(100.0, max(vea.min(), 0)+5), f"q={q}", fontsize=10)
    plt.ylim(0, 100)
    plt.xlabel('Componente principal')
    plt.ylabel('Percentual (%)')
    plt.title(title)
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0))
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()

# ====== Box-Cox + z-score ======
def fit_boxcox_zscore(X: np.ndarray):
    X = np.asarray(X, dtype=np.float64)
    D = X.shape[1]
    shift = np.zeros(D, dtype=np.float64)
    lambdas = np.zeros(D, dtype=np.float64)
    mu = np.zeros(D, dtype=np.float64)
    std = np.zeros(D, dtype=np.float64)
    for j in range(D):
        x = X[:, j].copy()
        finite_mask = np.isfinite(x)
        if not finite_mask.all():
            med = np.median(x[finite_mask]) if finite_mask.any() else 0.0
            x[~finite_mask] = med
        xmin = np.min(x) if np.isfinite(np.min(x)) else 0.0
        add = 0.0
        if xmin <= 0:
            add = -xmin + 1e-6
            x = x + add
        min_pos = np.min(x)
        if min_pos <= 0:
            x = x + (1e-6 - min_pos)
        try:
            x_bc, lam = stats.boxcox(x)
        except Exception:
            lam = 0.0
            x_bc = np.log(x)
        m = float(np.mean(x_bc))
        s = float(np.std(x_bc) + 1e-12)
        shift[j] = add; lambdas[j] = lam; mu[j] = m; std[j] = s
    return {'shift': shift.astype(np.float32), 'lambda': lambdas.astype(np.float32),
            'mean':   mu.astype(np.float32),    'std':    std.astype(np.float32)}

def apply_boxcox_zscore(X: np.ndarray, params):
    X = np.asarray(X, dtype=np.float64)
    shift = params['shift'].astype(np.float64)
    lambdas = params['lambda'].astype(np.float64)
    mu = params['mean'].astype(np.float64)
    std = params['std'].astype(np.float64)
    N, D = X.shape
    X_t = np.empty_like(X, dtype=np.float64)
    for j in range(D):
        x = X[:, j].copy()
        finite_mask = np.isfinite(x)
        if not finite_mask.all():
            med = np.median(x[finite_mask]) if finite_mask.any() else 0.0
            x[~finite_mask] = med
        x = x + shift[j]
        min_pos = np.min(x)
        if min_pos <= 0:
            x = x + (1e-6 - min_pos)
        lam = lambdas[j]
        if abs(lam) < 1e-8:
            x_bc = np.log(x)
        else:
            x_bc = (np.power(x, lam) - 1.0) / lam
        X_t[:, j] = (x_bc - mu[j]) / std[j]
    return X_t.astype(np.float32)

# ====== Makers ======
def make_MQ():   return MQLeastSquaresClassifier(alpha=1e-6)
def make_PL():   return LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='auto')
def make_MLP1(c):return MLPClassifier(random_state=0, **c)
def make_MLP2(c):return MLPClassifier(random_state=0, **c)
def make_KNN(c): return KNeighborsClassifier(**c)
def make_LDA(c): return LDA(**c)
def make_QDA(c): return QDA(**c)
def make_SVM(c): return SVC(**c)

# ====== Makers MLP manual ======
def make_MMLP1(c): return ManualMLPClassifier(random_state=0, **c)
def make_MMLP2(c): return ManualMLPClassifier(random_state=0, **c)

# ====== Maker QDA manual ======
def make_QDA_MANUAL(c): return ManualQDAClassifier(**c)

# ====== Grids MLP manual (otimizados) ======
MANUAL_MLP1_GRID = [
    dict(hidden_layer_sizes=(128,), activation='relu',
         max_iter=500, early_stopping=True, n_iter_no_change=10, tol=1e-4,
         learning_rate=1e-3, weight_decay=1e-4, batch_size=512, eval_every=5, grad_clip=None, val_fraction=0.1),
    dict(hidden_layer_sizes=(64,), activation='tanh',
         max_iter=500, early_stopping=True, n_iter_no_change=10, tol=1e-4,
         learning_rate=5e-4, weight_decay=1e-4, batch_size=512, eval_every=5, grad_clip=None, val_fraction=0.1),
]
MANUAL_MLP2_GRID = [
    dict(hidden_layer_sizes=(128, 128), activation='relu',
         max_iter=500, early_stopping=True, n_iter_no_change=10, tol=1e-4,
         learning_rate=1e-3, weight_decay=1e-4, batch_size=512, eval_every=5, grad_clip=None, val_fraction=0.1),
    dict(hidden_layer_sizes=(128, 64),  activation='relu',
         max_iter=500, early_stopping=True, n_iter_no_change=10, tol=1e-4,
         learning_rate=1e-3, weight_decay=1e-4, batch_size=512, eval_every=5, grad_clip=None, val_fraction=0.1),
    dict(hidden_layer_sizes=(64, 64),   activation='tanh',
         max_iter=500, early_stopping=True, n_iter_no_change=10, tol=1e-4,
         learning_rate=5e-4, weight_decay=1e-4, batch_size=512, eval_every=5, grad_clip=None, val_fraction=0.1),
]

# ====== Grid QDA manual ======
QDA_MANUAL_GRID = [dict(reg_param=0.0), dict(reg_param=1e-3), dict(reg_param=1e-2)]

# ====== Helpers prob/scores ======
def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)

def _get_scores(clf, X) -> Optional[np.ndarray]:
    if hasattr(clf, "predict_proba"):
        try:
            return clf.predict_proba(X)
        except Exception:
            pass
    if hasattr(clf, "decision_function"):
        try:
            s = clf.decision_function(X)
            return s if s.ndim > 1 else np.vstack([1 - s, s]).T
        except Exception:
            pass
    return None

def _get_proba(clf, X) -> Optional[np.ndarray]:
    if hasattr(clf, "predict_proba"):
        try:
            p = clf.predict_proba(X)
            if np.all(np.isfinite(p)):
                return p
        except Exception:
            pass
    if hasattr(clf, "decision_function"):
        try:
            s = clf.decision_function(X)
            if s.ndim == 1:
                s = np.vstack([0*s, s]).T
            p = _softmax(s)
            return p
        except Exception:
            pass
    return None

# >>> AJUSTE 1: helper para upcast/limpeza de finitos (usado no LDA) <<<
def _as64_finite(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64, order='C')
    if not np.isfinite(X).all():
        # substitui NaN/Inf pela mediana da coluna
        X_nan = np.where(np.isfinite(X), X, np.nan)
        col_med = np.nanmedian(X_nan, axis=0)
        bad = ~np.isfinite(X)
        X[bad] = np.take(col_med, np.where(bad)[1])
    return X

# ====== Tabelas bonitas no console ======
def _print_variant_table(df_grid: pd.DataFrame, cls_name: str, tag: str, topn: int = 8):
    """Mostra no console (e salva CSV) as variações testadas do classificador (OBS-2)."""
    if df_grid is None or df_grid.empty:
        print(f"   (sem resultados para {cls_name} em {tag})")
        return
    keep = [
        'normalizacao','activation','variante',
        'acc_media','acc_min','acc_max','acc_mediana','acc_std',
        'tempo_total_medio_s'
    ]
    d = df_grid[keep].copy().sort_values('acc_media', ascending=False)
    out_csv = OUT_DIR / tag / "classifiers" / cls_name / "variants_table.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(out_csv, index=False)

    dtop = d.head(topn)
    print(f"\n{tag} — {cls_name} — variações (top-{len(dtop)})")
    print("-"*114)
    print(f"{'Norm':10s} {'Act':8s} {'Variante':28s} "
          f"{'Média%':>8s} {'Mín%':>8s} {'Máx%':>8s} {'Mediana%':>9s} {'DP%':>7s} {'Tempo(ms)':>11s}")
    print("-"*114)
    for _, r in dtop.iterrows():
        print(f"{(r['normalizacao'] or '-'):10s} {str(r.get('activation','-') or '-'):8s} {str(r['variante'])[:28]:28s} "
              f"{100*r['acc_media']:8.2f} {100*r['acc_min']:8.2f} {100*r['acc_max']:8.2f} "
              f"{100*r['acc_mediana']:9.2f} {100*r['acc_std']:7.2f} {1000*r['tempo_total_medio_s']:11.1f}")
    print("-"*114)
    print(f"   [CSV] Variações salvas em: {out_csv}")

def print_activity_table_pdf_style(tag: str, classifiers: List[str] = None):
    """Imprime/salva a tabela no estilo do PDF (uma linha por classificador, melhor versão)."""
    if classifiers is None:
        classifiers = ["MQ", "PL", "k-NN", "LDA", "QDA", "QDA manual", "SVM", "MLP-1H", "MLP-2H", "MLP manual (1H)", "MLP manual (2H)"]
    sum_path = OUT_DIR / tag / "summary_best.csv"
    if not sum_path.exists():
        print(f"[WARN] {sum_path} não encontrado para montar tabela do PDF ({tag}).")
        return
    df = pd.read_csv(sum_path)
    df = df[df['classificador'].isin(classifiers)].copy()
    if df.empty:
        print(f"[WARN] Tabela do PDF ({tag}) sem linhas.")
        return

    df = df[['classificador','acc_media','acc_min','acc_max','acc_mediana','acc_std','tempo_total_medio_s']]
    print(f"\n===== Tabela (estilo PDF) — {tag} =====")
    print(f"{'Classificador':20s} {'Média%':>8s} {'Mín%':>8s} {'Máx%':>8s} {'Mediana%':>9s} {'DP%':>7s} {'Tempo(ms)':>11s}")
    print("-"*92)
    for _, r in df.sort_values('classificador').iterrows():
        print(f"{r['classificador']:20s} "
              f"{100*r['acc_media']:8.2f} {100*r['acc_min']:8.2f} {100*r['acc_max']:8.2f} "
              f"{100*r['acc_mediana']:9.2f} {100*r['acc_std']:7.2f} {1000*r['tempo_total_medio_s']:11.1f}")
    print("-"*92)

    out_csv = OUT_DIR / tag / "tabela_pdf.csv"
    df_out = df.copy()
    df_out.rename(columns={
        'classificador':'Classificador','acc_media':'Media','acc_min':'Minimo',
        'acc_max':'Maximo','acc_mediana':'Mediana','acc_std':'DesvioPadrao',
        'tempo_total_medio_s':'TempoExecucao_s'
    }, inplace=True)
    df_out.to_csv(out_csv, index=False)
    print(f"[OK] Tabela do PDF salva em: {out_csv}")

# ====== Impressão bonita (resumo melhores) ======
def pretty_print_best_table(tag: str, best_rows: List[Dict[str, Any]]):
    if not best_rows: return
    best_rows = sorted(best_rows, key=lambda r: r['acc_media'], reverse=True)
    print(f"\nResumo (melhores por classificador) — {tag}")
    print("-"*88)
    print(f"{'Classif.':20s} {'Acc%':>7s} {'F1%':>7s} {'AUC':>6s} {'Fit(ms)':>9s} {'Pred(ms)':>9s} {'Total(ms)':>10s}")
    print("-"*88)
    for r in best_rows:
        acc = r['acc_media']*100
        f1  = r['f1_macro']*100
        auc = r['auc_macro_ovr']
        print(f"{r['classificador']:20s} {acc:7.2f} {f1:7.2f} {auc:6.3f} "
              f"{r['tempo_fit_medio_s']*1000:9.1f} {r['tempo_pred_medio_s']*1000:9.1f} {r['tempo_total_medio_s']*1000:10.1f}")
    print("-"*88)

# ====== Avaliação (OBS-2) + salvamento organizado ======
def eval_classifier_with_variants(X: np.ndarray, y: np.ndarray, runs: int, train_ratio: float,
                                  cls_name: str, variants: List[Dict[str, Any]], tag: str):
    print(f"   >> {cls_name}: testando normalizações/variações...")

    if cls_name == "MQ":
        grid_variants = [dict()]; maker = lambda v: (lambda: make_MQ())
    elif cls_name == "PL":
        grid_variants = [dict()]; maker = lambda v: (lambda: make_PL())
    elif cls_name == "MLP-1H":
        grid_variants = variants; maker = lambda v: (lambda: make_MLP1(v))
    elif cls_name == "MLP-2H":
        grid_variants = variants; maker = lambda v: (lambda: make_MLP2(v))
    elif cls_name == "MLP manual (1H)":
        grid_variants = variants; maker = lambda v: (lambda: make_MMLP1(v))
    elif cls_name == "MLP manual (2H)":
        grid_variants = variants; maker = lambda v: (lambda: make_MMLP2(v))
    elif cls_name == "k-NN":
        grid_variants = KNN_GRID; maker = lambda v: (lambda: make_KNN(v))
    elif cls_name == "LDA":
        grid_variants = LDA_GRID; maker = lambda v: (lambda: make_LDA(v))
    elif cls_name == "QDA":
        grid_variants = QDA_GRID; maker = lambda v: (lambda: make_QDA(v))
    elif cls_name == "QDA manual":
        grid_variants = QDA_MANUAL_GRID; maker = lambda v: (lambda: make_QDA_MANUAL(v))
    elif cls_name == "SVM":
        grid_variants = SVM_GRID; maker = lambda v: (lambda: make_SVM(v))
    else:
        raise ValueError(cls_name)

    combos = [(norm_key, norm_maker, v) for norm_key, norm_maker in NORM_OPTIONS.items() for v in grid_variants]

    grid_rows = []
    best = None
    best_curves = None
    labels = np.unique(y)

    for norm_key, norm_maker, v in tqdm(combos, desc=f"{cls_name} | busca", leave=False):
        sss = StratifiedShuffleSplit(n_splits=runs, train_size=train_ratio, random_state=42)

        acc_tr, acc_te = [], []
        loss_tr = []
        precs, recs, f1s, aucs = [], [], [], []
        t_fit, t_pred, t_total = [], [], []
        pred_total_time = 0.0
        pred_total_samples = 0

        cm_sum = np.zeros((labels.size, labels.size), dtype=np.int64)
        last_internal_loss = None

        for tr, te in sss.split(X, y):
            Xtr, Xte = X[tr], X[te]
            ytr, yte = y[tr], y[te]

            scaler = norm_maker()
            if scaler is not None:
                Xtr = scaler.fit_transform(Xtr)
                Xte = scaler.transform(Xte)

            # >>> AJUSTE 2: upcast/limpeza apenas para LDA <<<
            if cls_name == "LDA":
                Xtr = _as64_finite(Xtr)
                Xte = _as64_finite(Xte)

            clf = maker(v)()

            t0 = time.time()
            clf.fit(Xtr, ytr)
            fit_time = time.time() - t0

            yhat_tr = clf.predict(Xtr)
            acc_tr.append(accuracy_score(ytr, yhat_tr))

            p_tr = _get_proba(clf, Xtr)
            if p_tr is not None and np.all(np.isfinite(p_tr)):
                try:
                    loss_tr.append(log_loss(ytr, p_tr, labels=np.unique(y)))
                except Exception:
                    loss_tr.append(np.nan)
            else:
                loss_tr.append(np.nan)

            t1 = time.time()
            yhat = clf.predict(Xte)
            pred_time = time.time() - t1
            total_time = fit_time + pred_time

            acc_te.append(accuracy_score(yte, yhat))
            pr, rc, f1, _ = precision_recall_fscore_support(yte, yhat, average='macro', zero_division=0)
            precs.append(pr); recs.append(rc); f1s.append(f1)

            scores = _get_scores(clf, Xte)
            if scores is not None and np.all(np.isfinite(scores)):
                try:
                    auc = roc_auc_score(yte, scores, multi_class='ovr', average='macro')
                except Exception:
                    auc = np.nan
            else:
                auc = np.nan
            aucs.append(auc)

            t_fit.append(fit_time)
            t_pred.append(pred_time)
            t_total.append(total_time)
            pred_total_time += pred_time
            pred_total_samples += len(yte)

            cm = confusion_matrix(yte, yhat, labels=labels)
            cm_sum += cm

            if hasattr(clf, "loss_curve_"):
                last_internal_loss = list(getattr(clf, "loss_curve_", []))

        acc_tr = np.array(acc_tr, dtype=np.float64)
        acc_te = np.array(acc_te, dtype=np.float64)
        loss_tr = np.array(loss_tr, dtype=np.float64)
        precs = np.array(precs, dtype=np.float64)
        recs  = np.array(recs,  dtype=np.float64)
        f1s   = np.array(f1s,   dtype=np.float64)
        aucs  = np.array(aucs,  dtype=np.float64)
        t_fit = np.array(t_fit, dtype=np.float64)
        t_pred= np.array(t_pred,dtype=np.float64)
        t_total=np.array(t_total,dtype=np.float64)

        pred_ms_per_sample = (pred_total_time / max(pred_total_samples, 1)) * 1000.0
        act = v.get('activation', '-') if isinstance(v, dict) else '-'

        nome_cls = cls_name

        row = {
            'tag': tag, 'classificador': nome_cls,
            'normalizacao': norm_key,
            'activation': act,
            'variante': str(v) if v else '-',
            # métricas (teste)
            'acc_media': acc_te.mean(), 'acc_min': acc_te.min(), 'acc_max': acc_te.max(),
            'acc_mediana': np.median(acc_te), 'acc_std': acc_te.std(ddof=1) if len(acc_te)>1 else 0.0,
            'prec_macro': precs.mean(), 'rec_macro': recs.mean(),
            'f1_macro': f1s.mean(), 'auc_macro_ovr': np.nanmean(aucs),
            # custo
            'tempo_fit_medio_s': t_fit.mean(),
            'tempo_pred_medio_s': t_pred.mean(),
            'tempo_total_medio_s': t_total.mean(),
            'pred_ms_por_amostra': pred_ms_per_sample,
            # curvas por rodada
            '_curve_acc_train': acc_tr.tolist(),
            '_curve_acc_test': acc_te.tolist(),
            '_curve_loss_train': loss_tr.tolist(),
            'loss_train_media': np.nanmean(loss_tr),
            # CM agregada
            '_cm_sum': cm_sum,
            '_cm_labels': labels,
        }
        grid_rows.append(row)

        if (best is None) or (row['acc_media'] > best['acc_media']):
            best = row
            best_curves = {
                'classificador': nome_cls,
                'tag': tag,
                'acc_train_curve': acc_tr.tolist(),
                'acc_test_curve': acc_te.tolist(),
                'loss_train_curve': loss_tr.tolist(),
                'mlp_internal_loss_curve': last_internal_loss,
                'cm_sum': cm_sum,
                'cm_labels': labels,
            }

    df_grid = pd.DataFrame(grid_rows).sort_values('acc_media', ascending=False)
    _print_variant_table(df_grid, cls_name, tag, topn=8)
    return best, df_grid, best_curves

# ====== Gráficos ======
def _annotate_bars(ax):
    for p in ax.patches:
        value = p.get_height()
        ax.annotate(f"{value:.1f}%", (p.get_x() + p.get_width()/2, value),
                    ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')

def plot_best_only(best_rows: List[Dict[str, Any]], out_path: Path, title: str):
    if not best_rows: return
    dfb = pd.DataFrame(best_rows).sort_values('acc_media', ascending=False)
    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    ax.bar(dfb['classificador'], dfb['acc_media'] * 100.0)
    _annotate_bars(ax)
    plt.ylabel('Acurácia (%)')
    plt.title(title)
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_acc_loss_curves(curves: Dict[str, Any], out_path: Path, title: str):
    if curves is None: return
    acc_tr = np.array(curves['acc_train_curve'], dtype=float)
    acc_te = np.array(curves['acc_test_curve'], dtype=float)
    loss_tr = np.array(curves['loss_train_curve'], dtype=float)
    x = np.arange(1, len(acc_tr)+1)
    fig, ax1 = plt.subplots(figsize=(10, 5))
    l1, = plt.plot(x, acc_tr * 100.0, label='Acurácia Treino', linewidth=2)
    l2, = plt.plot(x, acc_te * 100.0, label='Acurácia Teste', linewidth=2)
    ax1.set_xlabel('Rodada'); ax1.set_ylabel('Acurácia (%)')
    ax2 = ax1.twinx()
    l3, = ax2.plot(x, loss_tr, label='Perda (log-loss) Treino', linestyle='--')
    ax2.set_ylabel('Perda (log-loss)')
    lines = [l1, l2, l3]; labels = [ln.get_label() for ln in lines]
    fig.legend(lines, labels, loc='upper left', bbox_to_anchor=(1.02, 1.0))
    plt.title(title)
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150); plt.close()

def plot_mlp_internal_loss(curves: Dict[str, Any], out_path: Path, title: str):
    if curves is None: return
    lc = curves.get('mlp_internal_loss_curve', None)
    if lc is None: return
    y = np.array(lc, dtype=float); x = np.arange(1, len(y)+1)
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, label='Loss (interno, MLP)', linewidth=2)
    plt.xlabel('Iteração/Época'); plt.ylabel('Loss'); plt.title(title)
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0))
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150); plt.close()

def save_confusion_artifacts(cm: np.ndarray, labels: np.ndarray, out_csv: Path, out_png: Path, title: str):
    df = pd.DataFrame(cm, index=[f"true_{l}" for l in labels], columns=[f"pred_{l}" for l in labels])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=True)

    plt.figure(figsize=(8, 7))
    im = plt.imshow(cm, interpolation='nearest')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(title)
    plt.xlabel('Predito'); plt.ylabel('Verdadeiro')
    plt.xticks(ticks=np.arange(len(labels)), labels=labels, rotation=45, ha='right')
    plt.yticks(ticks=np.arange(len(labels)), labels=labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

# ====== Organização/salvamento por atividade e classificador ======
def save_classifier_outputs_for_activity(tag: str, cls_name: str,
                                         df_grid: pd.DataFrame,
                                         best: Optional[Dict[str, Any]],
                                         curves: Optional[Dict[str, Any]]):
    act_dir = OUT_DIR / tag
    clf_dir = act_dir / "classifiers" / cls_name
    clf_dir.mkdir(parents=True, exist_ok=True)

    if df_grid is not None and not df_grid.empty:
        df_grid.to_csv(clf_dir / "grid_all.csv", index=False)

    if best is not None:
        pd.DataFrame([best]).to_csv(clf_dir / "grid_best.csv", index=False)

    if curves is not None:
        plot_acc_loss_curves(curves, clf_dir / f"curvas_{cls_name}.png", title=f"{tag} — {cls_name} (vencedor)")
        if cls_name.startswith("MLP"):
            plot_mlp_internal_loss(curves, clf_dir / f"mlp_loss_{cls_name}.png",
                                   title=f"{tag} — Loss interno MLP — {cls_name}")

        cm = curves.get('cm_sum', None); labels = curves.get('cm_labels', None)
        if cm is not None and labels is not None:
            save_confusion_artifacts(
                cm, np.array(labels),
                clf_dir / f"cm_{cls_name}.csv",
                clf_dir / f"cm_{cls_name}.png",
                title=f"{tag} — Confusion Matrix — {cls_name}"
            )

# ====== Ordem dos classificadores ======
CLASSIFIER_ORDER = [
    ("MQ", []),
    ("PL", []),
    ("k-NN", KNN_GRID),
    ("LDA", LDA_GRID),
    ("QDA", QDA_GRID),
    ("QDA manual", QDA_MANUAL_GRID),     # << NOVO
    ("SVM", SVM_GRID),
    ("MLP-1H", MLP1_GRID),
    ("MLP-2H", MLP2_GRID),
    ("MLP manual (1H)", MANUAL_MLP1_GRID),  # novo otimizado
    ("MLP manual (2H)", MANUAL_MLP2_GRID),  # novo otimizado
]

def run_activity_eval_block(X: np.ndarray, y: np.ndarray, runs: int, train_ratio: float, tag: str):
    print(f">>> Avaliação ({tag}) — gerando pastas/CSVs/gráficos")
    act_dir = OUT_DIR / tag
    act_dir.mkdir(parents=True, exist_ok=True)

    best_rows = []
    for cls_name, grid in CLASSIFIER_ORDER:
        best, df_grid, curves = eval_classifier_with_variants(X, y, runs, train_ratio, cls_name, grid, tag)
        if best is not None:
            best_rows.append(best)
        save_classifier_outputs_for_activity(tag, cls_name, df_grid, best, curves)

    df_best = pd.DataFrame(best_rows).sort_values('acc_media', ascending=False)
    df_best.to_csv(act_dir / "summary_best.csv", index=False)

    plot_best_only(best_rows, act_dir / "best_by_classifier.png", title=f"{tag} — Melhor por classificador")

    pretty_print_best_table(tag, best_rows)
    return df_best

# ====== PCA helper para gráficos e arquivos ======
def pca_and_plots(X_raw: np.ndarray, tag_qD_dir: str, tag_q_base: str, var_tgt: float):
    X_rot, eigvals, V, mu = pca_full_rotation(X_raw)
    dir_full = OUT_DIR / tag_qD_dir
    dir_full.mkdir(parents=True, exist_ok=True)

    np.savetxt(dir_full / f"X_pca_full_{SIZE}x{SIZE}.txt", X_rot, fmt="%.6f")
    np.savetxt(dir_full / "eigvals.txt", eigvals, fmt="%.6f")

    ve = 100.0 * eigvals / max(float(eigvals.sum()), 1e-12)
    vea = np.cumsum(ve)
    pca_df = pd.DataFrame({
        "componente": np.arange(1, len(eigvals)+1),
        "autovalor": eigvals,
        "var_percent": ve,
        "var_acum_percent": vea
    })
    pca_df.to_csv(dir_full / "pca_stats_A3.csv", index=False)

    print("\n========================= ATIVIDADE 3 =========================")
    print("PCA sem redução (q = D): rotação/descorr. da covariância.")
    print(f"   X_rot: {X_rot.shape} | soma autovalores = {eigvals.sum():.4e}")
    plot_pca_variance(eigvals, q=X_rot.shape[1], out_png=dir_full / "PCA_variance_full.png",
                      title="PCA (q=D) — Variância explicada e acumulada")
    print(f"   [OK] Arquivos salvos em: {dir_full}")

    q = choose_q_for_variance(eigvals, var_tgt)
    X_pca_q = X_rot[:, :q]
    tag_q_dir = f"{tag_q_base}{q}"
    dir_q = OUT_DIR / tag_q_dir
    dir_q.mkdir(parents=True, exist_ok=True)

    np.savetxt(dir_q / f"X_pca_q{q}.txt", X_pca_q, fmt="%.6f")
    pca_df.to_csv(dir_q / f"pca_stats_A5_q{q}.csv", index=False)
    plot_pca_variance(eigvals, q=q, out_png=dir_q / f"PCA_variance_q{q}.png",
                      title=f"PCA — Variância explicada e acumulada (q={q})")

    var_acum_q = float(vea[q-1])
    print("\n========================= ATIVIDADE 5 =========================")
    print(f"Escolha de q para alcançar ≥ {var_tgt:.1f}% de variância:")
    print(f"   -> q = {q} componentes | variância acumulada em q = {var_acum_q:.2f}%")
    print(f"   [OK] Arquivos salvos em: {dir_q}")

    return X_rot, eigvals, V, mu, X_pca_q, q, tag_q_dir

# ====== Controle de acesso (Atividade 8) — versão original (LogReg) ======
def load_intruder_set(intruder_dir: Path, size: int):
    files = sorted([p for p in intruder_dir.iterdir() if p.is_file()])
    if len(files) == 0:
        raise FileNotFoundError(f"Nenhum arquivo encontrado em {intruder_dir}")
    X_list, names = [], []
    for p in files:
        vec = load_face_vector(p, size=size)
        X_list.append(vec); names.append(p.name)
    X = np.vstack(X_list).astype(np.float32)
    return X, names

def eval_intruder_binary(X_faces_bc: np.ndarray, y_faces: np.ndarray, X_intr_bc: np.ndarray,
                         runs: int, train_ratio: float, tag: str):
    print(f">>> CONTROLE DE ACESSO ({tag})")
    act_dir = OUT_DIR / tag; act_dir.mkdir(parents=True, exist_ok=True)

    intr_label = int(np.max(y_faces)) + 1
    y_intr = np.full((X_intr_bc.shape[0],), intr_label, dtype=np.int32)

    X_all = np.vstack([X_faces_bc, X_intr_bc])
    y_all = np.concatenate([y_faces, y_intr])
    y_bin = (y_all == intr_label).astype(int)

    sss = StratifiedShuffleSplit(n_splits=runs, train_size=train_ratio, random_state=123)
    accs, fnrs, fprs, recalls, precs = [], [], [], [], []

    for tr, te in tqdm(sss.split(X_all, y_bin), total=runs, desc=f"{tag} | CV binário", leave=False):
        Xtr, Xte = X_all[tr], X_all[te]
        ytr, yte = y_bin[tr], y_bin[te]

        clf = LogisticRegression(max_iter=500, solver='lbfgs')
        clf.fit(Xtr, ytr)
        ypred = clf.predict(Xte)

        acc = accuracy_score(yte, ypred)
        tn, fp, fn, tp = confusion_matrix(yte, ypred, labels=[0,1]).ravel()
        fnr = fn / (fn + tp + 1e-12)
        fpr = fp / (fp + tn + 1e-12)
        prec = (tp / (tp + fp + 1e-12))
        rec = (tp / (tp + fn + 1e-12))

        accs.append(acc); fnrs.append(fnr); fprs.append(fpr); precs.append(prec); recalls.append(rec)

    def agg(x):
        x = np.array(x, dtype=np.float64)
        return x.mean(), x.std(ddof=1) if len(x) > 1 else 0.0, x.min(), x.max(), np.median(x)

    row = {
        'tag': tag,
        'acc_media':   agg(accs)[0], 'acc_std':   agg(accs)[1], 'acc_min':   agg(accs)[2], 'acc_max':   agg(accs)[3], 'acc_mediana':   agg(accs)[4],
        'fnr_media':   agg(fnrs)[0], 'fnr_std':   agg(fnrs)[1], 'fnr_min':   agg(fnrs)[2], 'fnr_max':   agg(fnrs)[3], 'fnr_mediana':   agg(fnrs)[4],
        'fpr_media':   agg(fprs)[0], 'fpr_std':   agg(fprs)[1], 'fpr_min':   agg(fprs)[2], 'fpr_max':   agg(fprs)[3], 'fpr_mediana':   agg(fprs)[4],
        'rec_media':   agg(recalls)[0], 'rec_std': agg(recalls)[1], 'rec_min': agg(recalls)[2], 'rec_max': agg(recalls)[3], 'rec_mediana': agg(recalls)[4],
        'prec_media':  agg(precs)[0],  'prec_std': agg(precs)[1],  'prec_min': agg(precs)[2],  'prec_max': agg(precs)[3],  'prec_mediana': agg(precs)[4],
    }
    df = pd.DataFrame([row])
    df.to_csv(act_dir / "controle_acesso_summary.csv", index=False)
    print(f"   [OK] CSV salvo: {act_dir / 'controle_acesso_summary.csv'}")
    return df

# ====== Questão 2 — resumo específico (sem PCA) ======
def save_q2_summary_and_plot(tag: str):
    act_dir = OUT_DIR / tag
    sum_path = act_dir / "summary_best.csv"
    if not sum_path.exists():
        print(f"[WARN] Não encontrei {sum_path} para montar a Questão 2.")
        return
    df = pd.read_csv(sum_path)
    alvo = ["MQ","PL","k-NN","LDA","QDA","QDA manual","SVM","MLP-1H","MLP-2H","MLP manual (1H)","MLP manual (2H)"]
    df_q2 = df[df["classificador"].isin(alvo)].copy()
    if df_q2.empty:
        print("[WARN] Q2 sem linhas."); return
    df_q2 = df_q2.sort_values("acc_media", ascending=False)
    out_csv = act_dir / "questao2_summary.csv"
    df_q2.to_csv(out_csv, index=False)
    print(f"[OK] Q2: CSV salvo em {out_csv}")
    rows = df_q2.to_dict("records")
    out_png = act_dir / "questao2_barras.png"
    plot_best_only(rows, out_png, title=f"{tag} — Q2 (sem PCA) — melhores por classificador")
    print(f"[OK] Q2: gráfico salvo em {out_png}")

# ====== Questão 8 — TODOS os classificadores (INCLUSÃO) ======
def _agg_stats(x_list):
    x = np.asarray(x_list, dtype=np.float64)
    mean = x.mean()
    std  = x.std(ddof=1) if x.size > 1 else 0.0
    return mean, std, x.min(), x.max(), np.median(x)

def _binary_metrics_from_confusion(tn, fp, fn, tp):
    acc = (tp + tn) / max(tn + fp + fn + tp, 1)
    # Atenção: classe positiva = INTRUSO (1)
    fnr = fn / max(fn + tp, 1e-12)  # taxa de falsos negativos (acesso permitido incorretamente)
    fpr = fp / max(fp + tn, 1e-12)  # taxa de falsos positivos (acesso não foi permitido incorretamente)
    sens = tp / max(tp + fn, 1e-12) # sensibilidade (TPR)
    prec = tp / max(tp + fp, 1e-12) # precisão (PPV)
    return acc, fnr, fpr, sens, prec

def _maker_from_name_and_variant(cls_name: str, v: Dict[str, Any]):
    if cls_name == "MQ":
        return lambda: make_MQ()
    if cls_name == "PL":
        return lambda: make_PL()
    if cls_name == "k-NN":
        return lambda: make_KNN(v)
    if cls_name == "LDA":
        return lambda: make_LDA(v)
    if cls_name == "QDA":
        return lambda: make_QDA(v)
    if cls_name == "QDA manual":
        return lambda: make_QDA_MANUAL(v)
    if cls_name == "SVM":
        return lambda: make_SVM(v)
    if cls_name == "MLP-1H":
        return lambda: make_MLP1(v)
    if cls_name == "MLP-2H":
        return lambda: make_MLP2(v)
    if cls_name == "MLP manual (1H)":
        return lambda: make_MMLP1(v)
    if cls_name == "MLP manual (2H)":
        return lambda: make_MMLP2(v)
    raise ValueError(f"Classificador desconhecido: {cls_name}")

def _grid_for_classifier(cls_name: str):
    if cls_name == "MQ":  return [dict()]
    if cls_name == "PL":  return [dict()]
    if cls_name == "k-NN":return KNN_GRID
    if cls_name == "LDA": return LDA_GRID
    if cls_name == "QDA": return QDA_GRID
    if cls_name == "QDA manual": return QDA_MANUAL_GRID
    if cls_name == "SVM": return SVM_GRID
    if cls_name == "MLP-1H": return MLP1_GRID
    if cls_name == "MLP-2H": return MLP2_GRID
    if cls_name == "MLP manual (1H)": return MANUAL_MLP1_GRID
    if cls_name == "MLP manual (2H)": return MANUAL_MLP2_GRID
    raise ValueError(cls_name)

def eval_intruder_binary_one_classifier(X_faces_bc: np.ndarray, y_faces: np.ndarray,
                                        X_intr_bc: np.ndarray, runs: int, train_ratio: float, tag: str, cls_name: str):
    """Questão 8: avalia UM classificador em binário (intruso=1) com normalizações e variações."""
    print(f"   >> {cls_name}: Questão 8 (intruso) — normalizações/variações...")
    act_dir = OUT_DIR / tag
    clf_dir = act_dir / "classifiers" / cls_name
    clf_dir.mkdir(parents=True, exist_ok=True)

    intr_label = int(np.max(y_faces)) + 1
    y_intr = np.full((X_intr_bc.shape[0],), intr_label, dtype=np.int32)

    X_all = np.vstack([X_faces_bc, X_intr_bc])
    y_all = np.concatenate([y_faces, y_intr])
    y_bin = (y_all == intr_label).astype(int)  # 1 = intruso (POSITIVO)

    labels_bin = [0, 1]  # 0=autorizado, 1=intruso

    grid_variants = _grid_for_classifier(cls_name)
    combos = [(nk, nm, v) for nk, nm in NORM_OPTIONS.items() for v in grid_variants]

    rows = []
    best = None

    for norm_key, norm_maker, v in tqdm(combos, desc=f"{cls_name} | Q8 busca", leave=False):
        sss = StratifiedShuffleSplit(n_splits=runs, train_size=train_ratio, random_state=123)

        accs, fnrs, fprs, senss, precs = [], [], [], [], []
        cm_sum = np.zeros((2, 2), dtype=np.int64)
        pred_total_time, pred_total_samples = 0.0, 0
        fit_times, pred_times, total_times = [], [], []

        for tr, te in sss.split(X_all, y_bin):
            Xtr, Xte = X_all[tr], X_all[te]
            ytr, yte = y_bin[tr], y_bin[te]

            scaler = norm_maker()
            if scaler is not None:
                Xtr = scaler.fit_transform(Xtr)
                Xte = scaler.transform(Xte)

            # >>> AJUSTE 2 (também aqui): upcast/limpeza apenas para LDA <<<
            if cls_name == "LDA":
                Xtr = _as64_finite(Xtr)
                Xte = _as64_finite(Xte)

            maker = _maker_from_name_and_variant(cls_name, v)
            clf = maker()

            t0 = time.time()
            clf.fit(Xtr, ytr)
            fit_t = time.time() - t0

            t1 = time.time()
            ypred = clf.predict(Xte)
            pred_t = time.time() - t1
            total_t = fit_t + pred_t

            tn, fp, fn, tp = confusion_matrix(yte, ypred, labels=labels_bin).ravel()
            acc, fnr, fpr, sens, prec = _binary_metrics_from_confusion(tn, fp, fn, tp)

            accs.append(acc); fnrs.append(fnr); fprs.append(fpr); senss.append(sens); precs.append(prec)
            cm_sum += np.array([[tn, fp], [fn, tp]], dtype=np.int64)

            fit_times.append(fit_t); pred_times.append(pred_t); total_times.append(total_t)
            pred_total_time += pred_t; pred_total_samples += len(yte)

        def pack_stats(vals):
            mean, std, vmin, vmax, med = _agg_stats(vals)
            return dict(media=mean, std=std, minimo=vmin, maximo=vmax, mediana=med)

        row = {
            'tag': tag, 'classificador': cls_name,
            'normalizacao': norm_key, 'variante': str(v) if v else '-',
            # métricas pedidas (média + desvio padrão)
            'acuracia_media': pack_stats(accs)['media'], 'acuracia_std': pack_stats(accs)['std'],
            'fnr_media':      pack_stats(fnrs)['media'], 'fnr_std':      pack_stats(fnrs)['std'],
            'fpr_media':      pack_stats(fprs)['media'], 'fpr_std':      pack_stats(fprs)['std'],
            'sens_media':     pack_stats(senss)['media'], 'sens_std':     pack_stats(senss)['std'],
            'prec_media':     pack_stats(precs)['media'], 'prec_std':     pack_stats(precs)['std'],
            # tempos
            'tempo_fit_medio_s': np.mean(fit_times),
            'tempo_pred_medio_s': np.mean(pred_times),
            'tempo_total_medio_s': np.mean(total_times),
            'pred_ms_por_amostra': (pred_total_time / max(pred_total_samples, 1)) * 1000.0,
            # CM somada
            '_cm_sum': cm_sum,
            '_cm_labels': np.array(labels_bin),
        }
        rows.append(row)

        if (best is None) or (row['acuracia_media'] > best['acuracia_media']):
            best = row

    df = pd.DataFrame(rows).sort_values('acuracia_media', ascending=False)
    df.to_csv(clf_dir / "q8_binary_grid_all.csv", index=False)

    # salva melhor e a CM agregada correspondente
    if best is not None:
        pd.DataFrame([best]).to_csv(clf_dir / "q8_binary_best.csv", index=False)
        cm = best['_cm_sum']; labels = best['_cm_labels']
        save_confusion_artifacts(cm, labels, clf_dir / "q8_cm.csv", clf_dir / "q8_cm.png",
                                 title=f"{tag} — Confusion Matrix (binária) — {cls_name}")

    return best, df

def eval_intruder_binary_all_classifiers(X_faces_bc: np.ndarray, y_faces: np.ndarray,
                                         X_intr_bc: np.ndarray, runs: int, train_ratio: float, tag: str,
                                         classifier_order: List[Tuple[str, List[Dict[str, Any]]]] = None):
    """Questão 8: roda TODOS os classificadores e imprime com os termos do enunciado."""
    if classifier_order is None:
        classifier_order = CLASSIFIER_ORDER

    print(f">>> CONTROLE DE ACESSO — Questão 8 — {tag}")
    act_dir = OUT_DIR / tag
    act_dir.mkdir(parents=True, exist_ok=True)

    best_rows = []
    for cls_name, _ in classifier_order:
        best, df_grid = eval_intruder_binary_one_classifier(
            X_faces_bc, y_faces, X_intr_bc, runs, train_ratio, tag, cls_name
        )
        if best is not None:
            best_rows.append(best)

    if best_rows:
        df_best = pd.DataFrame(best_rows).sort_values('acuracia_media', ascending=False)
        df_best.to_csv(act_dir / "q8_binary_summary_best.csv", index=False)

        # ===== Impressão com os termos EXATOS do enunciado =====
        print("\n===== Questão 8 — Estatísticas (50 rodadas) =====")
        for r in df_best.itertuples(index=False):
            print(f"\nClassificador: {r.classificador}")
            print(f" - acurácia: {100*r.acuracia_media:.2f}% ± {100*r.acuracia_std:.2f}%")
            print(" - taxa de falsos negativos (proporção de pessoas às quais acesso foi permitido incorretamente): "
                  f"{100*r.fnr_media:.2f}% ± {100*r.fnr_std:.2f}%")
            print(" - taxa de falsos positivos (pessoas às quais acesso não foi permitido incorretamente): "
                  f"{100*r.fpr_media:.2f}% ± {100*r.fpr_std:.2f}%")
            print(f" - sensibilidade: {100*r.sens_media:.2f}% ± {100*r.sens_std:.2f}%")
            print(f" - precisão: {100*r.prec_media:.2f}% ± {100*r.prec_std:.2f}%")

        return df_best

    print("[WARN] Questão 8 não gerou linhas.")
    return None

# ====== Questão 8 (ALTERNATIVA) — Open-set por autorizados apenas + limiar ======
def _max_class_score(clf, X):
    """
    Retorna um score 'confiança de autorizado' por amostra.
    Preferência: prob máx (predict_proba). Alternativa: softmax(decision_function).
    """
    # 1) Probabilidade (se houver)
    if hasattr(clf, "predict_proba"):
        try:
            p = clf.predict_proba(X)
            if np.all(np.isfinite(p)):
                return np.max(p, axis=1)
        except Exception:
            pass
    # 2) Decision function -> softmax
    if hasattr(clf, "decision_function"):
        try:
            s = clf.decision_function(X)
            if s.ndim == 1:
                s = np.vstack([0*s, s]).T
            s = s - s.max(axis=1, keepdims=True)
            e = np.exp(s)
            p = e / np.sum(e, axis=1, keepdims=True)
            return np.max(p, axis=1)
        except Exception:
            pass
    # 3) Fallback: 1.0 (não deve ocorrer em seus classificadores)
    return np.ones(X.shape[0], dtype=np.float64)

def _threshold_from_train(scores_train: np.ndarray,
                          mode: str = "quantile",
                          q: float = 0.05,
                          k_std: float = 3.0) -> float:
    """
    Define o limiar τ a partir dos scores nos AUTORIZADOS de treino.
    - mode="quantile": τ = quantil q dos scores (ex.: q=0.05)
    - mode="mean-std": τ = mean - k_std*std
    """
    scores_train = np.asarray(scores_train, dtype=np.float64)
    if mode == "quantile":
        return float(np.quantile(scores_train, q))
    # robusto a caudas
    m, s = float(np.mean(scores_train)), float(np.std(scores_train) + 1e-12)
    return m - k_std*s

def eval_intruder_openset_authorized_only_one_classifier(
    X_faces_bc: np.ndarray, y_faces: np.ndarray, X_intr_bc: np.ndarray,
    runs: int, train_ratio: float, tag: str, cls_name: str,
    thr_mode: str = "quantile", thr_q: float = 0.05, thr_kstd: float = 3.0
):
    """
    Alternativa da Atividade 8 (open-set): treina MULTICLASSE apenas com AUTORIZADOS (y_faces).
    Na avaliação, calcula p_max (confiança) e rejeita como intruso se p_max < τ,
    onde τ é calibrado nos próprios autorizados de treino.
    Métrica binária final: 0=autorizado, 1=intruso.
    """
    print(f"   >> {cls_name}: Q8 (open-set) — autorizados-only + limiar…")
    act_dir = OUT_DIR / tag
    clf_dir = act_dir / "classifiers" / (cls_name + "_OPENSET")
    clf_dir.mkdir(parents=True, exist_ok=True)

    # Conjunto "autorizados" (multiclasse, como no resto do pipeline)
    X_auth = np.asarray(X_faces_bc, dtype=np.float32)
    y_auth = np.asarray(y_faces)

    # Injetamos intrusos apenas na fase de teste após calibrar τ nos autorizados de treino
    labels_bin = [0, 1]  # 0=autorizado, 1=intruso

    grid_variants = _grid_for_classifier(cls_name)
    combos = [(nk, nm, v) for nk, nm in NORM_OPTIONS.items() for v in grid_variants]

    rows = []
    best = None

    for norm_key, norm_maker, v in tqdm(combos, desc=f"{cls_name} | Q8 OPEN-SET busca", leave=False):
        sss = StratifiedShuffleSplit(n_splits=runs, train_size=train_ratio, random_state=123)

        accs, fnrs, fprs, senss, precs = [], [], [], [], []
        cm_sum = np.zeros((2, 2), dtype=np.int64)
        pred_total_time, pred_total_samples = 0.0, 0
        fit_times, pred_times, total_times = [], [], []

        for tr, te in sss.split(X_auth, y_auth):
            # Split só entre autorizados
            Xtr_a, Xte_a = X_auth[tr], X_auth[te]
            ytr_a, yte_a = y_auth[tr], y_auth[te]

            scaler = norm_maker()
            if scaler is not None:
                Xtr_a = scaler.fit_transform(Xtr_a)
                Xte_a = scaler.transform(Xte_a)
                Xintr = scaler.transform(X_intr_bc)
            else:
                Xintr = X_intr_bc

            # Estabilidade (como você já faz para LDA)
            if cls_name == "LDA":
                Xtr_a = _as64_finite(Xtr_a)
                Xte_a = _as64_finite(Xte_a)
                Xintr = _as64_finite(Xintr)

            maker = _maker_from_name_and_variant(cls_name, v)
            clf = maker()

            # Treina multiclasses apenas com AUTORIZADOS
            t0 = time.time()
            clf.fit(Xtr_a, ytr_a)
            fit_t = time.time() - t0

            # 1) Calibra o limiar τ com base em p_max nos AUTORIZADOS de treino
            pmax_train = _max_class_score(clf, Xtr_a)
            tau = _threshold_from_train(pmax_train, mode=thr_mode, q=thr_q, k_std=thr_kstd)

            # 2) Avalia em AUTORIZADOS de teste + INTRUSOS externos
            t1 = time.time()
            pmax_te_auth = _max_class_score(clf, Xte_a)
            ypred_auth_bin = (pmax_te_auth < tau).astype(int)  # 1=intruso se confiança baixa
            yauth_bin = np.zeros_like(ypred_auth_bin)          # rótulo verdadeiro=autorizado(0)

            pmax_intr = _max_class_score(clf, Xintr)
            ypred_intr_bin = (pmax_intr < tau).astype(int)
            yintr_bin = np.ones_like(ypred_intr_bin)           # verdadeiro=intruso(1)

            # Junta para métricas binárias
            ytrue = np.concatenate([yauth_bin, yintr_bin])
            ypred = np.concatenate([ypred_auth_bin, ypred_intr_bin])
            pred_t = time.time() - t1
            total_t = fit_t + pred_t

            # CM 2x2 e métricas no seu formato (positiva = intruso)
            tn, fp, fn, tp = confusion_matrix(ytrue, ypred, labels=labels_bin).ravel()
            acc, fnr, fpr, sens, prec = _binary_metrics_from_confusion(tn, fp, fn, tp)

            accs.append(acc); fnrs.append(fnr); fprs.append(fpr); senss.append(sens); precs.append(prec)
            cm_sum += np.array([[tn, fp], [fn, tp]], dtype=np.int64)

            fit_times.append(fit_t); pred_times.append(pred_t); total_times.append(total_t)
            pred_total_time += pred_t; pred_total_samples += len(ytrue)

        def pack_stats(vals):
            mean, std, vmin, vmax, med = _agg_stats(vals)
            return dict(media=mean, std=std, minimo=vmin, maximo=vmax, mediana=med)

        row = {
            'tag': tag, 'classificador': cls_name + " (OPEN-SET)",
            'normalizacao': norm_key, 'variante': str(v) if v else '-',
            'thr_mode': thr_mode, 'thr_q': thr_q, 'thr_kstd': thr_kstd,
            # métricas (média ± dp)
            'acuracia_media': pack_stats(accs)['media'], 'acuracia_std': pack_stats(accs)['std'],
            'fnr_media':      pack_stats(fnrs)['media'], 'fnr_std':      pack_stats(fnrs)['std'],
            'fpr_media':      pack_stats(fprs)['media'], 'fpr_std':      pack_stats(fprs)['std'],
            'sens_media':     pack_stats(senss)['media'], 'sens_std':     pack_stats(senss)['std'],
            'prec_media':     pack_stats(precs)['media'], 'prec_std':     pack_stats(precs)['std'],
            # tempos
            'tempo_fit_medio_s': np.mean(fit_times),
            'tempo_pred_medio_s': np.mean(pred_times),
            'tempo_total_medio_s': np.mean(total_times),
            'pred_ms_por_amostra': (pred_total_time / max(pred_total_samples, 1)) * 1000.0,
            # CM agregada
            '_cm_sum': cm_sum,
            '_cm_labels': np.array(labels_bin),
        }
        rows.append(row)
        if (best is None) or (row['acuracia_media'] > best['acuracia_media']):
            best = row

    df = pd.DataFrame(rows).sort_values('acuracia_media', ascending=False)
    df.to_csv(clf_dir / "q8_openset_grid_all.csv", index=False)

    # salva melhor e CM
    if best is not None:
        pd.DataFrame([best]).to_csv(clf_dir / "q8_openset_best.csv", index=False)
        cm = best['_cm_sum']; labels = best['_cm_labels']
        save_confusion_artifacts(cm, labels, clf_dir / "q8_openset_cm.csv", clf_dir / "q8_openset_cm.png",
                                 title=f"{tag} — Confusion Matrix (OPEN-SET) — {cls_name}")

    return best, df

def eval_intruder_openset_authorized_only_all_classifiers(
    X_faces_bc: np.ndarray, y_faces: np.ndarray, X_intr_bc: np.ndarray,
    runs: int, train_ratio: float, tag: str,
    classifier_order: List[Tuple[str, List[Dict[str, Any]]]] = None,
    thr_mode: str = "quantile", thr_q: float = 0.05, thr_kstd: float = 3.0
):
    """
    Alternativa da Q8 para TODOS os classificadores: open-set por limiar sobre p_max.
    """
    if classifier_order is None:
        classifier_order = CLASSIFIER_ORDER

    print(f">>> CONTROLE DE ACESSO — Q8 (OPEN-SET) — {tag}")
    act_dir = OUT_DIR / tag
    act_dir.mkdir(parents=True, exist_ok=True)

    best_rows = []
    for cls_name, _ in classifier_order:
        best, _ = eval_intruder_openset_authorized_only_one_classifier(
            X_faces_bc, y_faces, X_intr_bc, runs, train_ratio, tag, cls_name,
            thr_mode=thr_mode, thr_q=thr_q, thr_kstd=thr_kstd
        )
        if best is not None:
            best_rows.append(best)

    if best_rows:
        df_best = pd.DataFrame(best_rows).sort_values('acuracia_media', ascending=False)
        df_best.to_csv(act_dir / "q8_openset_summary_best.csv", index=False)

        print("\n===== Questão 8 (OPEN-SET) — Estatísticas (50 rodadas) =====")
        for r in df_best.itertuples(index=False):
            print(f"\nClassificador: {r.classificador}")
            print(f" - acurácia: {100*r.acuracia_media:.2f}% ± {100*r.acuracia_std:.2f}%")
            print(" - taxa de falsos negativos (acesso permitido indevidamente ao intruso): "
                  f"{100*r.fnr_media:.2f}% ± {100*r.fnr_std:.2f}%")
            print(" - taxa de falsos positivos (bloqueio indevido de autorizado): "
                  f"{100*r.fpr_media:.2f}% ± {100*r.fpr_std:.2f}%")
            print(f" - sensibilidade (TPR intruso): {100*r.sens_media:.2f}% ± {100*r.sens_std:.2f}%")
            print(f" - precisão (PPV intruso): {100*r.prec_media:.2f}% ± {100*r.prec_std:.2f}%")

        return df_best

    print("[WARN] Q8 (OPEN-SET) não gerou linhas.")
    return None

# ====== MAIN ======
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ATIVIDADE 1 — Vetorização SxS (sem PCA)
    print("\n========================= ATIVIDADE 1 =========================")
    print("Lendo faces e vetorizando (sem PCA)...")
    X_raw, y_raw, names = build_xy(DATA_DIR, size=SIZE)
    print(f"   [OK] X_raw: {X_raw.shape}, y: {y_raw.shape}, sujeitos únicos: {len(np.unique(y_raw))}")
    (OUT_DIR / "A1_raw").mkdir(parents=True, exist_ok=True)
    np.savetxt(OUT_DIR / "A1_raw" / f"X_raw_{SIZE}x{SIZE}.txt", X_raw, fmt="%.6f")
    np.savetxt(OUT_DIR / "A1_raw" / "y.txt", y_raw.reshape(-1,1), fmt="%d")

    # ATIVIDADE 2 — Sem PCA
    print("\n========================= ATIVIDADE 2 =========================")
    tag2 = f"A2_semPCA_{SIZE}x{SIZE}"
    _best2 = run_activity_eval_block(X_raw, y_raw, RUNS, TRAIN_RATIO, tag=tag2)
    save_q2_summary_and_plot(tag2)

    # ATIVIDADE 3 & 5 — PCA, gráficos e escolha de q
    print("\n========================= ATIVIDADE 3 e 5 =========================")
    tag3 = "A3_PCA_full"
    tag5_base = "A5_PCA_q"
    X_rot, eigvals, V, mu, X_pca_q, q, tag5 = pca_and_plots(X_raw, tag_qD_dir=tag3, tag_q_base=tag5_base, var_tgt=VAR_TGT)

    # ATIVIDADE 4 — Avaliação com PCA sem redução
    print("\n========================= ATIVIDADE 4 =========================")
    tag4 = "A4_PCAfull_qD"
    _best4 = run_activity_eval_block(X_rot, y_raw, RUNS, TRAIN_RATIO, tag=tag4)
    print_activity_table_pdf_style(tag4)

    # ATIVIDADE 6 — Avaliação com PCA reduzido
    print("\n========================= ATIVIDADE 6 =========================")
    tag6 = f"A6_PCA_q{q}"
    _best6 = run_activity_eval_block(X_pca_q, y_raw, RUNS, TRAIN_RATIO, tag=tag6)
    print_activity_table_pdf_style(tag6)

    # ATIVIDADE 7 — Box-Cox + z-score sobre PCA reduzido
    print("\n========================= ATIVIDADE 7 =========================")
    tag7 = f"A7_PCA_q{q}_BoxCoxZ"
    params_bc = fit_boxcox_zscore(X_pca_q)
    X_bc = apply_boxcox_zscore(X_pca_q, params_bc)
    (OUT_DIR / tag7).mkdir(parents=True, exist_ok=True)
    np.savetxt(OUT_DIR / tag7 / "X_pca_q_boxcox_z.txt", X_bc, fmt="%.6f")
    _best7 = run_activity_eval_block(X_bc, y_raw, RUNS, TRAIN_RATIO, tag=tag7)
    print_activity_table_pdf_style(tag7)

    # ATIVIDADE 8 — Controle de acesso (intruso) — TODOS os classificadores
    print("\n========================= ATIVIDADE 8 =========================")
    tag8 = f"A8_intruso_q{q}_BoxCoxZ"
    if INTRUDER_DIR.exists() and any(INTRUDER_DIR.iterdir()):
        print("Controle de acesso: Vetoriza -> PCA -> q -> Box-Cox + z -> Classificadores (todos)...")
        X_intr_raw, _names = load_intruder_set(INTRUDER_DIR, size=SIZE)
        X_intr_rot = (X_intr_raw - mu) @ V
        X_intr_q = X_intr_rot[:, :q]
        X_intr_bc = apply_boxcox_zscore(X_intr_q, params_bc)

        # X_faces já está em PCA->q->Box-Cox+z na variável X_bc (da Atividade 7)
        _best_q8 = eval_intruder_binary_all_classifiers(X_bc, y_raw, X_intr_bc, RUNS, TRAIN_RATIO, tag=tag8)
        print("   [OK] ATIVIDADE 8 concluída.")
    else:
        print("Pasta de intruso não encontrada ou vazia. Pulei a Atividade 8. (Crie 'intruder/' com imagens JPG/PNG)")

    print("\n========================= FIM =========================")
    print("Tudo pronto! Resultados organizados em:", OUT_DIR.resolve())

    # ATIVIDADE 8 — Controle de acesso (intruso) — TODOS os classificadores
    print("\n========================= ATIVIDADE 8 =========================")
    tag8 = f"A8_intruso_q{q}_BoxCoxZ"
    if INTRUDER_DIR.exists() and any(INTRUDER_DIR.iterdir()):
        print("Controle de acesso: Vetoriza -> PCA -> q -> Box-Cox + z -> Classificadores (todos)...")
        X_intr_raw, _names = load_intruder_set(INTRUDER_DIR, size=SIZE)
        X_intr_rot = (X_intr_raw - mu) @ V
        X_intr_q = X_intr_rot[:, :q]
        X_intr_bc = apply_boxcox_zscore(X_intr_q, params_bc)

        # (1) Versão original (binária)
        _best_q8 = eval_intruder_binary_all_classifiers(X_bc, y_raw, X_intr_bc, RUNS, TRAIN_RATIO, tag=tag8)
        print("   [OK] ATIVIDADE 8 binária concluída.")

        # (2) >>> NOVO: Alternativa open-set (autorizados + limiar sobre p_max)
        tag8_open = tag8 + "_OPENSET"
        _best_q8_open = eval_intruder_openset_authorized_only_all_classifiers(
            X_bc, y_raw, X_intr_bc, RUNS, TRAIN_RATIO, tag=tag8_open,
            thr_mode="quantile", thr_q=0.05, thr_kstd=3.0
        )
        print("   [OK] ATIVIDADE 8 OPEN-SET (autorizados + limiar) concluída.")
    else:
        print("Pasta de intruso não encontrada ou vazia. Pulei a Atividade 8. (Crie 'intruder/' com imagens JPG/PNG)")


if __name__ == "__main__":
    main()
