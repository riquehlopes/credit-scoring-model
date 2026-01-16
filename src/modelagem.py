import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import warnings

# -----------------------------
# Funções de métricas de modelo
# -----------------------------

def ks_statistic(y_true, y_score):
    """
    Calcula a estatística KS (Kolmogorov-Smirnov) para modelos de classificação.

    Parâmetros
    ----------
    y_true : array-like
        Valores reais da variável alvo (0 ou 1).
    y_score : array-like
        Scores ou probabilidades estimadas pelo modelo.

    Retorno
    -------
    float
        Valor do KS (varia entre 0 e 1).
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    ks = np.max(tpr - fpr)
    return ks


def gini_coefficient(y_true, y_score):
    """
    Calcula o coeficiente de Gini a partir do AUC.

    Fórmula:
        Gini = 2 * AUC - 1

    Parâmetros
    ----------
    y_true : array-like
        Valores reais da variável alvo.
    y_score : array-like
        Scores ou probabilidades do modelo.

    Retorno
    -------
    float
        Valor do coeficiente de Gini.
    """
    auc = roc_auc_score(y_true, y_score)
    return 2 * auc - 1


def compute_metrics(y_true, y_score):
    """
    Calcula as principais métricas de avaliação de um modelo de classificação.

    Métricas retornadas:
    - AUC (Área sob a curva ROC)
    - KS (Kolmogorov-Smirnov)
    - Gini

    Parâmetros
    ----------
    y_true : array-like
        Valores reais da variável alvo.
    y_score : array-like
        Scores ou probabilidades estimadas pelo modelo.

    Retorno
    -------
    dict
        Dicionário contendo AUC, KS e Gini.
    """
    auc = roc_auc_score(y_true, y_score)
    ks = ks_statistic(y_true, y_score)
    gini = 2 * auc - 1

    return {
        "AUC": auc,
        "KS": ks,
        "Gini": gini
    }


# -----------------------------
# Funções de PSI (Population Stability Index)
# -----------------------------

def psi_variavel(expected, actual, bins=10, eps=1e-6):
    """
    Calcula o PSI (Population Stability Index) para uma variável numérica.

    Parâmetros
    ----------
    expected : array-like
        Valores da variável na base de referência.
    actual : array-like
        Valores da variável na base comparada.
    bins : int, opcional (default=10)
        Número de faixas (quantis) utilizadas no cálculo.
    eps : float, opcional (default=1e-6)
        Valor pequeno para evitar log(0).

    Retorno
    -------
    float
        Valor do PSI da variável.
    """
    breakpoints = np.percentile(expected, np.arange(0, 100, 100 / bins))
    breakpoints = np.append(breakpoints, np.inf)

    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts   = np.histogram(actual,   bins=breakpoints)[0]

    expected_perc = expected_counts / len(expected)
    actual_perc   = actual_counts / len(actual)

    psi = np.sum(
        (actual_perc - expected_perc) *
        np.log((actual_perc + eps) / (expected_perc + eps))
    )

    return psi


def psi_por_variavel_com_coef(X_train, X_other, model, bins=10):
    """
    Calcula o PSI de cada variável numérica e associa o coeficiente
    correspondente de um modelo de regressão logística.

    Parâmetros
    ----------
    X_train : pandas.DataFrame
        Base de referência (normalmente treino/desenvolvimento).
    X_other : pandas.DataFrame
        Base a ser comparada (validação, OOT, produção, etc).
    model : sklearn.linear_model.LogisticRegression
        Modelo de regressão logística treinado.
    bins : int, opcional (default=10)
        Número de faixas (quantis) usadas no PSI.

    Retorno
    -------
    pandas.DataFrame
        DataFrame com PSI e coeficiente por variável,
        ordenado do maior para o menor PSI.
    """
    psi_dict = {}

    coeficientes = pd.Series(
        model.coef_[0],
        index=X_train.columns
    )

    for col in X_train.columns:
        if not pd.api.types.is_numeric_dtype(X_train[col]):
            continue

        psi = psi_variavel(
            X_train[col].values,
            X_other[col].values,
            bins=bins
        )

        psi_dict[col] = {
            "PSI": psi,
            "coeficiente": coeficientes[col]
        }

    df_psi = (
        pd.DataFrame.from_dict(psi_dict, orient='index')
        .sort_values("PSI", ascending=False)
    )

    return df_psi


def psi_score(expected, actual, bins=10, eps=1e-6):
    """
    Calcula o PSI para scores do modelo.

    Parâmetros
    ----------
    expected : array-like
        Scores da base de referência.
    actual : array-like
        Scores da base comparada.
    bins : int, opcional (default=10)
        Número de faixas (quantis).
    eps : float, opcional (default=1e-6)
        Valor pequeno para evitar log(0).

    Retorno
    -------
    float
        Valor do PSI do score.
    """
    breakpoints = np.percentile(expected, np.arange(0, 100, 100 / bins))
    breakpoints = np.append(breakpoints, np.inf)

    exp_counts = np.histogram(expected, bins=breakpoints)[0]
    act_counts = np.histogram(actual, bins=breakpoints)[0]

    exp_perc = exp_counts / len(expected)
    act_perc = act_counts / len(actual)

    return np.sum(
        (act_perc - exp_perc) *
        np.log((act_perc + eps) / (exp_perc + eps))
    )


def psi_por_safra(df, score_col, safra_col='safra'):
    """
    Calcula o PSI do score por safra, usando a primeira safra
    como base de referência.

    Parâmetros
    ----------
    df : pandas.DataFrame
        DataFrame contendo score e safra.
    score_col : str
        Nome da coluna de score do modelo.
    safra_col : str, opcional (default='safra')
        Nome da coluna que identifica a safra/período.

    Retorno
    -------
    pandas.Series
        Série com o PSI do score para cada safra.
    """
    safras = sorted(df[safra_col].unique())
    base_safra = safras[0]
    
    base_scores = df[df[safra_col] == base_safra][score_col]
    
    psi_values = {}
    for safra in safras:
        current_scores = df[df[safra_col] == safra][score_col]
        psi_values[safra] = psi_variavel(base_scores, current_scores)
    
    return pd.Series(psi_values)
