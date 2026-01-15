
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
import pandas as pd


# -----------------------------
# Funções de métricas
# -----------------------------

def ks_statistic(y_true, y_score):
    """
    Calcula o KS (Kolmogorov-Smirnov)
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    ks = np.max(tpr - fpr)
    return ks

def gini_coefficient(y_true, y_score):
    """
    Calcula o Gini a partir do AUC
    """
    auc = roc_auc_score(y_true, y_score)
    return 2 * auc - 1

def compute_metrics(y_true, y_score):
    """
    Retorna AUC, KS e Gini
    """
    auc = roc_auc_score(y_true, y_score)
    ks = ks_statistic(y_true, y_score)
    gini = 2 * auc - 1

    return {
        "AUC": auc,
        "KS": ks,
        "Gini": gini
    }




def psi_variavel(expected, actual, bins=10, eps=1e-6):
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

import pandas as pd

def psi_por_variavel_com_coef(X_train, X_other, model, bins=10):
    psi_dict = {}

    # coeficientes da regressão
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



def treinar_e_avaliar(
        pipelines, target_col="y", 
        max_iter=1000, 
        random_state=42, 
        penalty='l2',
        resultados_existentes=None,
        modelos_existentes=None,
        psi_existente=None

        ):
    """
    Avalia um ou mais pipelines usando Regressão Logística.
    
    Parâmetros
    ----------
    pipelines : dict
        Dicionário no formato:
        {
            "NomePipeline": (df_treino, df_validacao, df_teste)
        }
    target_col : str, default="y"
        Nome da coluna alvo

    max_iter : int, default=1000
        Número máximo de iterações do algoritmo de otimização.
        Valores maiores ajudam a garantir convergência, especialmente
        quando os dados não estão bem escalados.

    random_state : int, default=42
        Semente aleatória para garantir reprodutibilidade dos resultados.
        Garante que o treinamento produza os mesmos coeficientes
        em execuções diferentes.

    penalty : str, default='l2'
        Tipo de regularização aplicada ao modelo.
    
    Retorno
    -------
    dict
        Resultados por pipeline (DataFrame de métricas)
    """
    
    resultado_metricas = resultados_existentes or {}
    modelos_finais = modelos_existentes or {}
    psi_resultados = psi_existente or {}

    for nome, (df_train, df_valid, df_test) in pipelines.items():

        # if nome in resultado_metricas:
        #     warnings.warn(f"⚠️ Pipeline '{nome}' já existe — será sobrescrito.")
        #     continue

        X_train = df_train.drop(columns=[target_col, 'safra', 'id'])
        y_train = df_train[target_col]

        X_valid = df_valid.drop(columns=[target_col, 'safra', 'id'])
        y_valid = df_valid[target_col]

        X_test = df_test.drop(columns=[target_col, 'safra', 'id'])
        y_test = df_test[target_col]

        model = LogisticRegression(
            max_iter=max_iter,
            random_state=random_state,
            penalty=penalty
        )

        model.fit(X_train, y_train)

        y_train_score = model.predict_proba(X_train)[:, 1]
        y_valid_score = model.predict_proba(X_valid)[:, 1]
        y_test_score  = model.predict_proba(X_test)[:, 1]

        df_metricas = pd.DataFrame(
            {
                "Treino": compute_metrics(y_train, y_train_score),
                "Validação": compute_metrics(y_valid, y_valid_score),
                "Teste": compute_metrics(y_test, y_test_score),
            }
        ).T

        print(f"\nPipeline utilizada: {nome}")
        print(df_metricas.round(4))

        psi_valid = psi_por_variavel_com_coef(X_train, X_valid, model, bins=10)
        psi_test  = psi_por_variavel_com_coef(X_train, X_test, model,  bins=10)

        psi_score_valid = psi_score(y_train_score, y_valid_score, bins=10)
        psi_score_test  = psi_score(y_train_score, y_test_score,  bins=10)


        resultado_metricas[nome] = df_metricas
        modelos_finais[nome] = model
        psi_resultados[nome] = {
            "variaveis": {
                "treino_val": psi_valid,
                "treino_teste": psi_test,
            },
            "score": {
                "treino_val": psi_score_valid,
                "treino_teste": psi_score_test,
            },
        }

    return resultado_metricas, modelos_finais, psi_resultados



import numpy as np
import pandas as pd

