
import pandas as pd
import numpy as np
import warnings

def imputacao(df, cols, strategy='median', reference=None):
    """
    Realiza imputação de valores nulos em colunas numéricas utilizando
    média ou mediana.

    A função pode calcular o valor de imputação a partir do próprio
    dataframe ou utilizar valores previamente calculados (ex: treino),
    evitando data leakage.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contendo os dados.
    cols : list
        Lista de colunas numéricas a serem imputadas.
    strategy : str, default='median'
        Estratégia de imputação. Pode ser 'median' ou 'mean'.
    reference : dict or None, default=None
        Dicionário contendo valores de imputação calculados no conjunto
        de treino. Se None, o valor é calculado no próprio df.

    Returns
    -------
    pd.DataFrame
        DataFrame com os valores imputados.
    """
    df = df.copy()
    
    for col in cols:
        
        # imputação
        if reference is not None:
            # usar valor do treino
            fill_value = reference[col]
        else:
            # calcular no próprio df
            if strategy == 'median':
                fill_value = df[col].median()
            elif strategy == 'mean':
                fill_value = df[col].mean()
            else:
                raise ValueError("strategy deve ser 'median' ou 'mean'")
        
        df[col] = df[col].fillna(fill_value)
    
    return df


def identifica_outliers_iqr(df, features, factor=1.5):
    """
    Identifica colunas numéricas que apresentam outliers com base no método
    do Intervalo Interquartil (IQR).

    Variáveis binárias ou com baixa cardinalidade (<= 2 valores únicos)
    são ignoradas.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contendo os dados.
    features : list
        Lista de colunas numéricas a serem avaliadas.
    factor : float, default=1.5
        Fator multiplicador do IQR para definição dos limites.

    Returns
    -------
    list
        Lista de colunas que apresentam ao menos um outlier.
    """
    outlier_cols = []
    
    for col in features:
        if df[col].nunique() <= 2:
            continue
            
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - (factor * IQR)
        upper_bound = Q3 + (factor * IQR)
        
        if ((df[col] < lower_bound) | (df[col] > upper_bound)).any():
            outlier_cols.append(col)
            
    return outlier_cols


def fit_woe_binning(df, features_to_treat, target='y', n_bins=10):
    """
    Realiza o ajuste de binning e calcula o Weight of Evidence (WoE)
    para um conjunto de variáveis numéricas.

    O resultado é um dicionário contendo os bins e os respectivos valores
    de WoE, que pode ser utilizado posteriormente para transformação
    em novos dados (ex: validação ou teste).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de treino.
    features_to_treat : list
        Lista de variáveis numéricas a serem tratadas com WoE.
    target : str, default='y'
        Nome da variável target (binária).
    n_bins : int, default=10
        Número máximo de bins para discretização.

    Returns
    -------
    dict
        Dicionário contendo, para cada variável:
        - bins utilizados
        - valores de WoE por bin
    """
    woe_maps = {}
    
    for col in features_to_treat:
        try:
            df_temp = df[[col, target]].copy()
            df_temp['bin'], bins = pd.qcut(df_temp[col], q=n_bins, duplicates='drop', retbins=True)
            
            bins[0] = -np.inf
            bins[-1] = np.inf
            
            grouped = df_temp.groupby('bin', observed=False)[target].agg(['count', 'sum'])
            grouped.columns = ['total', 'bad']
            grouped['good'] = grouped['total'] - grouped['bad']
            
            total_bad = grouped['bad'].sum()
            total_good = grouped['good'].sum()
            
            if total_bad == 0 or total_good == 0:
                warnings.warn(f"Variável {col} não tem eventos suficientes para WoE.")
                continue

            grouped['dist_bad'] = grouped['bad'] / total_bad
            grouped['dist_good'] = grouped['good'] / total_good
            
            grouped['woe'] = np.log((grouped['dist_good'] + 0.0001) / (grouped['dist_bad'] + 0.0001))
            
            woe_dict = grouped['woe'].to_dict()
            
            woe_maps[col] = {
                'bins': bins,
                'woe_values': grouped['woe'].tolist()
            }
            
        except Exception as e:
            warnings.warn(f"Erro ao calcular WoE para {col}: {str(e)}")
            continue
            
    return woe_maps


def transform_with_woe(df, woe_maps):
    """
    Aplica a transformação WoE em um DataFrame utilizando os parâmetros
    previamente ajustados no conjunto de treino.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a ser transformado.
    woe_maps : dict
        Dicionário contendo bins e valores de WoE por variável,
        gerado pela função fit_woe_binning.

    Returns
    -------
    pd.DataFrame
        DataFrame com as variáveis transformadas em WoE.
    """
    df_transformed = df.copy()
    
    for col, map_info in woe_maps.items():
        bins = map_info['bins']
        woe_values = map_info['woe_values']
        
        df_transformed[col + '_binned'] = pd.cut(df_transformed[col], bins=bins, labels=False, include_lowest=True)
        
        map_dict = {i: val for i, val in enumerate(woe_values)}
        
        df_transformed[col] = df_transformed[col + '_binned'].map(map_dict).astype(float)
        
        df_transformed.drop(columns=[col + '_binned'], inplace=True)
        
    return df_transformed



def calcula_iv(df, feature, target='y', bins=10):
    """
    Calcula o Information Value (IV) de uma variável preditora
    em relação à variável target.

    O IV é utilizado para avaliar o poder preditivo da variável
    e auxiliar na seleção de features.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contendo os dados.
    feature : str
        Nome da variável preditora.
    target : str, default='y'
        Nome da variável target (binária).
    bins : int, default=10
        Número de bins para discretização da variável.

    Returns
    -------
    float
        Valor do Information Value (IV) da variável.
    """
    data = df[[feature, target]].copy()
    data[feature] = pd.qcut(data[feature], q=bins, duplicates='drop')

    grouped = data.groupby(feature)[target].agg(['count', 'sum'])
    grouped.columns = ['total', 'bad']
    grouped['good'] = grouped['total'] - grouped['bad']

    grouped['dist_good'] = grouped['good'] / grouped['good'].sum()
    grouped['dist_bad'] = grouped['bad'] / grouped['bad'].sum()

    grouped['woe'] = np.log(grouped['dist_good'] / grouped['dist_bad'])
    grouped['iv'] = (grouped['dist_good'] - grouped['dist_bad']) * grouped['woe']

    return grouped['iv'].sum()


