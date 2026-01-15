import pickle
import os


def salvar_objeto(obj, caminho):
    """
    Salva qualquer objeto Python usando pickle.
    """
    os.makedirs(os.path.dirname(caminho), exist_ok=True)

    with open(caminho, "wb") as f:
        pickle.dump(obj, f)


def carregar_objeto(caminho):
    """
    Carrega um objeto salvo com pickle.
    """
    with open(caminho, "rb") as f:
        return pickle.load(f)
