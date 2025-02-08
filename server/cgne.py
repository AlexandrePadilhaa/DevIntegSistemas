import torch
import random
import time

#TODO Implementar algoritimo
def cgne(H, g, max_iter=1000, tol=1e-4):
    """
    Implementação do algoritmo CGNE.
    :param H: Matriz de modelo.
    :param g: Vetor de sinal.
    :param max_iter: Número máximo de iterações.
    :param tol: Tolerância para critério de parada.
    :return: Imagem reconstruída (f).
    """
    # Implementação do algoritmo CGNE
    f = ...  # Cálculo da imagem reconstruída
    time.sleep(random.randint(0, 20))
    return torch.ones(H.shape[1], dtype=torch.float32)

    