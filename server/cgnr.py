import os
from matplotlib import pyplot as plt
import torch
import random
import time

#TODO Implementar algoritimo

def cgnr(H, g, max_iter=10000, tol=1e-4):
    """
    Implementação do algoritmo CGNR (Conjugate Gradient Normal Residual).
    :param H: Matriz de modelo (torch.Tensor de tamanho [m, n]).
    :param g: Vetor de sinal (torch.Tensor de tamanho [m, 1]).
    :param max_iter: Número máximo de iterações.
    :param tol: Tolerância para critério de parada.
    :return: Imagem reconstruída (f) como um tensor PyTorch.
    """
    # Verificação de shapes
    assert H.shape[0] == g.shape[0], "Dimensões incompatíveis entre H e g"
    assert g.shape[1] == 1, "O vetor g deve ter shape [m, 1]"

    # Inicializar f0 como um vetor de zeros
    f = torch.zeros((H.shape[1], 1), dtype=torch.float32)

    # Computar o resíduo inicial r0 = g - Hf0
    r = g - H @ f

    # Computar z0 = H^T * r0
    z = H.T @ r

    # Definir p0 = z0
    p = z

    for i in range(max_iter):
        if i % 10 == 0:
            print(f"Iterações realizadas: {i}")

        # wi = H * pi
        w = H @ p

        # αi = ||zi||^2 / ||wi||^2
        z_norm_sq = z.T @ z
        w_norm_sq = w.T @ w
        alpha = z_norm_sq / (w_norm_sq + 1e-10)  # Pequena constante para evitar divisão por zero

        # fi+1 = fi + αi * pi
        f = f + alpha * p

        # ri+1 = ri - αi * wi
        r = r - alpha * w

        # zi+1 = H^T * ri+1
        z_new = H.T @ r

        # Critério de convergência (norma do resíduo menor que a tolerância)
        if torch.norm(z_new) < tol:
            print(f"Convergiu em {i} iterações")
            break

        # βi = ||zi+1||^2 / ||zi||^2
        z_new_norm_sq = z_new.T @ z_new
        beta = z_new_norm_sq / (z_norm_sq + 1e-10)  # Pequena constante para evitar divisão por zero

        # pi+1 = zi+1 + βi * pi
        p = z_new + beta * p

        # Atualizar z
        z = z_new

    return f

