import os
from matplotlib import pyplot as plt
import torch
import random
import time


def cgnr(H, g, max_iter=1000, tol=1e-4):
    """
    Implementação do algoritmo CGNR (Conjugate Gradient Normal Residual).
    :param H: Matriz de modelo (torch.Tensor de tamanho [m, n]).
    :param g: Vetor de sinal (torch.Tensor de tamanho [m, 1]).
    :param max_iter: Número máximo de iterações.
    :param tol: Tolerância para critério de parada.
    :return: Imagem reconstruída (f) como um tensor PyTorch.
    """
    Ht = H.T
    f = torch.zeros(H.shape[1], 1, dtype=H.dtype)  
    r = g - torch.matmul(H, f)
    z = torch.matmul(Ht, r)
    p = z

    for i in range(max_iter):
        
        w = torch.matmul(H, p)
        alpha = torch.matmul(z.T, z) / torch.matmul(w.T, w)
        f = f + alpha * p
        r_next = r - alpha * w
        z_next = torch.matmul(Ht, r_next)

        error = abs(torch.norm(r, p=2).item() - torch.norm(r_next, p=2).item())
        if error < tol:
            numero_iteracoes = i
            break

        beta = torch.matmul(z_next.T, z_next) / torch.matmul(z.T, z)
        p = z_next + beta * p
        r = r_next
        z = z_next

    print('algoritmo CGNR finalizado.')
    return f, numero_iteracoes
