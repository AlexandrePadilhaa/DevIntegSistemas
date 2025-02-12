import torch

def cgne(H, g, max_iter=1000, tol=1e-4):
    """
    Implementação do algoritmo CGNE (Conjugate Gradient Normal Error).
    :param H: Matriz de modelo (torch.Tensor de tamanho [m, n]).
    :param g: Vetor de sinal (torch.Tensor de tamanho [m, 1]).
    :param max_iter: Número máximo de iterações.
    :param tol: Tolerância para critério de parada.
    :return: Imagem reconstruída (f) como um tensor PyTorch.
    """
    Ht = H.T
    f = torch.zeros(H.shape[1], 1, dtype=H.dtype, device=H.device)  # Inicialização f0
    r = g - torch.matmul(H, f)                                      # r0 = g - H f0
    z = torch.matmul(Ht, r)                                         # Projeção z0 = H^T r0
    p = z                                                           # p0 = z0

    for i in range(max_iter):
        Hp = torch.matmul(H, p)                                     # Calcula H * p_i
        alpha = torch.matmul(z.T, z) / torch.matmul(Hp.T, Hp)       # α_i = (z_i^T z_i)/(Hp_i^T Hp_i)
        
        f = f + alpha * p                                           # Atualiza f_{i+1}
        r_next = r - alpha * Hp                                     # Atualiza r_{i+1}
        z_next = torch.matmul(Ht, r_next)                           # Atualiza z_{i+1}
        
        # Verifica convergência pela variação da norma do residual
        error = abs(torch.norm(r, 2).item() - torch.norm(r_next, 2).item())
        print(f"i {i} error {error} tol {tol}")
        if error < tol:
            print(f"Convergência alcançada na iteração {i}.")
            break
        
        beta = torch.matmul(z_next.T, z_next) / torch.matmul(z.T, z) # β_i = (z_{i+1}^T z_{i+1})/(z_i^T z_i)
        p = z_next + beta * p                                       # Atualiza p_{i+1}
        r = r_next                                                  # Prepara próximo residual
        z = z_next                                                  # Prepara próxima projeção

    print("Processamento finalizado.")
    return f