import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import sys

def load_csv_to_tensor(file_path, expected_shape=None, sep=";", device=""):
    data = pd.read_csv(file_path, header=None, sep=sep)
    data = data.apply(pd.to_numeric, errors='coerce').fillna(0)
    tensor = torch.tensor(data.values, dtype=torch.float32, device=device)

    if expected_shape and tensor.shape != expected_shape:
        raise ValueError(f"Arquivo {file_path} tem forma {tensor.shape}, mas esperava {expected_shape}")
    
    return tensor

def save_signal_result_to_png(signal_result: torch.Tensor, shape, path, file_name):
    matriz_image = signal_result.reshape(shape).T
    min_val = torch.min(matriz_image)
    max_val = torch.max(matriz_image)
    matriz_image = ((matriz_image - min_val) / (max_val - min_val)) * 255
    matriz_image = matriz_image.to("cpu")
    
    plt.imshow(matriz_image.byte().numpy(), cmap='gray')
    plt.savefig(os.path.join(path, file_name))

def cgnr(H, g, epsilon=1e-6, max_rep=1000):
    Ht = H.T
    
    if H.shape[0] != g.shape[0]:
        raise ValueError(f"Tamanho incompatível: H tem {H.shape[0]} linhas, mas g tem {g.shape[0]} linhas.")
    
    # Cálculo do fator de redução (c)
    c = torch.linalg.norm(Ht @ H, ord=2)
    H = H / c  # Normaliza H para maior estabilidade numérica
    
    # Cálculo do coeficiente de regularização (lambda)
    lambd = torch.max(torch.abs(Ht @ g)) * 0.10
    HtH = Ht @ H + lambd * torch.eye(H.shape[1], device=H.device)  # Regularização
    
    # Aplicação do ganho de sinal (γ)
    S, _ = g.shape
    gamma = torch.tensor([100 + (1/20) * l * torch.sqrt(torch.tensor(float(l))) for l in range(1, S + 1)], dtype=torch.float32).view(-1, 1)
    g = g * gamma
    
    f = torch.zeros(H.shape[1], 1)  # Resultado inicial (f0)
    r = g - (H @ f)  # r0
    z = Ht @ r  # z0
    p = z.clone()  # p0
    
    for i in range(max_rep):
        if i % 10 == 0:
            print(f"Iterações realizadas: {i}")
        
        w = H @ p  # w_i = H p_i
        alpha = (z.T @ z) / (w.T @ w)
        
        f_aux = f + alpha * p
        r_aux = r - alpha * w
        z_aux = Ht @ r_aux  # z_{i+1}
        beta = (z_aux.T @ z_aux) / (z.T @ z)
        p_aux = z_aux + beta * p
        
        # Cálculo do erro (epsilon)
        epsilon_i = torch.linalg.norm(r_aux) - torch.linalg.norm(r)
        
        if torch.linalg.norm(r_aux) < epsilon or abs(epsilon_i) < epsilon:  # Critério de convergência
            print(f"Convergiu em {i} iterações")
            f = f_aux
            break
        
        f, r, z, p = f_aux, r_aux, z_aux, p_aux
    
    return f

def main(args):
    print(args)
    try:
        if not args or len(args) < 9:
            raise ValueError("Número de argumentos inválido! Esperado: 10")
        
        signal_path = args[0]
        signal_shape = (int(args[1]), int(args[2]))
        matriz_path = args[3]
        matriz_shape = (int(args[4]), int(args[5]))
        result_path = args[6]
        result_shape = (int(args[7]), int(args[7]))
        signal_name = args[8]
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Dispositivo: {device}")
        
        tensor_signal = load_csv_to_tensor(signal_path, expected_shape=signal_shape, device=device)
        print("Tensor de sinal carregado com sucesso!")
        
        matriz_tensor = load_csv_to_tensor(matriz_path, expected_shape=matriz_shape, sep=",", device=device)
        print("Tensor de matriz carregado com sucesso!")

        print(f"Forma de H: {matriz_tensor.shape}, Forma de g: {tensor_signal.shape}")
        
        result = cgnr(matriz_tensor, tensor_signal, epsilon=1e-6, max_rep=5000)
        save_signal_result_to_png(result, result_shape, result_path, signal_name + "-result.png")
        
        print("Fim do programa")
        
    except (IndexError, ValueError) as e:
        print(f"Erro: {e}")

if __name__ == '__main__':
    main(sys.argv[1:])
