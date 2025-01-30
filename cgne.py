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

def cgne(H, g, epsilon=1e-6, max_rep=1000):
    Ht = H.T
    n = H.shape[1]

    f = torch.zeros(n, 1)  # Resultado inicial (f0)
    r = g - (H @ f)  # r0
    p = Ht @ r  # p0

    for i in range(max_rep):
        ai = ((r.T @ r) / (p.T @ p)).item()
        f_aux = f + (ai * p)
        r_aux = r - (ai * (H @ p))
        Bi = ((r_aux.T @ r_aux) / (r.T @ r)).item()
        p_aux = (H.T @ r_aux) + (Bi * p)

        if torch.linalg.norm(r) < epsilon:  # Verifica se há convergência utilizando epsilon
            print(f"Convergiu em {i} iterações")
            f = f_aux
            break

        f = f_aux
        r = r_aux
        p = p_aux

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


        # Configurar o dispositivo (CPU ou GPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Dispositivo: {device}")

        # Leitura do sinal
        tensor_signal = load_csv_to_tensor(signal_path, expected_shape=signal_shape, device=device)
        print("Tensor de sinal carregado com sucesso!")

        # Leitura da matriz
        matriz_tensor = load_csv_to_tensor(matriz_path, expected_shape=matriz_shape, sep=",", device=device)
        print("Tensor de matriz carregado com sucesso!")

        result = cgne(matriz_tensor, tensor_signal, epsilon=1e-6, max_rep=5000)

        save_signal_result_to_png(result, result_shape, result_path, signal_name + "-result.png")

        print("Fim do programa")

    except (IndexError, ValueError) as e:
        print(f"Erro: {e}")

if __name__ == '__main__':

    main(sys.argv[1:])
