import datetime
import os
from flask import Flask, request, jsonify
import threading
from matplotlib import pyplot as plt
from numpy import sqrt
import pandas as pd
import torch
import json
from queue import Queue
import time
import psutil
from cgnr import cgnr  # Importa os algoritmos de reconstrução
from cgne import cgne

app = Flask(__name__)

# Fila global para armazenar os pedidos
pedidos_fila = Queue()

# Lock para sincronizar o acesso à fila
fila_lock = threading.Lock()

contador_id = 0

def carregar_matriz_H(shape_sinal, matrizes_path = ".\server\data"):
    """
    Carrega a matriz H correta com base no shape do sinal.
    :param shape_sinal: Shape do sinal (número de elementos).
    :return: Matriz H carregada como um tensor PyTorch.
    """
    if shape_sinal == 50816:
        caminho_H = os.path.join(matrizes_path, "H-1.csv")
    elif shape_sinal == 27904:
        caminho_H = os.path.join(matrizes_path, "H-2.csv")
    else:
        raise ValueError(f"Shape do sinal não suportado: {shape_sinal}")

    try:
        #print(f"Matriz utilizada: {caminho_H}")
        df = pd.read_csv(caminho_H, header=None)
        H = torch.tensor(df.values, dtype=torch.float32)
        return H
    except Exception as e:
        raise ValueError(f"Erro ao carregar a matriz H: {e}")

def save_signal_result_to_png(signal_result: torch.Tensor, shape, path="./server/images/", file_name="image_result.png"):
    os.makedirs(path, exist_ok=True) 
    
    matriz_image = signal_result.reshape((int(shape),int(shape))).T 
    min_val = torch.min(matriz_image)
    max_val = torch.max(matriz_image)
    matriz_image = ((matriz_image - min_val) / (max_val - min_val)) * 255  
    matriz_image = matriz_image.to("cpu")

    plt.imshow(matriz_image.byte().numpy(), cmap='gray')  
    plt.axis("off") 
    plt.savefig(os.path.join(path, file_name), bbox_inches="tight", pad_inches=0)
    print(f"Imagem salva em {os.path.join(path, file_name)}")

def process_data(data):
    try:
        start_time = time.time()
        start_datetime = datetime.date.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')
        cpu_start = psutil.cpu_percent(interval=None)
        mem_start = psutil.virtual_memory().used / (1024 * 1024)
        
        sinal = torch.tensor(data["sinal"], dtype=torch.float32)
        algoritmo = data["algoritmo"]
        shape = tuple(data["shape"])
        matriz_H = carregar_matriz_H(sinal.shape[0])

        if algoritmo == "cgne":
            f = cgne(matriz_H, sinal)
        elif algoritmo == "cgnr":
            f = cgnr(matriz_H, sinal)
        else:
            raise ValueError(f"Algoritmo desconhecido: {algoritmo}")
        
        save_signal_result_to_png(f, sqrt(matriz_H.shape[1]), file_name=f"resultado_{data['id']}.png")

        end_time = time.time()
        end_datetime = datetime.date.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')
        cpu_end = psutil.cpu_percent(interval=None)
        mem_end = psutil.virtual_memory().used / (1024 * 1024)

        resultado = {
            "algoritmo": algoritmo,
            "shape": shape,
            "tempo": {
                "inicio": start_datetime,
                "fim": end_datetime,
                "total_segundos": round(end_time - start_time, 2)
            },
            "desempenho": {
                "cpu_uso_percentual": round((cpu_start + cpu_end) / 2, 2),
                "memoria_mb": {
                    "inicio": round(mem_start, 2),
                    "fim": round(mem_end, 2),
                    "usada": round(mem_end - mem_start, 2)
                }
            }
        }
        
        with open(f"./server/results/resultado_{data['id']}.json", "w") as file:
            json.dump(resultado, file, indent=4)

        print(f"Reconstrução concluída para o processo {data['id']} e resultado salvo.")
        imprimir_estado_fila()
        return resultado

    except Exception as e:
        print(f"Erro durante o processamento: {e}")
        raise


    except Exception as e:
        print(f"Erro durante o processamento: {e}")
        raise


def processar_fila():
    while True:
        # Verifica se a fila está vazia (não precisa de lock, pois Queue é thread-safe)
        if pedidos_fila.empty():
            #print("Fila vazia. Aguardando novos pedidos...")
            time.sleep(10) 
            continue 

        # Remove o próximo pedido da fila (thread-safe) TODO Escolher forma de processar pedidos. Atualmente processa um de cada vez.
        data = pedidos_fila.get()
        print(f"Processando pedido: {data['id']}")

        try:
            # Processa os dados usando a função process_data
            resultado = process_data(data)
            print(f"Pedido {data['id']} processado com sucesso")
        except Exception as e:
            print(f"Falha ao processar pedido {data['id']}: {e}")
        finally:
            # Marca o pedido como concluído (thread-safe)
            pedidos_fila.task_done()


def imprimir_estado_fila():
    with fila_lock:  # Bloqueia o acesso à fila
        print("imprimir_estado_fila() acessando a fila")
        tamanho_fila = pedidos_fila.qsize()
        processos_na_fila = [p["id"] for p in list(pedidos_fila.queue)]
        print(f"Tamanho da fila: {tamanho_fila}")
        print("IDs dos processos na fila:", processos_na_fila)


@app.route('/processar', methods=['POST'])
def handle_client():
    print(f"Pedido recebido!")
    global contador_id

    data_str = request.get_json()  # Garante que os dados sejam interpretados como JSON
    if data_str is None:
        return jsonify({"error": "Dados inválidos ou ausentes"}), 400
    data = json.loads(data_str)
    

    # Adiciona um ID único ao processo
    with fila_lock:  # Bloqueia o acesso ao contador e à fila
        print("handle_client() acessando a fila")
        contador_id += 1
        data["id"] = contador_id
        pedidos_fila.put(data)
    
    imprimir_estado_fila()

    return jsonify({"message": f"Dados recebidos e adicionados à fila de processamento! ID: {data['id']}"}), 202

# Inicia a thread de processamento da fila
thread_fila = threading.Thread(target=processar_fila)
thread_fila.daemon = True  # Thread daemon para encerrar com o programa
thread_fila.start()

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=True)