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

########################################

#Inicialização de variaveis

app = Flask(__name__)

# Fila global para armazenar os pedidos
pedidos_fila = Queue()

# Lock para sincronizar o acesso à fila
fila_lock = threading.Lock()

# Configurar o dispositivo (CPU ou GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {device}")

contador_id = 0

########################################

# Funções auxiliares

def load_csv_to_tensor(file_path, expected_shape=None, sep=";", device=""):
    data = pd.read_csv(file_path, header=None, sep=sep)
    data = data.apply(pd.to_numeric, errors='coerce').fillna(0)
    tensor = torch.tensor(data.values, dtype=torch.float32, device=device)
    return tensor

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

def save_signal_result_to_png(signal_result: torch.Tensor, shape, path , file_name="image_result.png"):
    matriz_image = signal_result.reshape((int(shape),int(shape))).T 
    min_val = torch.min(matriz_image)
    max_val = torch.max(matriz_image)
    matriz_image = ((matriz_image - min_val) / (max_val - min_val)) * 255  
    matriz_image = matriz_image.to("cpu")

    plt.imsave(os.path.join(path, file_name), matriz_image.byte().numpy(), cmap='gray')
    print(f"Imagem salva em {os.path.join(path, file_name)}") 

def imprimir_estado_fila():
    with fila_lock:  # Bloqueia o acesso à fila
        print("imprimir_estado_fila() acessando a fila")
        tamanho_fila = pedidos_fila.qsize()
        processos_na_fila = [p["id"] for p in list(pedidos_fila.queue)]
        print(f"Tamanho da fila: {tamanho_fila}")
        print("IDs dos processos na fila:", processos_na_fila)

def adicionar_fila(id_processo):
    with open(f"./server/processos/{id_processo}/config.json", "r") as file:
        data = json.load(file)
    with fila_lock:  # Bloqueia o acesso ao contador e à fila
        print("adicionar_fila() acessando a fila")
        pedidos_fila.put(data)
    imprimir_estado_fila()

def verifica_checksum(id_processo):
    with open(f"./server/processos/{id_processo}/config.json", "r") as file:
        data_config = json.load(file)
    with open(f"./server/processos/{id_processo}/sinal.csv", "r") as file:
        data = pd.read_csv(file, header=None)
        data = data.apply(pd.to_numeric, errors='coerce').fillna(0)
        tensor = torch.tensor(data.values, dtype=torch.float32)
    checksum = torch.sum(tensor).item()
    return checksum == data_config["checksum"]    
    
########################################

# Processamento de pedidos

def inicializar_coordenador():
    while True:
        # Verifica se a fila está vazia (não precisa de lock, pois Queue é thread-safe)
        if pedidos_fila.empty():
            time.sleep(1)
            continue 

        id_pedido = pedidos_fila.get()
        print(f"Processando pedido {id_pedido['id']} da fila")
        threading.Thread(target=process_pedido, args=(id_pedido,)).start()
        
def process_pedido(data):
    try:
        global device

        print("O id do pedido é: ", data["id"])
        start_time = time.time()
        start_datetime = datetime.date.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')
        cpu_start = psutil.cpu_percent(interval=None)
        mem_start = psutil.virtual_memory().used / (1024 * 1024)
        sinal = load_csv_to_tensor(file_path=f"./server/processos/{data["id"]}/sinal.csv", device=device)
        algoritmo = data["algoritmo"]
        shape = tuple(data["shape"])
        matriz_H = carregar_matriz_H(sinal.shape[0])

        if algoritmo == "cgne":
            f, numero_iteracoes = cgne(matriz_H, sinal)
        elif algoritmo == "cgnr":
            f, numero_iteracoes= cgnr(matriz_H, sinal)
        else:
            raise ValueError(f"Algoritmo desconhecido: {algoritmo}")
        
        save_signal_result_to_png(f, shape=shape[0] , path=f"./server/processos/{data["id"]}/", file_name="image_result.png")

        end_time = time.time()
        end_datetime = datetime.date.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')
        cpu_end = psutil.cpu_percent(interval=None)
        mem_end = psutil.virtual_memory().used / (1024 * 1024)

        resultado = {
            "algoritmo": algoritmo,
            "shape": shape,
            "numero_iteracoes": numero_iteracoes,
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
        
        with open(f"./server/processos/{data["id"]}/result.json", "w") as file:
            json.dump(resultado, file, indent=4)

        print(f"Reconstrução concluída para o processo {data["id"]} e resultado salvo.")
        imprimir_estado_fila()
        return resultado

    except Exception as e:
        print(f"Erro durante o processamento: {e}")
        raise

########################################

# Rotas

@app.route('/processo/receber', methods=['POST'])
def receber_chunk():
    
    data_str = request.get_json()  # Garante que os dados sejam interpretados como JSON
    if data_str is None:
        return jsonify({"error": "Dados inválidos ou ausentes"}), 400
    data = json.loads(data_str)
    
    # verifica se o processo existe
    if not os.path.exists(f"./server/processos/{data['id']}"):
        return jsonify({"error": "Processo não encontrado"}), 404
 
    if not os.path.exists(f"./server/processos/{data['id']}/sinal.csv"):
        with open(f"./server/processos/{data['id']}/sinal.csv", "w") as file:
            file.write("")
            
    chunck = data["sinal"]
    with open(f"./server/processos/{data['id']}/sinal.csv", "a") as file:
        for elemento in chunck:
            file.write(f"{elemento[0]}\n")
        
        
    if data["isLast"]:
            print(f"Último chunk recebido para o processo {data['id']}")
            if verifica_checksum(data["id"]) == False:
                return jsonify({"error": "Checksum inválido"}), 400
            adicionar_fila(data["id"])
            return jsonify({"message": "Sinal recebido com sucesso"}), 200
    
    return jsonify({"message": "Chunk recebido com sucesso"}), 200

@app.route('/processo/iniciar', methods=['POST'])
def iniciar_processo():
    
    global contador_id
    contador_id += 1
    
    data_str = request.get_json()  # Garante que os dados sejam interpretados como JSON
    if data_str is None:
        return jsonify({"error": "Dados inválidos ou ausentes"}), 400
    data = json.loads(data_str)
    
    data["id"] = contador_id
    mkdir = f"./server/processos/{contador_id}"
    
    # Cria a pasta do processo
    os.makedirs(mkdir, exist_ok=True)
    
    # Salva o arquivo de configuração
    with open(f"{mkdir}/config.json", "w") as file:
        json.dump(data, file, indent=4)
        
    return jsonify({"message": "Processo iniciado com sucesso" , "id_processo": contador_id}), 200

########################################

# Inicia a thread de processamento da fila
thread_fila = threading.Thread(target=inicializar_coordenador)
thread_fila.daemon = True  # Thread daemon para encerrar com o programa
thread_fila.start()

if __name__ == "__main__":
    # Deleta pasta processos no windows
    try:
        os.system(r"rmdir /s /q .\server\processos")
    except:
        pass

    app.run(host='127.0.0.1', port=5000, debug=False)
    
     
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
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