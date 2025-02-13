import base64
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
from log import log, reset_log

########################################

#Inicialização de variaveis

app = Flask(__name__)

reset_log()

# Fila global para armazenar os pedidos
pedidos_fila = Queue()

# Fila global para armazenar os pedidos iniciados e não recebidos
pedidos_iniciados = Queue()

# Lock para sincronizar o acesso à fila
fila_lock = threading.Lock()

# Lock para sincronizar o acesso à fila de pedidos iniciados
fila_iniciados_lock = threading.Lock()

# Lock para sincronizar acesso aos valores de uso de CPU e memória
cpu_mem_lock = threading.Lock()

# Lock para sincronizar o acesso ao contador de pedidos
qtd_pedido_lock = threading.Lock()

# Quantidade maxima de pedidos que podem ser processados ao mesmo tempo
max_pedidos = 1


# Configurar o dispositivo (CPU ou GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(0, f"Dispositivo: {device}")

contador_id = 0

media_porcentagem_cpu = 0
media_porcentagem_memoria = 0

qtd_pedido_processando = 0

########################################

# Funções auxiliares

def load_csv_to_tensor(file_path, expected_shape=None, sep=";", device=""):
    data = pd.read_csv(file_path, header=None, sep=sep)
    data = data.apply(pd.to_numeric, errors='coerce').fillna(0)
    tensor = torch.tensor(data.values, dtype=torch.float32, device=device)
    return tensor

def adquiri_matriz_H(shape_sinal, matrizes_path=os.path.join("server", "data")):
    """
    Carrega a matriz H correta com base no shape do sinal.
    :param shape_sinal: Shape do sinal (número de elementos).
    :return: Matriz H carregada como um tensor PyTorch.
    """
    global H1
    global H2
    
    if shape_sinal == 50816:
        return H1
    elif shape_sinal == 27904:
        return H2
    else:
        raise ValueError(f"Shape do sinal não suportado: {shape_sinal}")

def save_signal_result_to_png(signal_result: torch.Tensor, shape, path , file_name="image_result.png"):
    matriz_image = signal_result.reshape((int(shape),int(shape))).T 
    min_val = torch.min(matriz_image)
    max_val = torch.max(matriz_image)
    matriz_image = ((matriz_image - min_val) / (max_val - min_val)) * 255  
    matriz_image = matriz_image.to("cpu")

    plt.imsave(os.path.join(path, file_name), matriz_image.byte().numpy(), cmap='gray')

def imprimir_estado_fila():
    with fila_lock:  # Bloqueia o acesso à fila
        processos_na_fila = [p["id"] for p in list(pedidos_fila.queue)]
        log(0,f"IDs dos processos na fila: {processos_na_fila}" )

def adicionar_fila(id_processo):
    with open(f"./server/processos/{id_processo}/config.json", "r") as file:
        data = json.load(file)
    with fila_lock:  # Bloqueia o acesso ao contador e à fila
        # faz log com o id da thread do coordenador
        log(0, f"Adicionando processo {id_processo} à fila")
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
    
def verifica_situacao_servidor():
    global media_porcentagem_cpu
    global media_porcentagem_memoria
    
    global qtd_pedido_processando
    
    cpu_percent = psutil.cpu_percent(interval=None)
    mem_percent = psutil.virtual_memory().percent
    
    # Verifica se o uso CPU e memória estipulado está acima de 95 
    v_estipulado_cpu = cpu_percent + (qtd_pedido_processando + 1.2) * media_porcentagem_cpu 
    v_estipulado_memoria = mem_percent + (qtd_pedido_processando + 1.2) * media_porcentagem_memoria

    return v_estipulado_cpu > 80 or v_estipulado_memoria > 80
    
########################################

# Processamento de pedidos

def inicializar_coordenador():
    log(0, "Coordenador iniciado com sucesso!")
    num_pedidos_estipular = 3
    i = 0
    
    global media_porcentagem_cpu
    global media_porcentagem_cpu
    
    global max_pedidos
    
    soma_porcentagem_cpu = 0
    soma_porcentagem_memoria = 0

    while True:
        
        # Verifica se a fila está vazia (não precisa de lock, pois Queue é thread-safe)
        if pedidos_fila.empty():
            time.sleep(0.05)
            continue 
   
            
        if i < num_pedidos_estipular:
            with fila_lock:
                id_pedido = pedidos_fila.get()
            i += 1
            time.sleep(0.5)
            time_init = datetime.datetime.now()
            process_pedido(id_pedido)
            time_end = datetime.datetime.now()
            time.sleep(0.5)
            porcentagem_cpu, porcentagem_memoria  = analisar_cpu_mem(time_init, time_end)
            soma_porcentagem_cpu += porcentagem_cpu
            soma_porcentagem_memoria += porcentagem_memoria
            
            continue            
        
        if(i == num_pedidos_estipular):
            media_porcentagem_cpu = soma_porcentagem_cpu / num_pedidos_estipular
            media_porcentagem_memoria = soma_porcentagem_memoria / num_pedidos_estipular
            # Calcula o número máximo de pedidos que podem ser processados com base na média de uso de CPU 
            max_pedidos = int(100/(media_porcentagem_cpu)) * 3 + 5
            log(0, f"O número máximo de pedidos que podem ser processados é {max_pedidos}")
            log(0, f"Porcentagem de uso de CPU Médio durante o pedido: {media_porcentagem_cpu}")
            log(0, f"Porcentagem de uso de memória Médio durante o pedido: {media_porcentagem_memoria}")
            i += 1
            continue
        
        with fila_lock:
            id_pedido = pedidos_fila.get()
        v = True
        while verifica_situacao_servidor():
            time.sleep(0.2)
            if(v):
                log(0, "Servidor sobrecarregado, aguardando...")
                v = False
  
        threading.Thread(target=process_pedido, args=(id_pedido,)).start()
        
def process_pedido(data):
    global qtd_pedido_processando
    try:
        with qtd_pedido_lock:
            qtd_pedido_processando += 1
            
        log(data['id'],f"Processando pedido {data['id']} da fila")
        global device
        
        start_time = time.time()
        start_datetime = datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')
        print(data)
        sinal = load_csv_to_tensor(file_path=f"./server/processos/{data['id']}/sinal.csv", device=device)
        algoritmo = data["algoritmo"]
        shape = tuple(data["shape"])
        matriz_H = adquiri_matriz_H(sinal.shape[0])

        if algoritmo == "cgne":
            f, numero_iteracoes = cgne(matriz_H, sinal)
        elif algoritmo == "cgnr":
            f, numero_iteracoes= cgnr(matriz_H, sinal)
        else:
            raise ValueError(f"Algoritmo desconhecido: {algoritmo}")
        
        save_signal_result_to_png(f, shape=shape[0] , path=f"./server/processos/{data['id']}/", file_name="image_result.png")

        end_time = time.time()
        end_datetime = datetime.datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')

        resultado = {
            "algoritmo": algoritmo,
            "shape": shape,
            "numero_iteracoes": numero_iteracoes,
            "tempo": {
                "inicio": start_datetime,
                "fim": end_datetime,
                "total_segundos": round(end_time - start_time, 2)
            }
        }
        
        with open(f"./server/processos/{data['id']}/result.json", "w") as file:
            json.dump(resultado, file, indent=4)

        log(data["id"],f"Reconstrução concluída para o processo {data['id']} e resultado salvo.")
        imprimir_estado_fila()
        with qtd_pedido_lock:
            qtd_pedido_processando -= 1
        with fila_iniciados_lock:
            pedidos_iniciados.get(data)
        return resultado
       

    except Exception as e:
        log(0,f"Erro durante o processamento: {e}")
        with qtd_pedido_lock:
            qtd_pedido_processando -= 1
        with fila_iniciados_lock:
            pedidos_iniciados.get(data)
        raise

########################################

# Monitoramento

def inicializar_monitoramento():
    log(0, "Monitor iniciado com sucesso!")
    ## Essa função irá recolher dados de uso de CPU e memória
    ## e guardar em um arquivo com o tempo que foi recolhido esse dado
    ## para que possa ser feito um gráfico de uso de CPU e memória
    with open(f"./server/relatorio/monitoramento.csv", "w") as file:
            file.write("")
    
    while True:
        with open(f"./server/relatorio/monitoramento.csv", "a") as file:
            cpu_percent = psutil.cpu_percent(interval=None)
            mem_percent = psutil.virtual_memory().percent
            file.write(f"{datetime.datetime.now()},{cpu_percent},{mem_percent}\n")
        time.sleep(0.1)  
        
def analisar_cpu_mem(time_init, time_end):
    # Deve recolher dados de uso de CPU e memória do arquivo monitoramento.csv
    log(0, f"O tempo inicial é {time_init} e o tempo final é {time_end}")
    df = pd.read_csv(f"./server/relatorio/monitoramento.csv", header=None)
    df.columns = ["datetime", "cpu", "memoria"]
    df["datetime"] = pd.to_datetime(df["datetime"])
    # adicione 0.5 segundos para garantir que o tempo final seja incluído
    time_init -= datetime.timedelta(seconds=0.5)
    time_end += datetime.timedelta(seconds=0.5)
    df = df[(df["datetime"] >= time_init) & (df["datetime"] <= time_end)]
    
    porcentagem_cpu = df["cpu"].max() - df["cpu"].min()
    porcentagem_memoria = df["memoria"].max() - df["memoria"].min()
    
    log(0, f"O minimo de uso de CPU durante o pedido: {df['cpu'].min()} e o máximo: {df['cpu'].max()}")
    log(0, f"Porcentagem de uso de CPU durante o pedido: {porcentagem_cpu}")
    log(0, f"Porcentagem de uso de memória durante o pedido: {porcentagem_memoria}")
    
    return porcentagem_cpu, porcentagem_memoria
    
    

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
            if verifica_checksum(data["id"]) == False:
                return jsonify({"error": "Checksum inválido"}), 400
            adicionar_fila(data["id"])
            return jsonify({"message": "Sinal recebido com sucesso"}), 200
    
    return jsonify({"message": "Chunk recebido com sucesso"}), 200

@app.route('/processo/iniciar', methods=['POST'])
def iniciar_processo():
    global pedidos_iniciados
    global max_pedidos
    
    if pedidos_iniciados.qsize() >= max_pedidos:
        return jsonify({"error": "Número máximo de pedidos atingido"}), 400
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
        
    with fila_iniciados_lock:
        pedidos_iniciados.put(data)
        
    return jsonify({"message": "Processo iniciado com sucesso" , "id_processo": contador_id}), 200

def obter_processos_disponiveis():
    processos_disponiveis = []
    
    if not os.path.exists("./server/processos"):
        return processos_disponiveis

    for id_processo in os.listdir("./server/processos"):
        caminho_config = os.path.join("./server/processos", id_processo, "config.json")
        caminho_resultado = os.path.join("./server/processos", id_processo, "result.json")

        if os.path.exists(caminho_config) and os.path.exists(caminho_resultado):
            with open(caminho_config, "r") as f:
                try:
                    config = json.load(f)
                    processos_disponiveis.append({
                        "id": config["id"],
                        "nome": config["nome"],
                        "algoritmo": config["algoritmo"],
                        "shape": config["shape"]
                    })
                except json.JSONDecodeError:
                    print(f"Erro ao ler config.json do processo {id_processo}")

    print(f"processos_disponiveis {processos_disponiveis}")
    return processos_disponiveis

@app.route('/processo/enviar_resultado', methods=['GET'])
def enviar_resultado():
    """Retorna os resultados de todos os processos de um cliente, incluindo imagens se disponíveis."""
    nome_cliente = request.args.get("nome")

    print(f"Requisição recebida para enviar_resultado - Cliente: {nome_cliente}")

    if not nome_cliente:
        return jsonify({"erro": "Nome do cliente não fornecido"}), 400

    caminho_base = "./server/processos/"

    if not os.path.exists(caminho_base):
        print(f"Pasta de processos não encontrada: {caminho_base}")
        return jsonify({"erro": "Pasta de processos não encontrada"}), 500

    ids_disponiveis = os.listdir(caminho_base)
    print(f"IDs de processos encontrados: {ids_disponiveis}")

    resultados = []

    for process_id in ids_disponiveis:
        caminho_config = os.path.join(caminho_base, process_id, "config.json")
        caminho_resultado = os.path.join(caminho_base, process_id, "result.json")
        caminho_imagem = os.path.join(caminho_base, process_id, "image_result.png")

        print(f"Verificando processo {process_id}")
        print(f" - Caminho Config: {caminho_config} - Existe? {os.path.exists(caminho_config)}")
        print(f" - Caminho Resultado: {caminho_resultado} - Existe? {os.path.exists(caminho_resultado)}")
        print(f" - Caminho Imagem: {caminho_imagem} - Existe? {os.path.exists(caminho_imagem)}")

        if os.path.exists(caminho_config):
            try:
                with open(caminho_config, "r") as f:
                    config = json.load(f)

                print(f"Config carregado para processo {process_id}: {config}")

                # Verifica se o processo pertence ao cliente
                if config.get("nome") == nome_cliente:
                    print(f"Processo {process_id} pertence ao cliente {nome_cliente}")

                    resultado_data = {"id": process_id}

                    if os.path.exists(caminho_resultado):
                        with open(caminho_resultado, "r") as f:
                            resultado = json.load(f)
                        resultado_data["resultado"] = resultado
                        print(f"Resultado encontrado para o processo {process_id}")
                    else:
                        resultado_data["status"] = "processando"
                        print(f"Resultado ainda não disponível para o processo {process_id}")

                    # Se a imagem existir, adiciona ao JSON
                    if os.path.exists(caminho_imagem):
                        with open(caminho_imagem, "rb") as img_file:
                            imagem_base64 = base64.b64encode(img_file.read()).decode("utf-8")
                            resultado_data["imagem"] = imagem_base64
                            print(f"Imagem do processo {process_id} anexada ao JSON.")

                    resultados.append(resultado_data)

            except json.JSONDecodeError as e:
                print(f"Erro ao ler JSON para processo {process_id}: {e}")
                return jsonify({"erro": f"Erro ao ler arquivos JSON do processo {process_id}"}), 500
            except Exception as e:
                print(f"Erro inesperado no processo {process_id}: {e}")
                return jsonify({"erro": "Erro interno no servidor"}), 500

    return jsonify(resultados), 200

@app.route('/processo/enviar_resultado_id', methods=['GET'])
def enviar_resultado_id():
    """Retorna o resultado de um processo específico se estiver disponível."""
    id_processo = request.args.get("id")
    nome_cliente = request.args.get("nome")

    if not id_processo or not nome_cliente:
        return jsonify({"erro": "ID do processo ou nome do cliente não fornecido"}), 400

    caminho_base = os.path.join("./server/processos", str(id_processo))
    caminho_config = os.path.join(caminho_base, "config.json")
    caminho_resultado = os.path.join(caminho_base, "result.json")
    caminho_imagem = os.path.join(caminho_base, "image_result.png")

    if not os.path.exists(caminho_config):
        return jsonify({"erro": "Processo não encontrado"}), 404

    if not os.path.exists(caminho_resultado):
        return jsonify({"id": id_processo, "status": "processando"}), 202

    try:
        with open(caminho_config, "r") as f:
            config = json.load(f)

        if config["nome"] != nome_cliente:
            return jsonify({"erro": "Processo não pertence ao usuário"}), 403

        with open(caminho_resultado, "r") as f:
            resultado = json.load(f)

        response = {
            "id": id_processo,
            "resultado": resultado
        }

        if os.path.exists(caminho_imagem):
            with open(caminho_imagem, "rb") as img_file:
                imagem_base64 = base64.b64encode(img_file.read()).decode("utf-8")
                response["imagem"] = imagem_base64

        return jsonify(response), 200

    except json.JSONDecodeError as e:
        return jsonify({"erro": "Erro ao ler arquivos JSON"}), 500


# @app.route('/processo/enviar_resultado_id', methods=['GET'])
# def enviar_resultado_id():
#     #verifica o status de um pedido pelo id. se estiver processado, retorna os resultados
    

########################################

# Inicia a thread de processamento da fila
thread_fila = threading.Thread(target=inicializar_coordenador)
thread_fila.daemon = True  # Thread daemon para encerrar com o programa
thread_fila.start()


# Inicia a thread de monitoramento
thread_monitoramento = threading.Thread(target=inicializar_monitoramento)
thread_monitoramento.daemon = True  # Thread daemon para encerrar com o programa
thread_monitoramento.start()



try:
    H1 = torch.tensor(pd.read_csv("server/data/H-1.csv", header=None).values, dtype=torch.float32)
    H2 = torch.tensor(pd.read_csv("server/data/H-2.csv", header=None).values, dtype=torch.float32)
    log(0, "Matrizes H carregadas com sucesso!")
except Exception as e:
    log(0, f"Erro ao carregar matriz H: {e}")
    SystemExit(1)


if __name__ == "__main__":
    # Deleta pasta processos no windows
    try:
        os.system(r"rmdir /s /q .\server\processos")
    except:
        pass


    app.run(host='127.0.0.1', port=5000, debug=False)
    
     
    
    
    
    
    