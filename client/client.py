import requests
import time
import sys
import os
import torch
import pandas as pd
import json
import numpy as np
import random


# URL do servidor Flask
URL = "http://127.0.0.1:5000/"

TIPO_ALGORITMO = ["cgne", "cgnr"]

TIPO_SINAIS = ["G-1","G-2","G-30x30-1", "G-30x30-2"]

TIPO_SHAPE = {
    "G-1": (60, 60),
    "G-2": (60, 60),
    "G-30x30-1": (30, 30),
    "G-30x30-2": (30, 30)
}


sinais = {
    "G-1": { "name": "G-1", "path": "./client/signal/G-1.csv", "shape": (50816, 1) , "S": 794 , "N": 64},
    "G-2": { "name": "G-2", "path": "./client/signal/G-2.csv", "shape": (50816, 1) , "S": 794 , "N": 64},
    "G-30x30-1": { "name": "G-30x30-1", "path": "./client/signal/g-30x30-1.csv", "shape": (27904, 1) , "S": 436 , "N": 64},
    "G-30x30-2": { "name": "G-30x30-2", "path": "./client/signal/g-30x30-2.csv", "shape": (27904, 1) , "S": 436 , "N": 64}
}

# Configurar o dispositivo (CPU ou GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {device}")


def load_csv_to_tensor(file_path, expected_shape=None, sep=",", device=""):
    data = pd.read_csv(file_path, header=None, sep=sep)
    data = data.apply(pd.to_numeric, errors='coerce').fillna(0)
    tensor = torch.tensor(data.values, dtype=torch.float32, device=device)

    if expected_shape and tensor.shape != expected_shape:
        raise ValueError(f"Arquivo {file_path} tem forma {tensor.shape}, mas esperava {expected_shape}")

    return tensor

def ganho_sinal(sinal):
    g = sinal["tensor"]
    for c in range( sinal["N"]):
        for i in range(sinal["S"]):
            Yi = 100 + 1/20 * i * np.sqrt(i)
            g[i + c * sinal["S"]] = g[i + c * sinal["S"]] * Yi
    return g

def carrega_sinais():
    for key in sinais:
        sinal = sinais[key]
        sinal["tensor"] = load_csv_to_tensor(sinal["path"], sinal["shape"], device=device)
        sinal["tensor"] = ganho_sinal(sinal)

# Faz um hash do sinal para garantir que o sinal não foi alterado
def gerar_checksum(sinal):
    return torch.sum(sinal["tensor"]).item()
    


def processo_enviar_sinal(data, id_processo, tipo_sinal):
    try :
        mkdir = f"./client/processos/{data["nome"]}/{id_processo}"
        os.makedirs(mkdir, exist_ok=True)
        
        sinal = sinais[tipo_sinal]
        
        tamanho_chuncks = len(sinal["tensor"]) // random.randint(50,100)
        chunks = torch.split(sinal["tensor"], tamanho_chuncks)
        for i, chunk in enumerate(chunks):
            data = {
                "sinal": chunk.tolist(),
                "id": id_processo,
                "isLast": i == len(chunks) - 1
            }
            json_data = json.dumps(data)
            response = requests.post(f'{URL}/processo/receber', json=json_data)
            if(response.status_code != 200):
                raise Exception("Erro ao enviar chunk")
            time.sleep(random.uniform(0.01, 0.1))
        # print (f"Enviado sinal {sinal['name']} para o processo {id_processo}")    
    
    except Exception as e:
        try:
            path = os.path.join(".", "client", "processos", data["nome"], id_processo)
            os.system(f"rmdir /s /q {path}")
        except:
            pass
        
        print(f"Erro ao enviar id do processo: {e}")
        return



def main():
    
    # Deleta pasta processos no windows
    try:
        os.system(r"rmdir /s /q .\client\processos")
    except:
        pass
        
    carrega_sinais()
    processos = []
    print("Digite o nome do cliente:")
    nome_cliente = "juliano"# input()
    
    while True:
        print("Escolha sua ação:")
        print("1 - Enviar sinal")
        print("2 - Pedir resultado")
        print("3 - Sair")
        opcao = input()
        if(opcao == "1"):
            tipo_sinal = random.choice(TIPO_SINAIS)
            data = {
                "algoritmo": random.choice(TIPO_ALGORITMO),
                "nome": nome_cliente,
                "checksum": gerar_checksum(sinais[tipo_sinal]),
                "shape": TIPO_SHAPE[tipo_sinal],
            }
            json_data = json.dumps(data)
            response = requests.post(f'{URL}/processo/iniciar', json=json_data)
            if(response.status_code == 200):
                id_processo = response.json()["id_processo"]
                thread = torch.threading.Thread(target=processo_enviar_sinal, args=(data, id_processo, tipo_sinal))
                thread.start()
            elif(response.status_code == 400):
                print("Servidor não aceitou o sinal")
            else:
                print("Erro ao iniciar envio de sinal")
                
        if(opcao == "2"):
            continue
        if(opcao == "3"):
            sys.exit()
    
    
    

if __name__ == "__main__":
    main()
