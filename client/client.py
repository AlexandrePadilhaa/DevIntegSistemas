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
URL = "http://127.0.0.1:5000/processar"

TIPO_ALGORITMO = ["cgne", "cgnr"]

TIPO_SINAIS = ["G-30x30-1"]

sinais = {
    #"G-1": { "name": "G-1", "path": "./client/signal/G-1.csv", "shape": (50816, 1) , "S": 794 , "N": 64},
    "G-30x30-1": { "name": "G-30x30-1", "path": "./client/signal/g-30x30-1.csv", "shape": (27904, 1) , "S": 436 , "N": 64}
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

# Função que simula uma sequência de sinais
def enviar_sinal(sinal , algoritmo):
    array_sinal = sinal["tensor"].tolist()
    data = {
        "algoritmo": algoritmo,
        "shape": sinal["shape"],
        "sinal": array_sinal
    }
    json_data = json.dumps(data)
    response = requests.post(URL, json=json_data)
    return response.json()

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

        


def main():
    carrega_sinais()
    
    for i in range(1):
        tipo_sinal = random.choice(TIPO_SINAIS)
        tipo_algoritmo = random.choice(TIPO_ALGORITMO)
        response = enviar_sinal(sinais[tipo_sinal], tipo_algoritmo)
        print(response)
    
    
    

if __name__ == "__main__":
    main()
