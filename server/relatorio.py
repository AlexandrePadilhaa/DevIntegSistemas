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


def plotar_grafico(file_path):
    ## Essa função irá fazer um gráfico com os dados de uso de CPU e memória
    ## que foram coletados pela função inicializar_monitoramento
    ## E criar um arquivo na pasta server/relatorio
    ## com o nome de monitoramento.png
    try:
        df = pd.read_csv(file_path, header=None)
        df.columns = ["datetime", "cpu", "memoria"]
        df["datetime"] = pd.to_datetime(df["datetime"])
        
        fig, ax = plt.subplots(2, 1, figsize=(10, 10))
        ax[0].plot(df["datetime"], df["cpu"], label="CPU (%)")
        ax[0].set_title("Uso de CPU")
        ax[0].set_ylabel("CPU (%)")
        ax[0].set_xlabel("Tempo")
        ax[0].legend()
        
        ax[1].plot(df["datetime"], df["memoria"], label="Memória (%)")
        ax[1].set_title("Uso de Memória")
        ax[1].set_ylabel("Memória (%)")
        ax[1].set_xlabel("Tempo")
        ax[1].legend()
        if not os.path.exists("./server/relatorio"):
            os.mkdir("./server/relatorio")
        plt.tight_layout()
        plt.savefig(f"./server/relatorio/monitoramento.png")
    except Exception as e:
        log(0,f"Erro ao plotar gráfico: {e}")
        raise
    
    
if __name__ == "__main__":
    path = os.path.join(".", "server", "relatorio", "monitoramento.csv")
    plotar_grafico(path)