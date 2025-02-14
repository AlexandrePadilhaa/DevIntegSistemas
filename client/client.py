import base64
import datetime
import requests
import time
import sys
import os
import torch
import pandas as pd
import json
import numpy as np
import random
from fpdf import FPDF
from relatorio_geral import plotar_grafico


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
        pasta_processo = f"./client/processos/{data['nome']}/{id_processo}"
        os.makedirs(pasta_processo, exist_ok=True)
        processo_json = os.path.join(pasta_processo, "status.json")
        
        sinal = sinais[tipo_sinal]
        
        tamanho_chuncks = len(sinal["tensor"]) // random.randint(50,100)
        chunks = torch.split(sinal["tensor"], tamanho_chuncks)

        status_processo = {
            "id_processo": id_processo,
            "nome_cliente": data["nome"],
            "tipo_sinal": tipo_sinal,
            "algoritmo": data["algoritmo"],
            "checksum": data["checksum"],
            "shape": data["shape"],
            "status": "enviando",
            "chunks_enviados": 0,
            "total_chunks": len(chunks),
            "data_criacao": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "data_envio": "null",
            "data_resposta": "null"
        }

        # Salvar JSON inicial
        with open(processo_json, "w") as f:
            json.dump(status_processo, f, indent=4)

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
            #print (f"Enviado sinal {sinal['name']} para o processo {id_processo}")  

            status_processo["chunks_enviados"] = i + 1
            with open(processo_json, "w") as f:
                json.dump(status_processo, f, indent=4)  

        status_processo["status"] = "enviado"
        status_processo["data_envio"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with open(processo_json, "w") as f:
            json.dump(status_processo, f, indent=4)

    
    except Exception as e:
        try:
            path = os.path.join(".", "client", "processos", data["nome"], id_processo)
            os.system(f"rmdir /s /q {path}")
        except:
            pass
        
        print(f"Erro ao enviar id do processo: {e}")
        return

def obter_processos_enviados(nome_cliente):
    pasta_cliente = f"./client/processos/{nome_cliente}"
    
    if not os.path.exists(pasta_cliente):
        print("Cliente não encontrado.")
        return []

    processos_enviados = []

    for id_processo in os.listdir(pasta_cliente):
        caminho_json = os.path.join(pasta_cliente, id_processo, "status.json")

        if os.path.exists(caminho_json):
            with open(caminho_json, "r") as f:
                try:
                    status_processo = json.load(f)
                    if status_processo.get("status") == "enviado":
                        processos_enviados.append(status_processo["id_processo"])
                except json.JSONDecodeError:
                    print(f"Erro ao ler JSON do processo {id_processo}")

    print(f"processos: {processos_enviados}")
    return processos_enviados


def salvar_resultados(nome_usuario, processos):
    base_dir = f"client/processos/{nome_usuario}"
    os.makedirs(base_dir, exist_ok=True)

    # Se processos for um dicionário, transformá-lo em uma lista
    if isinstance(processos, dict):
        processos = [processos]

    for processo in processos:
        processo_id = processo.get("id")
        if processo_id:
            processo_dir = os.path.join(base_dir, processo_id)
            os.makedirs(processo_dir, exist_ok=True) 

            caminho_arquivo = os.path.join(processo_dir, f"{processo_id}.json")
            with open(caminho_arquivo, "w", encoding="utf-8") as f:
                json.dump(processo, f, ensure_ascii=False, indent=4)
            print(f"Processo salvo em {caminho_arquivo}")

            # Atualizar status.json
            caminho_status = os.path.join(processo_dir, "status.json")
            status_data = {"data_resposta": None}  # Valor padrão

            if os.path.exists(caminho_status):
                with open(caminho_status, "r", encoding="utf-8") as f:
                    try:
                        status_data = json.load(f)
                    except json.JSONDecodeError:
                        pass  # Se der erro ao ler, mantém o valor padrão

            # Atualiza "data_resposta" 
            status_data["status"] = "recebido"
            status_data["data_resposta"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            
            with open(caminho_status, "w", encoding="utf-8") as f:
                json.dump(status_data, f, ensure_ascii=False, indent=4)
            print(f"Status atualizado em {caminho_status}")

            salvar_base64_como_png(processo.get("imagem"),processo_dir,f"imagem_{processo_id}")
            gerar_relatorio(processo_dir,processo_id)


def salvar_base64_como_png(base64_string, diretorio, nome_arquivo):

    try:
        os.makedirs(diretorio, exist_ok=True)

        caminho_arquivo = os.path.join(diretorio, f"{nome_arquivo}.png")
        imagem_bytes = base64.b64decode(base64_string)
        
        with open(caminho_arquivo, "wb") as imagem_arquivo:
            imagem_arquivo.write(imagem_bytes)
        
        print(f"Imagem salva com sucesso em {caminho_arquivo}")
    except Exception as e:
        print(f"Erro ao salvar a imagem: {e}")


def gerar_relatorio(caminho, id_arquivo):
    """
    Gera um relatório em PDF reunindo as informações dos arquivos JSON e a imagem.
    
    :param caminho: Caminho onde os arquivos estão armazenados
    :param id_arquivo: ID do arquivo a ser processado
    """
    try:
        json_path = os.path.join(caminho, f"{id_arquivo}.json")
        status_path = os.path.join(caminho, "status.json")
        image_path = os.path.join(caminho, f"imagem_{id_arquivo}.png")
        pdf_path = os.path.join(caminho, f"relatorio_{id_arquivo}.pdf")

        # Verificar se os arquivos existem
        if not all(os.path.exists(p) for p in [json_path, status_path, image_path]):
            raise FileNotFoundError("Um ou mais arquivos necessários não foram encontrados.")
        
        # Carregar JSON principal
        with open(json_path, "r", encoding="utf-8") as f:
            dados_json = json.load(f)

        # Carregar JSON de status
        with open(status_path, "r", encoding="utf-8") as f:
            status_json = json.load(f)
        
        # Criar o PDF
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Arial", style="B", size=16)
        pdf.cell(200, 10, "Relatório de Processamento", ln=True, align="C")
        pdf.ln(10)
        
        # Informações do Cliente
        pdf.set_font("Arial", style="B", size=14)
        pdf.cell(200, 10, "Informações do Cliente:", ln=True)
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, f"ID do Processo: {status_json['id_processo']}", ln=True)
        pdf.cell(200, 10, f"Nome do Cliente: {status_json['nome_cliente']}", ln=True)
        pdf.cell(200, 10, f"Tipo de Sinal: {status_json['tipo_sinal']}", ln=True)
        pdf.cell(200, 10, f"Algoritmo: {status_json['algoritmo']}", ln=True)
        pdf.ln(5)
        
        # Informações do Servidor
        pdf.set_font("Arial", style="B", size=14)
        pdf.cell(200, 10, "Informações do Servidor:", ln=True)
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, f"Checksum: {status_json['checksum']}", ln=True)
        pdf.cell(200, 10, f"Shape: {status_json['shape']}", ln=True)
        pdf.cell(200, 10, f"Status: {status_json['status']}", ln=True)
        pdf.cell(200, 10, f"Chunks Enviados: {status_json['chunks_enviados']}/{status_json['total_chunks']}", ln=True)
        pdf.cell(200, 10, f"Data de Criação: {status_json['data_criacao']}", ln=True)
        pdf.cell(200, 10, f"Data de Envio: {status_json['data_envio']}", ln=True)
        pdf.cell(200, 10, f"Data de Resposta: {status_json['data_resposta']}", ln=True)
        pdf.ln(5)
        
        # Resultados
        pdf.set_font("Arial", style="B", size=14)
        pdf.cell(200, 10, "Resultados:", ln=True)
        pdf.set_font("Arial", size=12)
        resultado = dados_json['resultado']
        pdf.cell(200, 10, f"Número de Iterações: {resultado['numero_iteracoes']}", ln=True)
        pdf.cell(200, 10, f"Shape: {resultado['shape']}", ln=True)
        pdf.cell(200, 10, f"Tempo de Início: {resultado['tempo']['inicio']}", ln=True)
        pdf.cell(200, 10, f"Tempo de Fim: {resultado['tempo']['fim']}", ln=True)
        pdf.cell(200, 10, f"Tempo Total: {resultado['tempo']['total_segundos']} segundos", ln=True)
        pdf.ln(10)
        
        # Adicionar imagem ao PDF
        pdf.set_font("Arial", style="B", size=14)
        pdf.cell(200, 10, "Imagem Resultado:", ln=True, align="C")
        pdf.ln(5)
        img_width = pdf.w / 2
        img_x = (pdf.w - img_width) / 2
        pdf.image(image_path, x=img_x, w=img_width)
        
        
        # Salvar PDF
        pdf.output(pdf_path)
        print(f"Relatório gerado com sucesso: {pdf_path}")
    except Exception as e:
        print(f"Erro ao gerar relatório: {e}")

def baixar_relatorio_geral(response, destino):
    if response.status_code == 200:
        with open(destino, "wb") as f:
            f.write(response.content)
        print(f"Arquivo salvo em {destino}")
        gerar_relatorio_geral()
    else:
        erro_msg = f"Erro ao baixar o relatório. Código: {response.status_code}, Resposta: {response.text}"
        print(erro_msg)

def gerar_relatorio_geral():
    plotar_grafico("./client/monitoramento/monitoramento.csv")
    
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", style='B', size=16)
    pdf.cell(200, 10, "Monitoramento de CPU e Memória", ln=True, align='C')
    pdf.ln(10)
    
    caminho_imagem = "./client/monitoramento/monitoramento.png"
    caminho_pdf = "./client/monitoramento/relatorio_monitoramento.pdf"

    if os.path.exists(caminho_imagem):
        pdf.image(caminho_imagem, x=10, y=None, w=190)
    else:
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, "Erro: Imagem não encontrada", ln=True, align='C')
    
    pdf.output(caminho_pdf)
    print(f"Relatório salvo em: {caminho_pdf}")



def main():
    
    # Deleta pasta processos no windows
    try:
        os.system(r"rmdir /s /q .\client\processos")
    except:
        pass
        
    carrega_sinais()
    print("Digite o nome do cliente:")
    nome_cliente = "alexandre"#input()
    
    while True:
        print("Escolha sua ação:")
        print("1 - Enviar sinal")
        print("2 - Pedir resultado")
        print("3 - Gerar relatório geral")
        print("4 - Sair")
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
                
        elif opcao == "2":
            resultados = obter_processos_enviados(nome_cliente)
            print("Processos disponíveis:")
            for res in resultados:
                print(f"ID: {res}")
            
            print("Escolha uma opção:")
            print("1 - Receber todos os resultados")
            print("2 - Escolher um processo específico")
            sub_opcao = input()
            
            if sub_opcao == "1":
                print("Resultados completos:")
                response = requests.get(f"{URL}/processo/enviar_resultado", params={"nome": nome_cliente})
                if response.status_code == 200:
                    resultados = response.json()
                    salvar_resultados(nome_cliente,resultados)
                    print("Resultado dos processos salvos")
                else:
                    print("Nenhum resultado encontrado ou erro ao buscar resultados.")
            elif sub_opcao == "2":
                print("Digite o ID do processo desejado:")
                id_escolhido = input()
                response = requests.get(f"{URL}/processo/enviar_resultado_id", params={"id": id_escolhido, "nome": nome_cliente})
                    
                if response.status_code == 200:
                    processo = response.json()
                    salvar_resultados(nome_cliente,processo)
                    print("Resultado do processo salvo")
                elif response.status_code == 403:
                    print("Erro: Este processo não pertence a este usuário.")
                else:
                    print("ID não encontrado ou erro ao buscar resultado.")
        
        elif opcao == "3":
            response = requests.get(f"{URL}/processo/relatorio")
            baixar_relatorio_geral(response, "./client/monitoramento/monitoramento.csv")

        elif opcao == "4":
            sys.exit()
        else:
            print("Opção inválida. Tente novamente.")
    

if __name__ == "__main__":
    main()
