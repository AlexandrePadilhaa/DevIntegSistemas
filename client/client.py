import requests
import time

# URL do servidor Flask
url = "http://127.0.0.1:5000/processar"

# Função que simula uma sequência de sinais
def enviar_sinais():
    sinais = [
        "Sinal 1",
        "Sinal 2",
        "Sinal 3"
    ]
    
    for sinal in sinais:
        # Envia o sinal para o servidor
        response = requests.post(url, data=sinal)
        
        # Verifica a resposta do servidor
        if response.status_code == 202:
            print(f"Sinal enviado: {sinal}")
            print("Resposta do servidor:", response.json())
        else:
            print(f"Falha ao enviar sinal: {sinal}")
        
        time.sleep(3)

if __name__ == "__main__":
    enviar_sinais()
