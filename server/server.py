from flask import Flask, request, jsonify
import threading
import torch
import json

app = Flask(__name__)

def process_data(data):
    print(data)
    response = "Sinal recebido com sucesso!"
    return response
    

@app.route('/processar', methods=['POST'])
def handle_client():
    data_str = request.get_json()  # Garante que os dados sejam interpretados como JSON
    if data_str is None:
        return jsonify({"error": "Dados inválidos ou ausentes"}), 400
    data = json.loads(data_str)

    # Cria uma thread para processar os dados de forma assíncrona
    thread = threading.Thread(target=process_data, args=(data,))
    thread.start()
    
    
    return jsonify({"message": "Dados recebidos e em processamento!"}), 202

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=True)
