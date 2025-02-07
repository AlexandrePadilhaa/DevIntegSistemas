from flask import Flask, request, jsonify
import threading

app = Flask(__name__)

def process_data(data):
    print(f"Dados recebidos: {data}")
    response = "Sinal recebido com sucesso!"
    return response

@app.route('/processar', methods=['POST'])
def handle_client():
    data = request.get_data(as_text=True)
    print(f"Dados recebidos: {data}")
    
    # Cria uma thread para processar os dados de forma assíncrona
    thread = threading.Thread(target=process_data, args=(data,))
    thread.start()
    
    # Retorna uma resposta imediatamente
    return jsonify({"message": "Dados recebidos e em processamento!"}), 202

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=True)
