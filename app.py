from flask import Flask, request, jsonify
from rag_pipeline import query_rag_pipeline

app = Flask(__name__)

@app.route('/consultar', methods=['POST'])
def consultar():
    if 'pregunta' not in request.json or not request.json['pregunta']:
        return jsonify({"error": "La pregunta es obligatoria"}), 400

    pregunta = request.json['pregunta']
    respuesta = query_rag_pipeline(pregunta)  # Llama a tu pipeline RAG
    return jsonify({"respuesta": respuesta})

if __name__ == '__main__':
    app.run(debug=True)