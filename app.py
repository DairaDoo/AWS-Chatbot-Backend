from flask import Flask, request, jsonify
from flask_swagger_ui import get_swaggerui_blueprint
from rag_pipeline import query_rag_pipeline
from flask_cors import CORS

app = Flask(__name__)

### Swagger specific ###
SWAGGER_URL = '/swagger'
API_URL = '/static/swagger.json'  # Define la ruta a tu archivo swagger.json
CORS(app)

swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "RAG Chatbot API"
    }
)

app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)
### end Swagger specific ###

@app.route('/consultar', methods=['POST'])
def consultar():
    if 'pregunta' not in request.json or not request.json['pregunta']:
        return jsonify({"error": "La pregunta es obligatoria"}), 400

    pregunta = request.json['pregunta']
    respuesta = query_rag_pipeline(pregunta)  # Llama a tu pipeline RAG
    return jsonify({"respuesta": respuesta})

if __name__ == '__main__':
    app.run(debug=True)