import os
from sentence_transformers import SentenceTransformer
import PyPDF2
from faiss import IndexFlatL2
import numpy as np
import pickle
import faiss
import requests
from dotenv import load_dotenv  # Importa la función load_dotenv

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

# Cargar el modelo de embeddings
model_name = "all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)

# Directorio donde se guardarán los archivos
DATA_DIR = "data"
INDEX_FILE = "knowledge_index.faiss"
METADATA_FILE = "knowledge_metadata.pkl"

knowledge_base = []
embeddings_list = []
metadata_list = []

# --- OpenRouter Configuration ---
OPENROUTER_API_KEY = os.getenv(
    "OPENROUTER_API_KEY"
)  # Obtén la API key desde las variables de entorno
MODEL_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "mistralai/mistral-7b-instruct:free"

headers = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "HTTP-Referer": "mi-aws-chatbot",  # Reemplaza con tu aplicación
}


def query_llm(prompt):
    if not OPENROUTER_API_KEY:
        print("Error: La variable de entorno OPENROUTER_API_KEY no está configurada.")
        return "Error: API key no configurada."
    payload = {"model": MODEL_NAME, "messages": [{"role": "user", "content": prompt}]}
    try:
        response = requests.post(MODEL_ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        print(f"Error querying OpenRouter API: {e}")
        return "No se pudo obtener una respuesta del modelo LLM."
    except (KeyError, IndexError, TypeError) as e:
        print(f"Error parsing OpenRouter API response: {e}")
        print(f"Response content: {response.text}")
        return "Error al procesar la respuesta del modelo LLM."


def load_documents(data_dir=DATA_DIR):
    documents = []
    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)
        if filename.endswith(".txt"):
            with open(filepath, "r", encoding="utf-8") as f:
                documents.append(f.read())
                metadata_list.append({"source": filename})
                print(f"Cargado texto de: {filename}")
        elif filename.endswith(".pdf"):
            try:
                with open(filepath, "rb") as pdf_file:
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    num_pages = len(pdf_reader.pages)
                    print(f"Procesando PDF: {filename} con {num_pages} páginas.")
                    for page_num in range(num_pages):
                        page = pdf_reader.pages[page_num]
                        text = page.extract_text()
                        if text.strip():
                            documents.append(text)
                            metadata_list.append(
                                {"source": filename, "page": page_num + 1}
                            )
            except Exception as e:
                print(f"Error al leer {filename}: {e}")
    print(f"Total de documentos cargados: {len(documents)}")
    return documents


def create_embeddings_and_index(documents):
    print(f"Número de documentos para crear embeddings: {len(documents)}")
    if not documents:
        return None, []
    embeddings = model.encode(documents)
    print(f"Shape de los embeddings creados: {embeddings.shape}")
    index = IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    print("Embeddings añadidos al índice.")
    return index, documents


def query_rag_pipeline(pregunta):
    if knowledge_index is None:
        return "El índice de conocimiento no se ha cargado."
    pregunta_embedding = model.encode([pregunta])
    k = 5
    distances, indices = knowledge_index.search(np.array(pregunta_embedding), k)
    context = [knowledge_base[i] for i in indices[0]]
    sources = [metadata_list[i] for i in indices[0]]

    prompt = f"Basándote principalmente en la siguiente información:\n\n"
    for i, doc in enumerate(context):
        prompt += f"Fuente: {sources[i]}\n{doc}\n\n"
    prompt += f"Responde a la siguiente pregunta de forma natural y concisa: {pregunta}"

    llm_response = query_llm(prompt)
    return llm_response


# Cargar los documentos, crear embeddings e índice al iniciar el backend
if not os.path.exists(INDEX_FILE):
    documents = load_documents(DATA_DIR)
    if documents:
        knowledge_base = documents
        print("Creando índice...")
        knowledge_index, _ = create_embeddings_and_index(knowledge_base)
        if knowledge_index:
            print("Guardando índice...")
            try:
                faiss.write_index(knowledge_index, INDEX_FILE)
                print("Índice guardado exitosamente.")
                with open(METADATA_FILE, "wb") as f:
                    pickle.dump(metadata_list, f)
                    print("Metadatos guardados exitosamente.")
            except Exception as e:
                print(f"Error al guardar el índice: {e}")
        else:
            print("No se pudo crear el índice.")
    else:
        knowledge_index = None
        print("No se encontraron documentos en la carpeta 'data'.")
else:
    print("Cargando índice...")
    try:
        knowledge_index = faiss.read_index(INDEX_FILE)
        with open(METADATA_FILE, "rb") as f:
            metadata_list = pickle.load(f)
        documents = load_documents(DATA_DIR)
        knowledge_base = documents
        print("Índice cargado exitosamente.")
    except Exception as e:
        knowledge_index = None
        print(f"Error al cargar el índice: {e}")
        print("Se intentará crear el índice nuevamente.")
        documents = load_documents(DATA_DIR)
        if documents:
            knowledge_base = documents
            print("Creando índice...")
            knowledge_index, _ = create_embeddings_and_index(knowledge_base)
            if knowledge_index:
                print("Guardando índice...")
                try:
                    faiss.write_index(knowledge_index, INDEX_FILE)
                    print("Índice guardado exitosamente.")
                    with open(METADATA_FILE, "wb") as f:
                        pickle.dump(metadata_list, f)
                        print("Metadatos guardados exitosamente.")
                except Exception as e:
                    print(f"Error al guardar el índice: {e}")
            else:
                print("No se pudo crear el índice.")
        else:
            print("No se encontraron documentos para crear el índice.")
