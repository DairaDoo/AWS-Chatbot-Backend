import os
from sentence_transformers import SentenceTransformer
import PyPDF2  # O pypdf
from faiss import IndexFlatL2
import numpy as np
import pickle  # Asegúrate de que esté importado
import faiss  # Asegúrate de importar faiss aquí también

# Cargar el modelo de embeddings (¡elige uno pequeño y eficiente!)
model_name = 'sentence-transformers/all-MiniLM-L6-v2'
model = SentenceTransformer(model_name)

# Directorio donde se guardarán los archivos
DATA_DIR = 'data'
INDEX_FILE = 'knowledge_index.faiss'
METADATA_FILE = 'knowledge_metadata.pkl' # Para guardar la referencia a los chunks

knowledge_base = []
embeddings_list = []
metadata_list = []

def load_documents(data_dir=DATA_DIR):
    documents = []
    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)
        if filename.endswith('.txt'):
            with open(filepath, 'r', encoding='utf-8') as f:
                documents.append(f.read())
                metadata_list.append({"source": filename})
                print(f"Cargado texto de: {filename}")
        elif filename.endswith('.pdf'):
            try:
                with open(filepath, 'rb') as pdf_file:
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    num_pages = len(pdf_reader.pages)
                    print(f"Procesando PDF: {filename} con {num_pages} páginas.")
                    for page_num in range(num_pages):
                        page = pdf_reader.pages[page_num]
                        text = page.extract_text()
                        if text.strip():
                            documents.append(text)
                            metadata_list.append({"source": filename, "page": page_num + 1})
                            # print(f"  Extraído texto de la página {page_num + 1}") # Descomentar si quieres ver cada página
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
    k = 3
    distances, indices = knowledge_index.search(np.array(pregunta_embedding), k)
    context = [knowledge_base[i] for i in indices[0]]
    sources = [metadata_list[i] for i in indices[0]]
    respuesta = "Basándome en la siguiente información:\n\n"
    for i, doc in enumerate(context):
        respuesta += f"Fuente: {sources[i]}\n{doc}\n\n"
    respuesta += f"Respuesta a tu pregunta: {pregunta} (Esta es una respuesta simple basada en la información encontrada)."
    return respuesta

# Cargar los documentos, crear embeddings e índice al iniciar el backend
if not os.path.exists(INDEX_FILE):
    documents = load_documents(DATA_DIR)
    if documents:
        knowledge_base = documents
        print("Creando índice...")
        knowledge_index, _ = create_embeddings_and_index(knowledge_base)
        if knowledge_index:
            print("Guardando índice...")
            print(f"Tipo de knowledge_index antes de guardar: {type(knowledge_index)}")
            try:
                with open(INDEX_FILE, 'wb+') as f: # <-- Intento con 'wb+'
                    print(f"Tipo de f: {type(f)}")
                    faiss.write_index(knowledge_index, f)
                    print("Índice guardado exitosamente.")
                with open(METADATA_FILE, 'wb') as f:
                    pickle.dump(metadata_list, f)
                    print("Metadatos guardados exitosamente.")
            except TypeError as e:
                print(f"Error al guardar el índice (TypeError): {e}")
            except Exception as e:
                print(f"Error al guardar el índice (Other): {e}")
        else:
            print("No se pudo crear el índice.")
    else:
        knowledge_index = None
        print("No se encontraron documentos en la carpeta 'data'.")
else:
    import faiss
    import pickle
    print("Cargando índice...")
    try:
        knowledge_index = faiss.read_index(INDEX_FILE)
        with open(METADATA_FILE, 'rb') as f:
            metadata_list = pickle.load(f)
        documents = load_documents(DATA_DIR) # Volver a cargar los documentos
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
                print(f"Tipo de knowledge_index antes de guardar (intento 2): {type(knowledge_index)}")
                try:
                    with open(INDEX_FILE, 'wb+') as f: # <-- Intento con 'wb+' (intento 2)
                        print(f"Tipo de f (intento 2): {type(f)}")
                        faiss.write_index(knowledge_index, f)
                        print("Índice guardado exitosamente (intento 2).")
                    with open(METADATA_FILE, 'wb') as f:
                        pickle.dump(metadata_list, f)
                        print("Metadatos guardados exitosamente (intento 2).")
                except TypeError as e:
                    print(f"Error al guardar el índice (TypeError - intento 2): {e}")
                except Exception as e:
                    print(f"Error al guardar el índice (Other - intento 2): {e}")
            else:
                print("No se pudo crear el índice (intento 2).")
        else:
            print("No se encontraron documentos para crear el índice (intento 2).")


if __name__ == '__main__':
    pregunta_ejemplo = "Dime algo interesante sobre los documentos."
    if knowledge_index:
        respuesta_ejemplo = query_rag_pipeline(pregunta_ejemplo)
        print(f"\nPregunta de prueba: {pregunta_ejemplo}\nRespuesta de prueba: {respuesta_ejemplo}")
    else:
        print("\nEl índice de conocimiento no está disponible para la prueba.")

    # --- Código de prueba para guardar un índice simple ---
    try:
        print("\n--- Prueba de guardado de índice simple ---")
        d = 128  # dimensión del vector
        nb = 1000  # número de vectores de la base de datos
        xq = np.random.random((1, d)).astype('float32')
        index_simple = faiss.IndexFlatL2(d)
        vb = np.random.random((nb, d)).astype('float32')
        index_simple.add(vb)
        print(f"Tipo de índice simple: {type(index_simple)}")
        with open("simple_index.faiss", "wb+") as f_simple:
            print(f"Tipo de f_simple: {type(f_simple)}")
            faiss.write_index(index_simple, f_simple)
            print("Índice simple guardado exitosamente.")
    except TypeError as e:
        print(f"Error al guardar el índice simple (TypeError): {e}")
    except Exception as e:
        print(f"Error al guardar el índice simple (Other): {e}")