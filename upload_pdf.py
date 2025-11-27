import os
import uuid
import PyPDF2
from pinecone import Pinecone, ServerlessSpec
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# Configuracion (las APY keys se agregan al path de windows)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "cv-rag-index")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1")

# Embeddings (HuggingFace)

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Configuracion Pinecone

pc = Pinecone(api_key=PINECONE_API_KEY)

if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=384,     # MiniLM-L6-v2
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
    )
    print("Índice creado:", PINECONE_INDEX_NAME)

index = pc.Index(PINECONE_INDEX_NAME)

# Funciones

def extract_text_from_pdf(path):
    reader = PyPDF2.PdfReader(path)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content + "\n"
    return text


def create_chunks(text):
    """
    Chunking inteligente usando LangChain.
    Obtiene bloques coherentes sin mezclar secciones.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        separators=[
            "\n## ", "\n### ", "\n- ", "\n•",
            "\n", ". ", " "
        ]
    )
    return splitter.split_text(text)

# Main

def upload_pdf(pdf_path):
    print("Extrayendo texto del PDF...")
    text = extract_text_from_pdf(pdf_path)

    print("Creando chunks inteligentes...")
    chunks = create_chunks(text)
    print(f"Generados {len(chunks)} chunks correctamente")

    print("Generando embeddings y subiendo a Pinecone...")
    vectors = []

    for ch in chunks:
        emb = embedder.encode(ch).tolist()
        vectors.append({
            "id": str(uuid.uuid4()),
            "values": emb,
            "metadata": {"text": ch}
        })

    index.upsert(vectors=vectors)
    print("Upload completado. Chunks y embeddings almacenados en Pinecone.")


if __name__ == "__main__":
    upload_pdf("Jorge_cv.pdf")
