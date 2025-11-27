import os
import streamlit as st
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from groq import Groq

# Configuracion

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "cv-rag-index")

LLM_MODEL = "llama-3.3-70b-versatile"

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
groq_client = Groq(api_key=GROQ_API_KEY)

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# RAG

def rag_query(question):
    q_emb = embedder.encode(question).tolist()

    results = index.query(
        vector=q_emb,
        top_k=5,
        include_metadata=True
    )

    context = "\n\n".join([match["metadata"]["text"] for match in results["matches"]])

    prompt = f"""
Responde únicamente con la información del CV. 
Si no está en el CV, responde: "No se encuentra en el CV".

### CONTEXTO
{context}

### PREGUNTA
{question}

### RESPUESTA
"""

    response = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


# Streamlit

st.title("Chatbot RAG — Consulta el CV")

question = st.text_input("Pregunta sobre el CV:")

if st.button("Enviar"):
    if question.strip():
        answer = rag_query(question)
        st.write("### Respuesta:")
        st.write(answer)
