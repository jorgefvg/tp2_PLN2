# Chatbot RAG – Consulta de CV

Este proyecto implementa un **chatbot con RAG (Retrieval-Augmented Generation)** para responder preguntas sobre un Curriculum Vitae.

```
Estructura del proyecto:
tp1/
│
├── upload_pdf.py       ← Carga el CV, aplica chunking, embeddings de HF y sube los vectores a Pinecone
├── chat_rag.py         ← Chatbot RAG con Streamlit para interfaz web
├── environment.yml     ← Creacion del entorno con Conda
└── README.md           
```

El sistema funciona de la siguiente forma:

1. Se sube en formato pdf el **CV** de una persona.
2. Se extrae el texto.
3. Se aplica **chunking** usando LangChain.
4. Cada chunk se convierte en un embedding mediante **HuggingFace / SentenceTransformers**.
5. Los vectores se guardan en **Pinecone**.
6. El chatbot consulta a Pinecone para obtener la pregunta del usuario y el contexto para asi generar una respuestas usando **Groq Llama3**.

---

## Tecnologías utilizadas

- **Chunking:** LangChain – `RecursiveCharacterTextSplitter`
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
- **Vector DB:** Pinecone Serverless
- **LLM:** Groq – Llama 3.3 70B Versatile
- **Frontend:** Streamlit
- **PDF parsing:** PyPDF2

---

## Cómo correr el proyecto

### Prerrequisitos
- Anaconda o miniconda
- Git
- Visual Studio Code

### 1. Configurar variables de entorno:
- Tener una API KEY de base de datos vectoriales. En este proyecto se utilizo el "free tier" de Pinecone (https://www.pinecone.io/).
- Tener una API KEY para utilizar modelos de embeddings y LLMs. En este proyecto se utilizo Groq, que cuenta con una capa gratuita (https://groq.com/)
- Agregar las KEYs a las variables de entorno del sistema operativo para poder llamarlas desde Python.
    
    PINECONE_API_KEY=<TU_CLAVE_PRIVADA>

    GROQ_API_KEY=<TU_CLAVE_PRIVADA>

- Clonar este repositorio o descargarlo.

### 2. Crear y activar el environment:

```bash
conda env create -f environment.yml
conda activate pln-env
```

### 3. Crear los embeddings del CV
```bash
python upload_pdf.py
```
Esto:
- Extrae el texto del CV.
- Genera los chunks
- Crea embeddings con HuggingFace
- Sube todo a Pinecone

### 4. Ejecutar el chatbot
```bash
streamlit run chat_rag.py
```
Aparecerá una interfaz web en donde se pueden hacer preguntas sobre el cv como:

- “¿Como te llamas?”

- “¿Cuál es tu profesion?”

- “¿Cuál es tu experiencia laboral?” etc ....