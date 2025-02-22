from flask import Flask, request, render_template
import os
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline
from huggingface_hub import login

# Setup and login for HuggingFace API
login("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")

# Prevent Hugging Face symlink warning (Windows fix)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Initialize Flask app
app = Flask(__name__)

# Define a global variable to hold the vector store
vector_store = None

# Load and process PDF once and build vector store
def preprocess_pdf():
    pdf_path = "Bangladesh_The_Penal_Code_1860.pdf"
    pdf = fitz.open(pdf_path)
    text = "\n\n".join(page.get_text("text") for page in pdf)

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)

    # Load embedding model and create vector store
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/msmarco-distilbert-base-tas-b")
    vector_store = FAISS.from_texts(chunks, embedding_model)
    vector_store.save_local("faiss_index")

    return vector_store

# Load the FAISS vector store on startup if it exists
def load_vector_store():
    if os.path.exists("faiss_index"):
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/msmarco-distilbert-base-tas-b")
        return FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
    else:
        return preprocess_pdf()

# Load vector store into the global variable
vector_store = load_vector_store()

# Initialize QA pipeline
qa_pipeline = pipeline("question-answering", model="deepset/tinyroberta-squad2")

# Define route for the web interface
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_query = request.form["query"]
        results = vector_store.similarity_search(user_query, k=3)
        context = " ".join([doc.page_content for doc in results])

        # Generate answer
        input_data = {"question": user_query, "context": context}
        answer = qa_pipeline(**input_data)

        return render_template("index.html", question=user_query, answer=answer["answer"], results=results)
    
    return render_template("index.html", question="", answer="", results=[])

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
