import gradio as gr
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import os

# Define file paths
pdf_path = "/home/karma/Law-Enforcement/data/Bangladesh_The_Penal_Code_1860.pdf"
vector_store_path = "/home/karma/Law-Enforcement/data/faiss_vector_store"

# Create Embedding Model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load or create FAISS vector store
if os.path.exists(vector_store_path):
    print("✅ Vector store loaded from disk.")
    vector_db = FAISS.load_local(vector_store_path, embeddings=embedding_model, allow_dangerous_deserialization=True)
else:
    print("⚠️ Vector store not found. Rebuilding vector store...")
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    docs = splitter.split_documents(documents)

    print(f"✅ PDF loaded and split into {len(docs)} chunks.")
    vector_db = FAISS.from_documents(docs, embedding=embedding_model)
    vector_db.save_local(vector_store_path)
    print("✅ New vector store created and saved successfully!")

# Load TinyLlama-1.1B
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

# Set up the text generation pipeline
tinyllama_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=100,
    do_sample=False
)

# Optimized retriever with a lower k-value
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

# Function to get answer
def get_answer(query):
    try:
        retrieved_docs = retriever.invoke(query)
        context = " ".join([doc.page_content for doc in retrieved_docs])
        prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
        
        response = tinyllama_pipeline(prompt)
        return response[0]['generated_text'].split("Answer:")[-1].strip()
    
    except Exception as e:
        return f"⚠️ An error occurred: {str(e)}"

# Create Gradio interface
def gradio_interface(query):
    return get_answer(query)

# Define the Gradio interface layout
iface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Textbox(label="Ask a Question About Bangladesh Penal Code", placeholder="Enter your question here..."),
    outputs=gr.Textbox(label="Answer"),
    title="Legal Q&A System",
    description="Ask questions about the Bangladesh Penal Code, and the system will generate answers based on the text of the penal code.",
    theme="default"
)


# Launch the interface
if __name__ == "__main__":
    iface.launch(share=True)