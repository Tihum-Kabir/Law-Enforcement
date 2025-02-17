# Required Libraries
import os, torch, re
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

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

    # Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    docs = splitter.split_documents(documents)

    print(f"✅ PDF loaded and split into {len(docs)} chunks.")
    vector_db = FAISS.from_documents(docs, embedding=embedding_model)
    vector_db.save_local(vector_store_path)
    print("✅ New vector store created and saved successfully!")

# Load TinyLlama-1.1B model and tokenizer
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

# Enable cuDNN benchmark for speed optimization (If you have GPU)
# torch.backends.cudnn.benchmark = True

# Create text generation pipeline
my_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=100,  
    do_sample=False
)

# Wrap the pipeline in a HuggingFacePipeline object
llm = HuggingFacePipeline(pipeline=my_pipeline)

# Optimized retriever with a lower k-value
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

# Function to get answer
def get_answer(query):
    try:
        # Retrieve documents based on the query
        retrieved_docs = retriever.invoke(query)
        context = " ".join([doc.page_content for doc in retrieved_docs])

        # 1. Handling Empty Context:
        if not context:
            return "No relevant information found in the document."

        # 2. Refining Prompt Structure
        prompt = f"### Instruction: Answer the question based on the context.\n### Context: {context}\n### Question: {query}\n### Answer:"

        # Generate response using your existing pipeline:
        response = my_pipeline(prompt)  

        # 3. Post-processing: Extracting and cleaning the answer:
        answer = response[0]['generated_text'].split("### Answer:")[-1].strip()
        answer = re.sub(r'\s+', ' ', answer)

        # 4. Limiting Response Length
        max_answer_length = 300
        if len(answer) > max_answer_length:
            answer = answer[:max_answer_length] + "..."  # Truncate and add ellipsis

        return answer

    except Exception as e:
        # Handle any exceptions that occur
        return f"An unexpected error occurred: {e}"

# Example query
user_query = "What is the punishment for theft under Bangladesh Penal Code, 1860?"
answer = get_answer(user_query)
print(answer)