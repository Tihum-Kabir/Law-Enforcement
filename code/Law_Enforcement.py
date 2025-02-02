# Required Libraries
import fitz  # PyMuPDF
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Extract Text from the PDF
pdf_path = "Bangladesh_The_Penal_Code_1860.pdf"
pdf_doc = fitz.open(pdf_path)

text = ""
for page in pdf_doc:
    text += page.get_text()

# Save extracted text to a .txt file
with open("penal_code.txt", "w", encoding="utf-8") as f:
    f.write(text)

print("✅ Text extraction complete!")

# Preprocess the Extracted Text
loader = TextLoader("penal_code.txt", encoding="utf-8")
documents = loader.load()

# Split text into manageable chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)

print(f"✅ Document split into {len(docs)} chunks.")
