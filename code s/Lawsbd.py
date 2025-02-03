import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter

pdf = fitz.open("Bangladesh_The_Penal_Code_1860.pdf")  # Opening the PDF


text = "\n\n".join(page.get_text("text") for page in pdf)  # Extract

# Splitting into smaller chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_text(text)

print(f"Total chunks: {len(chunks)}")

# In need of printing chunks
print_chunks = False  # Change to True if you wanna print

if print_chunks:
    for idx, chunk in enumerate(chunks):
        print(f"Chunk {idx + 1}:")
        print(chunk)
        print("\n" + "=" * 50 + "\n")  # Separate chunks visually
    