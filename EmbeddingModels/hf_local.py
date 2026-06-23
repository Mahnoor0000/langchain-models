from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

documents = [
    "Islamabad is the capital of Pakistan",
    "Madrid is the capital of Spain",
    "Paris is the capital of France"
]

vector = embedding.embed_documents(documents)

print(str(vector))