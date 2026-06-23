from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load local embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

documents = [
    "Breaking Bad follows Walter White, a chemistry teacher who turns into a drug kingpin after being diagnosed with cancer.",
    "Stranger Things is set in the 1980s and follows a group of kids facing supernatural events and secret government experiments.",
    "Game of Thrones is a fantasy series filled with political conflict, dragons, and the battle for the Iron Throne.",
    "Friends is a sitcom about six friends living in New York City, focusing on their relationships, careers, and daily life humor.",
    "Sherlock is a modern adaptation of Sherlock Holmes, showcasing his brilliant detective skills in solving complex crimes."
]

query = "tell me about enola holmes"

# Create embeddings (LOCAL)
doc_embeddings = embedding_model.encode(documents)
query_embedding = embedding_model.encode([query])

# Compute similarity
scores = cosine_similarity(query_embedding, doc_embeddings)[0]

# Get best match
index, score = sorted(list(enumerate(scores)), key=lambda x: x[1])[-1]

print("Query:", query)
print("Best Match:", documents[index])
print("Similarity score:", score)