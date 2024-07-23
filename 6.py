import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial.distance import euclidean, cityblock
from sklearn.metrics.pairwise import cosine_similarity

documents = [
    "Shipment of gold damaged in a fire",
    "Delivery of silver arrived in a silver truck",
    "Shipment of gold arrived in a truck",
    "Purchased silver and gold arrived in a wooden truck",
    "The arrival of gold and silver shipment is delayed."
]

query = "gold silver truck"

vectorizer = CountVectorizer(stop_words="english")
X = vectorizer.fit_transform(documents + [query]).toarray()

doc_vectors, query_vector = X[:-1], X[-1]

euclidean_distances = [euclidean(doc, query_vector) for doc in doc_vectors]
manhattan_distances = [cityblock(doc, query_vector) for doc in doc_vectors]
cosine_similarities = cosine_similarity(doc_vectors, query_vector.reshape(1, -1)).flatten()

euclidean_ranking = np.argsort(euclidean_distances)[:2] + 1
manhattan_ranking = np.argsort(manhattan_distances)[:2] + 1
cosine_ranking = np.argsort(-cosine_similarities)[:2] + 1

print("Euclidean Distance:", euclidean_distances)
print("Manhattan Distance:", manhattan_distances)
print("Cosine Similarity:", cosine_similarities)

print("\nTop 2 documents using Euclidean distance:", euclidean_ranking)
print("Top 2 documents using Manhattan distance:", manhattan_ranking)
print("Top 2 documents using Cosine similarity:", cosine_ranking)
