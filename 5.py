from collections import defaultdict, Counter
import math
reviews = [
    ("fun, couple, love, love", "comedy"),
    ("fast, furious, shoot", "action"),
    ("couple, fly, fast, fun, fun", "comedy"),
    ("furious, shoot, shoot, fun", "action"),
    ("fly, fast, shoot, love", "action")
]

D = "fast, couple, shoot, fly"

def tokenize(text):
    return text.split(", ")

class_docs = defaultdict(list)
vocabulary = set()
class_count = defaultdict(int)

for review, category in reviews:
    tokens = tokenize(review)
    class_docs[category].extend(tokens)
    class_count[category] += 1
    vocabulary.update(tokens)

vocab_size = len(vocabulary)
total_docs = len(reviews)
priors = {category: count / total_docs for category, count in class_count.items()}

likelihoods = {category: {word: (Counter(tokens)[word] + 1) / (len(tokens) + vocab_size) 
                          for word in vocabulary} 
               for category, tokens in class_docs.items()}

tokens = tokenize(D)
posteriors = {category: priors[category] * math.prod([likelihoods[category].get(token, 1 / (len(class_docs[category]) + vocab_size)) 
                  for token in tokens]) for category in priors}

most_likely_class = max(posteriors, key=posteriors.get)
print('Posterior Probability:', posteriors)
print(f"The most likely class for the document '{D}' is: {most_likely_class}")
