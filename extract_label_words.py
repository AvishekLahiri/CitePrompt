import json
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def find_keywords(model, top_n, doc, anchor):
      n_gram_range = (1, 1)
      stop_words = "english"

      # Extract candidate words/phrases
      count = CountVectorizer(ngram_range=n_gram_range, stop_words=stop_words).fit([doc]) #Convert a collection of text documents to a matrix of token counts.
      candidates = count.get_feature_names()
      
      doc_embedding = model.encode([anchor])
      candidate_embeddings = model.encode(candidates)

      distances = cosine_similarity(doc_embedding, candidate_embeddings)
      keywords = [candidates[index] for index in distances.argsort()[0][-top_n:]]

      return keywords

with open('store_dict.json') as f:
    store_dict = json.load(f)

f1 = open("label_words.txt", "w")

model = SentenceTransformer('')
#anchor_words_dict = {"background": ["background", "introduction"] , "method": ["method", "technique", "procedure"], "result": ["result", "outcome"], "motivation": ["motivation", "introduction"], "uses": ["uses", "evaluation", "method", "results"], "compares": ["compares", "contrasts", "analysis"], "extends": ["extends", "addition"], "future": ["future", "conclusion"]}
#anchor_dict = {"background": ["introduction", "related work", "motivation"] , "method": ["methodology"], "result": ["results"], "motivation": ["introduction"], "uses": ["motivation", "evaluation", "methodology", "results"], "compares": ["results", "discussion", "related work"], "extends": ["methodology", "motivation"], "future": ["conclusion", "discussion"]}
#anchor_words_dict = {"introduction": ["introduction", "background"], "related work": ["related work","motivation", "introduction"] , "method": ["method", "technique", "procedure"], "experiments": ["experiments", "compares", "analysis"], "conclusion": ["conclusion", "future"]}
#anchor_dict = {"introduction": ["introduction", "motivation"], "related work": ["related work", "introduction"], "method": ["methodology", "results"], "experiments": ["methodology", "results", "discussion"], "conclusion": ["conclusion", "discussion"]}
anchor_words_dict = {"cited": ["cited", "reference", "show", "refer"], "not cited": ["introduction", "motivation", "related work", "methodology", "results"] }
anchor_dict = {"cited": ["introduction", "motivation", "related work", "methodology", "results"], "not cited": ["introduction", "motivation", "related work", "methodology", "results"]}

for i in anchor_dict:
      x = anchor_dict[i]
      y = anchor_words_dict[i]
      for j in x:
            for k in y:
                  keywords = find_keywords(model, 100, store_dict[j], k)
                  print(keywords)
                  f1.write(k + ", ")
                  for word in keywords:
                        if word != i:
                             f1.write(word + ", ")
      f1.write("\n")