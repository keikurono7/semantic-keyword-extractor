from flask import Flask, request, jsonify
import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util
import spacy
import numpy as np

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

app = Flask(__name__)

stop_words = set(stopwords.words('english'))
nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer('all-MiniLM-L6-v2')


def extract_candidates(sentence, phrase_n=4, custom_stopwords=None):
    filtered_words = stop_words.copy()
    if custom_stopwords:
        filtered_words.update(custom_stopwords)

    tokens = word_tokenize(sentence.lower())
    phrases = []
    for n in range(1, phrase_n + 1):
        for gram in ngrams(tokens, n):
            if not any(word in filtered_words for word in gram):
                phrase = " ".join(gram)
                phrases.append(phrase)

    doc = nlp(sentence)
    entities = [ent.text.strip() for ent in doc.ents
                if ent.label_ in ["DATE", "TIME", "PERSON", "ORG", "GPE", "EVENT", "PRODUCT"]]

    combined = list(set(entities + phrases))
    return [p for p in combined if len(p) > 2 and p.lower() not in filtered_words]


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    sentence = data.get("sentence", "")
    keywords = data.get("keywords", [])
    phrase_n = data.get("phrase_n", 3)

    candidates = extract_candidates(sentence, phrase_n)
    if not candidates:
        return jsonify({"matches": [], "confidence": []})

    candidate_embeddings = model.encode(candidates, convert_to_tensor=True)
    results = []
    confidences = []

    for keyword in keywords:
        keyword_embedding = model.encode(keyword, convert_to_tensor=True)
        similarities = util.cos_sim(keyword_embedding, candidate_embeddings)[0]
        best_idx = similarities.argmax().item()
        best_score = similarities[best_idx].item()

        results.append(candidates[best_idx])
        confidences.append(round(best_score, 4))

    return jsonify({
        "matches": dict(zip(keywords, results)),
        "confidence": dict(zip(keywords, confidences))
    })

@app.route("/test", methods=["GET"])
def test():
    sentence = "Despite the rain, Tesla announced a new car for 19 March 2026."
    keywords = ["car", "company", "future date"]

    phrase_n = 3
    candidates = extract_candidates(sentence, phrase_n)
    if not candidates:
        return jsonify({"matches": [], "confidence": []})

    candidate_embeddings = model.encode(candidates, convert_to_tensor=True)
    results = []
    confidences = []

    for keyword in keywords:
        keyword_embedding = model.encode(keyword, convert_to_tensor=True)
        similarities = util.cos_sim(keyword_embedding, candidate_embeddings)[0]
        best_idx = similarities.argmax().item()
        best_score = similarities[best_idx].item()

        results.append(candidates[best_idx])
        confidences.append(round(best_score, 4))

    return jsonify({
        "sentence": sentence,
        "keywords": keywords,
        "matches": dict(zip(keywords, results)),
        "confidence": dict(zip(keywords, confidences))
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
