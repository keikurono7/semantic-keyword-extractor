from fastapi import FastAPI, Request
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
from typing import List, Optional
import spacy
import uvicorn

app = FastAPI()

nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer('all-MiniLM-L6-v2')

class AnalyzeRequest(BaseModel):
    sentence: str
    keywords: List[str]
    phrase_n: Optional[int] = 3
    custom_stopwords: Optional[List[str]] = []

def extract_candidates(text, phrase_n=4, custom_stopwords=None):
    doc = nlp(text.lower())
    stop_words = set(spacy.lang.en.stop_words.STOP_WORDS)
    if custom_stopwords:
        stop_words.update(custom_stopwords)

    tokens = [token.text for token in doc if token.is_alpha and token.text not in stop_words]
    phrases = []

    for n in range(1, phrase_n + 1):
        n_grams = zip(*[tokens[i:] for i in range(n)])
        for gram in n_grams:
            phrase = " ".join(gram)
            if phrase not in stop_words:
                phrases.append(phrase)

    entities = [ent.text.strip() for ent in doc.ents if ent.label_ in ["DATE", "TIME", "PERSON", "ORG", "GPE", "EVENT", "PRODUCT"]]
    combined = list(set(phrases + entities))
    return [p for p in combined if len(p) > 2]

@app.get("/")
async def root():
    return {"message": "API is live"}

@app.post("/analyze")
async def analyze(request: AnalyzeRequest):
    sentence = request.sentence
    keywords = request.keywords
    phrase_n = request.phrase_n
    custom_stopwords = request.custom_stopwords

    candidates = extract_candidates(sentence, phrase_n, custom_stopwords)
    if not candidates:
        return {"matches": {}, "confidence": {}}

    candidate_embeddings = model.encode(candidates, convert_to_tensor=True)
    matches = {}
    confidences = {}

    for keyword in keywords:
        keyword_embedding = model.encode(keyword, convert_to_tensor=True)
        similarities = util.cos_sim(keyword_embedding, candidate_embeddings)[0]
        best_idx = similarities.argmax().item()
        best_score = similarities[best_idx].item()

        matches[keyword] = candidates[best_idx]
        confidences[keyword] = round(best_score, 4)

    return {"matches": matches, "confidence": confidences}

# Run this in local dev
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
