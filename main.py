import os
os.environ['NLTK_DATA'] = '/usr/local/nltk_data'

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional, Set
import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util
import spacy
import numpy as np
from functools import lru_cache

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

app = FastAPI()

# Initialize resources
stop_words = set(stopwords.words('english'))
nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Entity types we're interested in
RELEVANT_ENTITY_TYPES = {"DATE", "TIME", "PERSON", "ORG", "GPE", "EVENT", "PRODUCT"}

class AnalysisRequest(BaseModel):
    sentence: str
    keywords: List[str]
    phrase_n: int = 3
    custom_stopwords: Optional[List[str]] = None

class AnalysisResponse(BaseModel):
    matches: Dict[str, str]
    confidence: Dict[str, float]

class TestResponse(BaseModel):
    sentence: str
    keywords: List[str]
    candidates: List[str]
    matches: Dict[str, str]
    confidence: Dict[str, float]

@lru_cache(maxsize=128)
def get_model_embedding(text):
    """Cache embeddings to avoid recomputing for the same text"""
    return model.encode(text, convert_to_tensor=True)

def extract_candidates(sentence: str, phrase_n: int = 4, custom_stopwords: Optional[Set[str]] = None):
    filtered_words = stop_words.copy()
    if custom_stopwords:
        filtered_words.update(custom_stopwords)

    doc = nlp(sentence)
    
    # Merge multi-word entities and chunks
    with doc.retokenize() as retokenizer:
        for ent in doc.ents:
            retokenizer.merge(ent)
        for chunk in doc.noun_chunks:
            retokenizer.merge(chunk)

    # Extract useful entities + noun phrases
    entities = [ent.text.strip() for ent in doc.ents if ent.label_ in RELEVANT_ENTITY_TYPES]
    noun_chunks = [chunk.text.strip() for chunk in doc.noun_chunks if len(chunk.text.split()) <= phrase_n]

    candidates = list(set(entities + noun_chunks))
    return [c for c in candidates if len(c) > 2 and c.lower() not in filtered_words]


def find_best_matches(candidates: List[str], keywords: List[str]):
    """Find the best matching candidate for each keyword"""
    if not candidates:
        return {}, {}
    
    # Encode all candidates at once
    candidate_embeddings = model.encode(candidates, convert_to_tensor=True)
    
    matches = {}
    confidences = {}
    
    for keyword in keywords:
        keyword_embedding = get_model_embedding(keyword)
        similarities = util.cos_sim(keyword_embedding, candidate_embeddings)[0]
        best_idx = similarities.argmax().item()
        best_score = similarities[best_idx].item()
        
        matches[keyword] = candidates[best_idx]
        confidences[keyword] = round(best_score, 4)
    
    return matches, confidences

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(request: AnalysisRequest):
    if not request.sentence or not request.keywords:
        raise HTTPException(status_code=400, detail="Missing required parameters")
    
    custom_stopwords_set = set(request.custom_stopwords) if request.custom_stopwords else None
    candidates = extract_candidates(request.sentence, request.phrase_n, custom_stopwords_set)
    matches, confidences = find_best_matches(candidates, request.keywords)
    
    return AnalysisResponse(matches=matches, confidence=confidences)

@app.get("/test", response_model=TestResponse)
async def test():
    sentence = "Despite the rain, Tesla announced a new car for 19 March 2026."
    keywords = ["car", "company", "future date"]
    phrase_n = 3
    
    candidates = extract_candidates(sentence, phrase_n)
    matches, confidences = find_best_matches(candidates, keywords)
    
    return TestResponse(
        sentence=sentence,
        keywords=keywords,
        candidates=candidates,
        matches=matches,
        confidence=confidences
    )
