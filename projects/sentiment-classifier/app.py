import json
import contractions
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException, Request
from nltk.tokenize import RegexpTokenizer
from models import classifier
from pydantic import BaseModel

import torch
import torch.nn.functional as F


app = FastAPI(title='Sentiment Classifier')

class TextHandler:
    def __init__(self, token_to_id: Dict[str, int]):
        self.tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
        self.token_to_id = token_to_id
        self.tokens = set(token_to_id)

    def preprocess(self, text: str) -> List[str]:
        if not isinstance(text, str):
            raise ValueError("Text must be a string")
        
        text = text.lower()
        text = contractions.fix(text)
        text = self.tokenizer.tokenize(text)
        return text

    def convert_text_to_tensor(self, text: str) -> torch.Tensor:
        unknown_token = '[UNK]'
        text = self.preprocess(text)

        tokens_ids = []
        for token in text:
            if token in self.tokens:
                token_id = self.token_to_id[token]
            else:
                token_id = self.token_to_id[unknown_token]
            tokens_ids.append(token_id)
        text_tensor = torch.tensor([tokens_ids])
        return text_tensor
    
class PredictRequest(BaseModel):
    text: Optional[str] = None

class PredictResponse(BaseModel):
    prediction: str

def load_json(path: str):
    with open(path, 'r', encoding='utf-8') as json_file:
        return json.load(json_file)

def load_model(model_path: str) -> torch.nn.Module:
    device = torch.device('cpu')
    model = classifier
    model.load_state_dict(torch.load('model.pth', map_location=device, weights_only=True))
    model.eval()
    return model

def load_vocabulary(vocabulary_path: str) -> Dict[str, int]:
    vocabulary = load_json(vocabulary_path)
    token_to_id_map = vocabulary['token_to_id']
    return token_to_id_map

@app.post('/predict')
async def predict_sentiment(request: PredictRequest) -> PredictResponse:
    id_to_label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    text = request.text

    if not text:
        raise HTTPException(status_code=400, detail='Provide the text')
    try:
        text_tensor = text_handler.convert_text_to_tensor(text)
    except Exception as exception:
        raise HTTPException(status_code=400, detail=f'Preprocess error: {exception}')
    
    with torch.no_grad():
        logits = model(text_tensor)

    prediction_id = torch.argmax(logits).item()
    prediction = id_to_label_map[prediction_id]
    return PredictResponse(prediction=prediction)

vocabulary_path = './vocabulary.json'
model_path = './models/classifier.pth'

model = load_model(model_path)
token_to_id = load_vocabulary(vocabulary_path)

text_handler = TextHandler(token_to_id)
