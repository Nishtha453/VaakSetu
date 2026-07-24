from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import hf_hub_download

app = FastAPI(title="VaakSetu API", description="Hinglish NLU for Indian E-commerce")

INTENT2ID = {
    "SEARCH": 0, "COMPARE": 1, "BUY": 2, "TRACK": 3, "RETURN": 4
}
ID2INTENT = {v: k for k, v in INTENT2ID.items()}
NER2ID = {
    "O": 0,
    "B-PRODUCT": 1, "I-PRODUCT": 2,
    "B-BRAND": 3, "I-BRAND": 4,
    "B-ATTRIBUTE": 5, "I-ATTRIBUTE": 6,
    "B-PRICE_RANGE": 7, "I-PRICE_RANGE": 8,
    "B-SIZE": 9, "I-SIZE": 10,
    "B-DELIVERY_CONSTRAINT": 11, "I-DELIVERY_CONSTRAINT": 12
}
ID2NER = {v: k for k, v in NER2ID.items()}

class VaakSetuModel(nn.Module):
    def __init__(self, num_intents=5, num_ner_labels=13):
        super().__init__()
        self.muril = AutoModel.from_pretrained("google/muril-base-cased")
        hidden_size = self.muril.config.hidden_size
        self.intent_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_intents)
        )
        self.ner_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_ner_labels)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.muril(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_output = outputs.last_hidden_state[:, 0, :]
        sequence_output = outputs.last_hidden_state
        intent_logits = self.intent_head(cls_output)
        ner_logits = self.ner_head(sequence_output)
        return intent_logits, ner_logits

# Load model once at startup
device = torch.device("cpu")
tokenizer = None
model = None

@app.on_event("startup")
async def load_model():
    global tokenizer, model
    model_path = hf_hub_download(
        repo_id="Nishtha555/VaakSetu",
        filename="vaaksetu_final_model.pt"
    )
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")
    model = VaakSetuModel().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

class QueryRequest(BaseModel):
    query: str

class EntityResponse(BaseModel):
    text: str
    type: str

class PredictResponse(BaseModel):
    query: str
    intent: str
    entities: list[EntityResponse]

@app.get("/")
def root():
    return {"message": "VaakSetu API", "demo": "https://vaaksetu.streamlit.app"}

@app.post("/predict", response_model=PredictResponse)
def predict(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    words = request.query.strip().split()

    encoding = tokenizer(
        words,
        is_split_into_words=True,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        intent_logits, ner_logits = model(
            encoding["input_ids"],
            encoding["attention_mask"]
        )

    intent_id = torch.argmax(intent_logits, dim=1).item()
    intent = ID2INTENT[intent_id]

    word_ids = encoding.word_ids(batch_index=0)
    ner_predictions = torch.argmax(ner_logits, dim=2).squeeze(0)

    entities = []
    current_entity = None
    current_type = None

    for idx, word_id in enumerate(word_ids):
        if word_id is None or word_id >= len(words):
            continue
        label = ID2NER[ner_predictions[idx].item()]
        word = words[word_id]

        if label.startswith("B-"):
            if current_entity:
                entities.append({"text": current_entity, "type": current_type})
            current_entity = word
            current_type = label[2:]
        elif label.startswith("I-") and current_entity:
            current_entity += " " + word
        else:
            if current_entity:
                entities.append({"text": current_entity, "type": current_type})
                current_entity = None
                current_type = None

    if current_entity:
        entities.append({"text": current_entity, "type": current_type})

    seen = set()
    unique_entities = []
    for e in entities:
        if e["text"] not in seen:
            seen.add(e["text"])
            unique_entities.append(e)

    return {
        "query": request.query,
        "intent": intent,
        "entities": unique_entities
    }