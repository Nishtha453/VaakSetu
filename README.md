# VaakSetu — Hinglish NLU for Indian E-commerce

**Live Demo:** https://vaaksetu.streamlit.app  
**Model:** https://huggingface.co/Nishtha555/VaakSetu  
**Dataset:** 1,000 annotated Hinglish e-commerce queries  

## What is VaakSetu?

Indian e-commerce platforms like Meesho and Flipkart fail on natural Hinglish queries like "Samsung ka phone chahiye 15000 ke andar fast delivery". Hindi words get broken into meaningless fragments by English-only models, losing intent and entity information entirely.

VaakSetu solves this by fine-tuning MuRIL — a multilingual model trained on 17 Indian languages — on the first publicly available labeled Hinglish e-commerce dataset.

## Results

| Task | Metric | Score |
|------|--------|-------|
| Intent Classification | Accuracy | 100% |
| Named Entity Recognition | F1 | 0.75 |

## Intent Classes
SEARCH, COMPARE, BUY, TRACK, RETURN

## Entity Types
PRODUCT, BRAND, ATTRIBUTE, PRICE_RANGE, SIZE, DELIVERY_CONSTRAINT

## Tech Stack
- Model: MuRIL + joint intent/NER heads
- Training: Weighted CrossEntropyLoss for class imbalance
- Dataset: 1,000 Hinglish queries annotated manually
- Deployment: Streamlit Cloud

## Try It
https://vaaksetu.streamlit.app