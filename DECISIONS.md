Decision: JSON format for annotation storage
Why not CSV: entities are nested and variable in number
per query — CSV can't represent this without ugly hacks
Why not XML: verbose, harder to read and parse in Python
Why JSON: nested structure matches our data naturally,
json.load() reads it in one line, HuggingFace datasets
accepts it directly
Interview question: Why not just use a spreadsheet?
Answer: A spreadsheet can't represent variable-length
nested entities per row without denormalizing the data
into something unreadable and error-prone

Decision: validate_data.py runs after every annotation batch, not just once
Why: class imbalance (intent or entity type) only becomes visible 
at scale — checking once at the end means discovering the problem 
after 1,000 queries are already written, too late to course-correct cheaply
Interview question: How did you ensure dataset quality at scale 
with a single annotator?
Answer: Built a validation script run after every batch that 
checks schema compliance and class distribution. This caught 
intent imbalance early (RETURN over-represented vs SEARCH in 
initial batches) and let me deliberately balance future batches 
rather than discovering skew only after the full dataset was done.

Decision: weighted CrossEntropyLoss for intent classification
Why not standard CrossEntropyLoss: model achieved 63% accuracy 
by always predicting SEARCH/BUY — minority classes COMPARE, 
TRACK, RETURN got 0% recall entirely
Why weighted loss: inverse frequency weighting penalizes 
minority class errors more heavily — TRACK and RETURN get 
weight 2.0 vs SEARCH at 0.4
Result: accuracy jumped from 63% to 100% on test set
Interview question: How did you handle class imbalance?
Answer: Computed inverse frequency weights per class, passed 
to CrossEntropyLoss weight parameter. Without this the model 
exploited majority class distribution and learned nothing 
useful for minority intents.