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