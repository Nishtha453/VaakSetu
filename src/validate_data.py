import json
from collections import Counter

VALID_INTENTS = {"SEARCH", "COMPARE", "BUY", "TRACK", "RETURN"}
VALID_ENTITY_TYPES = {"PRODUCT", "BRAND", "ATTRIBUTE", "PRICE_RANGE", "SIZE", "DELIVERY_CONSTRAINT"}


def load_data(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def validate_query(query, index):
    errors = []

    required_fields = ["id", "query", "intent", "entities"]
    for field in required_fields:
        if field not in query:
            errors.append(f"Query {index}: missing field '{field}'")

    if "intent" in query and query["intent"] not in VALID_INTENTS:
        errors.append(f"Query {index}: invalid intent '{query['intent']}'")

    if "entities" in query:
        for entity in query["entities"]:
            if "type" not in entity:
                errors.append(f"Query {index}: entity missing 'type' field")
            elif entity["type"] not in VALID_ENTITY_TYPES:
                errors.append(f"Query {index}: invalid entity type '{entity['type']}'")

    return errors


def print_statistics(data):
    print(f"\nTotal queries: {len(data)}")

    intent_counts = Counter(q["intent"] for q in data)
    print("\nIntent distribution:")
    for intent, count in intent_counts.items():
        print(f"  {intent}: {count}")

    entity_counts = Counter(
        entity["type"]
        for q in data
        for entity in q["entities"]
    )
    total_entities = sum(entity_counts.values())
    print(f"\nTotal entities: {total_entities}")
    print("Entity type distribution:")
    for entity_type, count in entity_counts.items():
        print(f"  {entity_type}: {count}")


if __name__ == "__main__":
    data = load_data("data/annotations.json")

    all_errors = []
    for i, query in enumerate(data):
        errors = validate_query(query, i)
        all_errors.extend(errors)

    if all_errors:
        print("VALIDATION FAILED:")
        for error in all_errors:
            print(f"  ✗ {error}")
    else:
        print("✓ All queries passed validation")

    print_statistics(data)