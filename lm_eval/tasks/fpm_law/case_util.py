def doc_to_text(doc):
    options = ["A", "B", "C", "D", "E"]
    text = f"Context: {doc['context']}\nOptions:\n"
    for i, ending in enumerate(doc["endings"]):
        text += f"{options[i]}. {ending}\n"
    text += "\nAnswer:"
    return text

def doc_to_target(doc):
    return ["A", "B", "C", "D", "E"][doc["label"]]
