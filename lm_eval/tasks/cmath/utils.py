import datasets

def process_docs(dataset):
    def _process(doc):
        return {
            "question": doc["question"],
            "golden": doc["golden"],
        }
    return dataset.map(_process)
