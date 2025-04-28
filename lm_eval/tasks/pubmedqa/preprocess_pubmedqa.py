def doc_to_text(doc) -> str:
    ctxs = "\n".join(doc["context"])
    return "Abstract: {}\nQuestion: {}\nAnswer:".format(
        ctxs,
        doc["question"],
    )
