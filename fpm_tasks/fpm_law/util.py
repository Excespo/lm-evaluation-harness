import datasets
import numpy as np
import transformers.data.metrics.squad_metrics as squad_metrics

from lm_eval.api.metrics import metric_max_over_ground_truths


def doc_to_target(doc):
    return doc['options'][doc['gold_index']]

# def process_docs(dataset):
#     # label数量太多的情况
#     unique_labels = set()
#     for ex in dataset:
#         unique_labels.update(ex['label'])
#     unique_labels = sorted(unique_labels)

#     def _process(doc):
#         return {
#             "text": doc["text"],
#             "labels": doc["labels"],  # keep all, but use labels[0] as target
#             "candidates": unique_labels
#         }

#     return dataset.map(_process)

# def process_results(doc, results):
#     # - Pick the maximum likelihood prediction entity
#     # - Evaluate the accuracy and token F1 PER EXAMPLE
#     # - Average over all examples
#     max_idx = np.argmax(np.array([result[0] for result in results]))

#     prediction = doc["entities"][max_idx]
#     gold_label_set = doc["answers"]
#     f1 = metric_max_over_ground_truths(
#         squad_metrics.compute_f1, prediction, gold_label_set
#     )
#     em = metric_max_over_ground_truths(
#         squad_metrics.compute_exact, prediction, gold_label_set
#     )

#     return {
#         "f1": f1,
#         "em": em,
#     }
