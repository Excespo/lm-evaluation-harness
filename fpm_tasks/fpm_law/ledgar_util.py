import datasets
import numpy as np
import transformers.data.metrics.squad_metrics as squad_metrics

from lm_eval.api.metrics import metric_max_over_ground_truths


def doc_to_choice(doc):
    return [str(i) for i in range(100)]