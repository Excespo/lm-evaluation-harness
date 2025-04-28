from typing import List, Tuple
import re

def parse_entities(text: str) -> List[Tuple[str, str]]:
    text = text.strip()
    pattern = r"([^。，；！？]+?) is a (person|organization|location)"
    entities = []
    for line in text.split("\n"):
        entities.extend(re.findall(pattern, line))
    return [(e.strip(), t) for e, t in entities]  

def compute_scores(gold: str, pred: str) -> dict:
    gold_entities = set(parse_entities(gold))
    pred_entities = set(parse_entities(pred))
    
    # 统一小写比较
    gold_lower = {(e.lower(), t) for e, t in gold_entities}
    pred_lower = {(e.lower(), t) for e, t in pred_entities}
    correct = gold_lower & pred_lower
    
    # 指标计算
    precision = len(correct) / len(pred_lower) if pred_lower else 0
    recall = len(correct) / len(gold_lower) if gold_lower else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0
    
    return {
        "em": int(gold_lower == pred_lower),
        "entity_f1": f1,
        # "entity_precision": precision,
        # "entity_recall": recall,
    }

def process_results(doc, results):
    gold = doc["output"]
    pred = results[0].strip()
    
    return compute_scores(gold, pred)