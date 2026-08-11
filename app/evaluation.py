from rouge_score import rouge_scorer
from bert_score import score as bert_score
import nltk

def compute_rouge(reference: str, candidate: str) -> dict:
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return {k: v.fmeasure for k, v in scores.items()}

def compute_bertscore(reference: str, candidate: str, lang="en"):
    P, R, F1 = bert_score([candidate], [reference], lang=lang)
    return {"precision": P.item(), "recall": R.item(), "f1": F1.item()}