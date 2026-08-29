import numpy as np
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


class MetricsCalculator:

    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True
        )
        self.smoothie = SmoothingFunction().method1

    def calculate_single(self, reference: str, candidate: str):
        scores = self.rouge_scorer.score(reference, candidate)
        r1 = scores['rouge1'].fmeasure
        r2 = scores['rouge2'].fmeasure
        rl = scores['rougeL'].fmeasure

        ref_tokens = [reference.split()]
        cand_tokens = candidate.split()
        bleu = sentence_bleu(ref_tokens, cand_tokens,
                              smoothing_function=self.smoothie)

        return {"rouge1": r1, "rouge2": r2, "rougeL": rl, "bleu": bleu}

    def evaluate_batch(self, references: list, candidates: list, threshold=0.3):
        if not references or not candidates:
            return {}

        r1_list, r2_list, rl_list, bleu_list = [], [], [], []
        accuracy_hits = 0

        for ref, cand in zip(references, candidates):
            metrics = self.calculate_single(ref, cand)
            r1_list.append(metrics['rouge1'])
            r2_list.append(metrics['rouge2'])
            rl_list.append(metrics['rougeL'])
            bleu_list.append(metrics['bleu'])
            if metrics['rougeL'] >= threshold:
                accuracy_hits += 1

        results = {
            "rouge1": np.mean(r1_list),
            "rouge2": np.mean(r2_list),
            "rougeL": np.mean(rl_list),
            "bleu": np.mean(bleu_list),
            "accuracy": accuracy_hits / len(references) if references else 0.0,
            "threshold": threshold,
            "samples": len(references),
        }
        return {k: round(v, 4) for k, v in results.items()}
