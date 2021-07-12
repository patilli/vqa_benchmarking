
from nltk import data
from vqa_benchmarking_backend.datasets.dataset import DatasetModelAdapter, DataSample, DiagnosticDataset
import torch
from torch.distributions import Categorical
from typing import Tuple, Dict
import math
from collections import defaultdict


@torch.no_grad()
def certainty(dataset: DiagnosticDataset, adapter: DatasetModelAdapter, sample: DataSample, trials: int = 15) -> Tuple[Dict[int, float], Dict[int, float], float]:
    """
    Monte-Carlo uncertainty: predict on same sample num_iters times with different dropouts -> measure how often prediction rank changes

    Returns:
        Tuple:
        * Mapping from best prediction class -> fraction of total predictions
        * Mapping from best prediction class -> certainty score in range [0,1]
        * Entropy
    """ 
    adapter.train() # important to activate dropout for this method!
    class_pred_counter = defaultdict(float)
    answer_pred_counter = defaultdict(float)
    
    # MC method
    certainties = torch.zeros(adapter.get_output_size()) # num_classes
    for _ in range(trials):
        preds = adapter.forward([sample]).squeeze() # num_classes
        top_pred_class = preds.argmax(dim=-1).item() # scalar
        class_pred_counter[top_pred_class] += 1 # increment class prediction counter
        answer = dataset.class_idx_to_answer(top_pred_class)
        answer_pred_counter[answer] += 1
        certainties += preds.cpu()  # MC measure
    certainties /= trials
    
    # Assign certainty per top predicted classes
    certainty_scores = {}
    certainty_scores_answers = defaultdict(float)
    for class_idx in class_pred_counter:
        certainty_scores[class_idx] = certainties[class_idx].item() # scalar
        class_pred_counter[class_idx] /= trials

        answer = dataset.class_idx_to_answer(class_idx)
        certainty_scores_answers[answer] += certainties[class_idx].item() # scalar
    for answer in answer_pred_counter:
        answer_pred_counter[answer] /= trials

    entropy = (Categorical(probs=certainties).entropy() / math.log(adapter.get_output_size())) # scalar
    
    adapter.eval() # deactivate dropout
    return answer_pred_counter, certainty_scores_answers, entropy.cpu().item()
