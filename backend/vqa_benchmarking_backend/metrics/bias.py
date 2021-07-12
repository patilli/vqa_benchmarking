import torch
from vqa_benchmarking_backend.datasets.dataset import DiagnosticDataset, DataSample, DatasetModelAdapter
from typing import List, Set, Tuple, Dict
import random
from copy import deepcopy
import spacy
from collections import defaultdict

"""
Requirements for a data sample:
* image_feats
* image_id
* labels
* question
* question_feats
"""

nlp = spacy.load("en_core_web_sm")


@torch.no_grad()
def inputs_for_question_bias_featurespace(current_sample: DataSample,
                                          min_img_feat_val: torch.FloatTensor, max_img_feat_val: torch.FloatTensor,
                                          min_img_feats: int = 10, max_img_feats: int = 100,
                                          trials: int = 15) -> List[DataSample]:
    """
    Creates inputs for measuring bias towards questions by creating random image features.

    Args:
        min_img_feat_val (img_feat_dim): vector containing minimum value per feature dimension
        max_img_feat_val (img_feat_dim): vector containing maximum value per feature dimension

    Returns:
        trials x [min_img_feats..max_img_feats] x img_feat_dim : Tensor of randomly generated feature inputs in range [min_img_feat_val, max_img_feat_val].
                                                                 Number of drawn features (dim=1) is randomly drawn from [min_img_feats, max_img_feats]
    """

    assert len(min_img_feat_val.squeeze().size()) == 1, f'too many dimensions for feature vector, expected be 1-dim. tensor with one column per feature, got {len(min_img_feat_val.squeeze().size())}'
    feature_size = min_img_feat_val.squeeze().size(dim=0)
    
    candidates = []
    for i in range(trials):
        # generate random image input with num_rois regions of interest for this trial
        # draw random amount of roi's in [min_img_feats, max_img_feats]
        num_rois = torch.randint(low=min_img_feats, high=max_img_feats+1, size=(1,)).item()
        rand_imgfeats = torch.rand(num_rois, feature_size)                         # [0,1)
        rand_imgfeats = min_img_feat_val + (max_img_feat_val - min_img_feat_val) * rand_imgfeats  # scale features: [0,1) -> [min_imgfeat_val, max_imgfeat_val)
        candidate = deepcopy(current_sample)
        candidate.image_feats = rand_imgfeats
        candidates.append(candidate)
    
    return candidates


@torch.no_grad()
def inputs_for_question_bias_imagespace(current_sample: DataSample, dataset: DiagnosticDataset, trials: int = 15) -> List[DataSample]:
    """
    Creates inputs for measuring bias towards questions by replacing the current sample's image with images drawn randomly from the dataset.
    Also, checks that the labels of the current sample and the drawn samples don't overlap.
    """
    candidates = []
    size = len(dataset)
    while(len(candidates)) < trials:
        # sample next candidate for image replacement
        candidate_idx = random.randrange(0, size)
        candidate = dataset[candidate_idx]

        # ensure non-overlapping labels
        if len(set(candidate.answers.keys()).intersection(current_sample.answers.keys())) == 0:
            # add current_sample with replaced image to candidate inputs
            new_sample = deepcopy(current_sample)
            new_sample.image = candidate.image
            candidates.append(candidate)
        
    return candidates

@torch.no_grad()
def inputs_for_image_bias_featurespace(current_sample: DataSample,
                                       min_question_feat_val: torch.FloatTensor, max_question_feat_val: torch.FloatTensor,
                                       min_tokens: int, max_tokens: int,
                                       trials: int = 15) -> List[DataSample]:
    """
    Creates inputs for measuring bias towards images by creating random question features.
    """
    assert len(min_question_feat_val.squeeze().size()) == 1, f'too many dimensions for feature vector, expected be 1-dim. tensor with one column per feature, got {len(min_question_feat_val.squeeze().size())}'
    feature_size = min_question_feat_val.squeeze().size(dim=0)
    
    candidates = []
    for i in range(trials):
        # generate random question input with num_token tokens for this trial
        # draw random amount of tokens in [min_tokens, max_tokens]
        num_tokens = torch.randint(low=min_tokens, high=max_tokens+1, size=(1,)).item()
        rand_questionfeats = torch.rand(num_tokens, feature_size)                         # [0,1)
        rand_questionfeats = min_question_feat_val + (max_question_feat_val - min_question_feat_val) * rand_questionfeats  # scale features: [0,1) -> [min_imgfeat_val, max_imgfeat_val)
        candidate = deepcopy(current_sample)
        candidate.question_feats = rand_questionfeats
        candidates.append(candidate)

    return candidates


def _extract_subjects_and_objects_from_text(text: str) -> Tuple[Set[str], Set[str]]:
    doc = nlp(text)
    subj = set()
    obj = set()
    for token in doc:
        if token.dep_.endswith('subj'):
            subj.add(token.text)
        elif token.dep_.endswith('obj'):
            obj.add(token.text)
    return subj, obj


def _questions_different(q_a: str, q_b: str) -> bool:
    """
    Simple comparison for the semantic equality of 2 questions.
    Tests, if the subjects and objects in the question are the same.
    """
    a_subj, a_obj = _extract_subjects_and_objects_from_text(q_a)
    b_subj, b_obj = _extract_subjects_and_objects_from_text(q_b)

    diff_subj = len(a_subj.intersection(b_subj)) == 0
    diff_obj = len(a_obj.intersection(b_obj)) == 0
    diff_asubj_bobj = len(a_subj.intersection(b_obj)) > 0
    diff_aobj_bsubj = len(a_obj.intersection(b_subj)) > 0

    if (diff_subj or diff_obj) and not (diff_asubj_bobj and diff_aobj_bsubj): 
        return True # subjects or objects are completley different and did not just swap places

    return False


@torch.no_grad()
def inputs_for_image_bias_wordspace(current_sample: DataSample, dataset: DiagnosticDataset, trials: int = 15) -> List[DataSample]:
    """
    Creates inputs for measuring bias towards images by replacing the current sample's question with questions drawn randomly from the dataset.
    Also, checks that the questions don't overlap.
    """
    candidates = []
    size = len(dataset)
    while(len(candidates)) < trials:
        # sample next candidate for question replacement
        candidate_idx = random.randrange(0, size)
        candidate = dataset[candidate_idx]

        # ensure non-overlapping questions
        if _questions_different(current_sample.question, candidate.question):
            # add current_sample with replaced question to candidate inputs
            new_sample = deepcopy(current_sample)
            new_sample.question = candidate.question
            candidates.append(candidate)
        
    return candidates


@torch.no_grad()
def eval_bias(dataset: DiagnosticDataset, original_class_prediction: str, predictions: torch.FloatTensor) -> Tuple[Dict[int, float], float]:
    """
    Evalutate predictions generated with `inputs_for_question_bias_featurespace`,
                                         `inputs_for_question_bias_imagespace`,
                                         `inputs_for_image_bias_featurespace` or
                                         `inputs_for_image_bias_wordspace`.

    Args:
        predictions (trials x answer space): Model predictions (probabilities)
     
    Returns:
        * Mapping from best prediction class -> fraction of total predictions
        * normalized bias score (float), where 0 means no bias, and 1 means 100% bias
    """
    assert len(predictions.size()) == 2, f'inputs have wrong dimension, expected trials x answer space, got {len(predictions.size())}'
    trials = predictions.size(dim=0)
    class_pred_counter = defaultdict(float)

    bias_score = []
    for trial in range(trials):
        top_pred_class = predictions[trial].squeeze().argmax(dim=-1).item() # scalar
        top_answer = dataset.class_idx_to_answer(top_pred_class)
        class_pred_counter[top_answer] += 1
        if original_class_prediction == top_answer:
            bias_score.append(1.0)
        else:
            bias_score.append(0.0)
    for class_idx in class_pred_counter:
        class_pred_counter[class_idx] /= trials
    return class_pred_counter, sum(bias_score)/len(bias_score)
