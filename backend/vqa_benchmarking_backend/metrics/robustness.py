from copy import deepcopy
from typing import List, Dict, Tuple

from cv2 import data
from vqa_benchmarking_backend.datasets.dataset import DataSample, DatasetModelAdapter, DiagnosticDataset
import torch
from skimage.util import noise, random_noise
from collections import defaultdict

@torch.no_grad()
def inputs_for_image_robustness_imagespace(current_sample: DataSample, trials: int = 3,
                                           gaussian_mean: float = 0.0, gaussian_variance: float = 0.025,
                                           salt_pepper_amount: float = 0.1, salt_vs_pepper_ratio: float = 0.5,
                                           speckle_mean: float = 0.0, speckle_variance: float = 0.05,
                                           noise_types = ['gaussian', 'poisson', 's&p', 'speckle'],
                                           seed: int = 12345) -> List[DataSample]:
    """
    NOTE: creates len(noise_types) * trials outputs, because 1 output per noise type ()

    https://scikit-image.org/docs/stable/api/skimage.util.html#random-noise

    Args:
        noise_types: sub-list of ['gaussian', 'localvar', 'poisson', 'salt', 'pepper', 's&p', 'speckle']

    Returns:
        List[DataSample] of length len(noise_types)*trials
    """
    candidates = []
    noise_seed = seed
    for i in range(trials):
        # generate image noise
        for noise_mode in noise_types:
            kwargs = {'seed': noise_seed, 'clip': True, 'mode': noise_mode}
            if noise_mode in ['gaussian', 'speckle']:
                kwargs['mean'] = gaussian_mean if noise_mode == 'gaussian' else speckle_mean
                kwargs['var'] = gaussian_variance if noise_mode == 'gaussian' else speckle_variance
            if noise_mode in ['s&p']:
                kwargs['amount'] = salt_pepper_amount
                kwargs['salt_vs_pepper'] = salt_vs_pepper_ratio
            noisy_img = random_noise(current_sample.image, **kwargs)
            candidate = deepcopy(current_sample)
            candidate.image = noisy_img
            candidate.image_features = None # reset image features, they have to be recalculated
            candidates.append(candidate)
        noise_seed += 1 # need to change seed between iterations s.t. we obtain different samples each trial
    return candidates



@torch.no_grad()
def inputs_for_image_robustness_featurespace(current_sample: DataSample, std: float = 0.01, trials: int = 15) -> List[DataSample]:
    """
    Additive gaussian noise for input features
    """
    candidates = []
    for i in range(trials):
        # generate gaussian noise, add to question features
        candidate = deepcopy(current_sample)
        candidate.image_features = torch.normal(mean=candidate.image_features, std=std)
        candidates.append(candidate)
    return candidates


@torch.no_grad()
def inputs_for_question_robustness_wordspace(current_sample: DataSample, trials: int = 15,
                                  noise_types=['typo', 'insert', 'permute', 'synonyms', 'delete'],
                                  max_edits_per_sample: int =2 ) -> List[DataSample]:
    """
    Ideas:
    * typos (might not be in vocab... - should be doable with BERT and fastText.subwords though)
    * change order of words (does it have to be grammatically safe?)
    * insert unneccessary words (when is that safe?)
    * replace with synonyms (where to get synonym map?)
    * delete word (when is that safe? e.g. don't delete 'color' from 'What color is...?')

    maybe noise is more meaningful in feature space than word space
    """
    raise NotImplementedError


@torch.no_grad()
def inputs_for_question_robustness_featurespace(current_sample: DataSample, adapter: DatasetModelAdapter, std: float = 0.01, trials: int = 15) -> List[DataSample]:
    """
    Additive gaussian noise for input features
    """
    candidates = []
    for i in range(trials):
        # generate gaussian noise, add to question features
        candidate = deepcopy(current_sample)
        if isinstance(candidate.question_features, type(None)):
            candidate.question_features = adapter.get_question_embedding(candidate)
        candidate.question_features = torch.normal(mean=candidate.question_features, std=std)
        candidates.append(candidate)
    return candidates


@torch.no_grad()
def eval_robustness(dataset: DiagnosticDataset, original_class_prediction: str, predictions: torch.FloatTensor) -> Tuple[Dict[int, float], float]:
    """
    Evalutate predictions generated with `inputs_for_question_bias_featurespace`,
                                         `inputs_for_question_bias_imagespace`,
                                         `inputs_for_image_bias_featurespace` or
                                         `inputs_for_image_bias_wordspace`.

    Args:
        predictions (trials): Model predictions (probabilities)

    Returns:
        * Mapping from best prediction class -> fraction of total predictions
        * normalized robustness score (float), where 0 means not robust, and 1 means 100% robust
    """
    trials = predictions.size(dim=0)
    class_pred_counter = defaultdict(float)
    robustness_score = []
    for trial in range(trials):
        top_pred_class = predictions[trial].squeeze().argmax(dim=-1).item() # scalar
        top_answer = dataset.class_idx_to_answer(top_pred_class)
        class_pred_counter[top_answer] += 1
        if original_class_prediction == top_answer:
            robustness_score.append(1.0)
        else:
            robustness_score.append(0.0)
    for class_idx in class_pred_counter:
        class_pred_counter[class_idx] /= trials
    return class_pred_counter, sum(robustness_score)/len(robustness_score)
    