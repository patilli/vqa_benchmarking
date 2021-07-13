# NOTES on dataset
# RETURN: dictionary with {question: ..., img_features: ...}
from typing import Dict, List, Union
from vqa_benchmarking_backend.utils.vocab import Vocabulary
import numpy as np

import torch
from torch.utils.data.dataset import Dataset


class DataSample:
    """
    Superclass for data samples
    """
    def __init__(self, question_id: str, question: str, answers: Dict[str, float], image_id: str, image_path: str, image_feat_path: str, image_transform = None) -> None:
        self._question_id = question_id
        self._question = question
        self._answers = answers
        self._image_id = image_id

        self._image_path = image_path
        self._image_feat_path = image_feat_path

        self._q_token_ids = None
        self._q_feats = None
        self._img = None
        self._img_feats = None
        self._img_transform = image_transform

    @property
    def question_id(self) -> str:
        return self._question_id

    @property
    def question(self) -> str:
        return self._question

    @property
    def answers(self) -> Dict[str, float]:
        return self._answers

    @property
    def image_id(self) -> str:
        return self._image_id

    @property
    def image(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def image_features(self) -> Union[torch.FloatTensor, None]:
        return self._img_feats

    @image_features.setter
    def image_features(self, img_feats) -> torch.FloatTensor:
        self._img_feats = img_feats

    @property
    def question_tokenized(self) -> List[str]:
        raise NotImplementedError

    @property
    def question_features(self) -> Union[None, torch.FloatTensor]:
        return self._q_feats

    @question.setter
    def question(self, question: str):
        self._question = question # ATTENTION: tokenize here in subclass, if needed
        # reset tokens, token ids and embeddings since question updated
        self._q_token_ids = None
        self._q_feats = None

    @question_features.setter
    def question_features(self, q_feats):
        self._q_feats = q_feats

    @answers.setter
    def answers(self, answers):
        self._answers = answers

    @image.setter
    def image(self, image: np.ndarray):
        self._img = image
        # reset image features, since image updated
        self._img_feats = None



class DiagnosticDataset(Dataset):
    """
    Superclass for datasets
    """
    def __len__(self):
        """
        Returns length of dataset
        """
        raise NotImplementedError()

    def __getitem__(self, index) -> DataSample:
        """
        Returns a data sample
        """
        raise NotImplementedError()

    def get_name(self) -> str:
        """
        Required for file caching
        """
        raise NotImplementedError

    def class_idx_to_answer(self, class_idx: int) -> str:
        """
        Returns natural language answer for a class index
        """
        raise NotImplementedError



class DatasetModelAdapter:
    """
    Superclass for model adapters
    When inheriting from this class, make sure to
        * move the model to the intended device
        * move the data to the intended device inside the _forward method
    """

    def get_name(self) -> str:
        """
        Required for file caching
        """
        raise NotImplementedError

    def get_output_size(self) -> int:
        """
        Amount of classes in prediction
        """
        raise NotImplementedError

    def get_torch_module(self) -> torch.nn.Module:
        """
        Return the model
        """
        raise NotImplementedError

    def train(self):
        """
        Set model to train mode (for MC Uncertainty)
        """
        self.get_torch_module().train()

    def eval(self):
        """
        Set model to eval mode
        """
        self.get_torch_module().eval()

    def get_question_embedding(self, sample: DataSample) -> torch.FloatTensor:
        """
        Embed questions without full model run-through
        """
        raise NotImplementedError

    def get_image_embedding(self, sample: DataSample) -> torch.FloatTensor:
        """
        Embed image without full model run-through
        """
        raise NotImplementedError

    def _forward(self, samples: List[DataSample]) -> torch.FloatTensor:
        """
        Overwrite this function to connect a list of samples to your model.
        IMPORTANT: 
            * Make sure that the outputs are probabilities, not logits!
            * Make sure that the data samples are using the samples' question embedding field, if assigned (instead of re-calculating them, they could be modified from feature space methods)
            * Make sure that the data samples are moved to the intended device here
        """
        raise NotImplementedError

    def forward(self, samples: List[DataSample]) -> torch.FloatTensor:
        """
        Return samples x classes results AS PROBABILITIES
        """
        results = self._forward(samples)
        # assert type(results.dtype) == torch.long, f"expected torch.LongTensor, got {type(results)}"
        assert len(results.size()) == 2, f'expected output to be len(samples) x classes, got {results.size()}'        
        assert results.size(dim=0) == len(samples), f'expected {len(samples)} rows, got {results.size()}'
        return results
