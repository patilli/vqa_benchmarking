import json
import os
import random
import re
from typing import Dict, List, Tuple, Union

import cv2 as cv
import numpy as np
import PIL
import torch
from tqdm.auto import tqdm
from vqa_benchmarking_backend.datasets.dataset import (DataSample,
                                                       DiagnosticDataset)
from vqa_benchmarking_backend.tokenizers.vqatokenizer import (
    process_digit_article, process_punctuation)
from vqa_benchmarking_backend.utils.vocab import Vocabulary


def preprocess_question(question: str) -> List[str]:
    return re.sub(
        r"([.,'!?\"()*#:;])",
        '',
        question.lower()
    ).replace('-', ' ').replace('/', ' ')

def load_img(path: str, transform = None) -> np.ndarray:
    img = cv.imread(path)
    if transform:
        img = transform(img)
    return img

def load_img_feats(path: str) -> torch.FloatTensor:
    # Format:
    # f['info']: image_id, objects_id (object class id per ROI), objects_conf (propabily in [0,1] per object id), attrs_id (attribute id per ROI)
    # f["num_bbox"]: number of ROIs
    # f['x]: feature matrix (number of ROIs x feature_dim = 204)
    img_feats = np.load(path, allow_pickle=True)["x"]
    return torch.from_numpy(img_feats)

class CLEVRDataSample(DataSample):
    def __init__(self, question_id: str, question: str, answers: Dict[str, float], image_id: str, image_path: str, image_feat_path: str, image_transform = None) -> None:
        super().__init__(question_id, question, answers, image_id, image_path, image_feat_path, image_transform)
        self._question = preprocess_question(question)

    @property
    def image(self) -> np.ndarray:
        if isinstance(self._img, type(None)):
            self._img = load_img(self._image_path)
        return self._img
    
    @image.setter
    def image(self, image: np.ndarray):
        self._img = image
        # reset image features, since image updated
        self._img_feats = None

    @property
    def question_tokenized(self) -> List[str]:
        return self._question.split()
    
    @property
    def question(self) -> str:
        return self._question
    
    @question.setter
    def question(self, question):
        self._question = preprocess_question(question)
        # reset tokens, token ids and embeddings since question updated
        self._q_token_ids = None
        self._q_feats = None
    
    def __str__(self):
        str_dict = {
            'question_id': self.question_id,
            'question': self.question,
            'tokens': self.question_tokenized,
            'answer': self.answers,
            'imageId': self.image_id,
            'image_path': self._image_path            
        }
        return str(str_dict)

    def question_token_ids(self, vocab: Vocabulary) -> torch.LongTensor:
        return torch.tensor([vocab.stoi(token) for token in self.question_tokenized], dtype=torch.long)

class CLEVRDataset(DiagnosticDataset):
    def __init__(self, question_file: str, 
                       img_dir, 
                       img_feat_dir,
                       idx2ans,
                       transform=None, 
                       load_img_features=False,
                       dataset_fraction: float = 0.05, # percentage of data to keep
                       random_seed: int = 12345):

        self.img_dir      = img_dir
        self.img_feat_dir = img_feat_dir
        self.transform   = transform
        self.load_img_features = load_img_features
        self.idx2ans = idx2ans

        self.data, self.qid_to_sample, self.q_vocab, self.a_vocab = self._load_data(question_file, dataset_fraction, random_seed)

    def _load_data(self, question_file: str, dataset_fraction, random_seed) -> Tuple[List[DataSample], Dict[str, DataSample], Vocabulary, Vocabulary]:
        random.seed(random_seed)
        data = []
        qid_to_sample = {}
        answer_vocab = Vocabulary(itos={}, stoi={})
        question_vocab = Vocabulary(itos={}, stoi={})
            # load questions
        ques = json.load(open(question_file))['questions']
        if dataset_fraction < 1.0:
            # draw fraction of dataset at random
            num_keep = int(len(ques) * dataset_fraction) 
            print(f"Keeping {dataset_fraction*100}%: {num_keep}/{len(ques)} samples")
            ques = random.sample(ques, k=num_keep)
        for question in tqdm(ques):
            iid = question['image_filename']
            qid = str(question['question_index'])
            sample = CLEVRDataSample(question_id=qid,
                                question=question['question'], 
                                answers={question['answer']: 1.0},
                                image_id=iid,
                                image_path=os.path.join(self.img_dir, f"{iid}"),
                                image_feat_path=os.path.join(self.img_feat_dir, f"{iid}.npz"))
            answer_vocab.add_token(question['answer'])
            for token in sample.question_tokenized:
                question_vocab.add_token(token)
            qid_to_sample[qid] = sample
            data.append(qid_to_sample[qid])
        
        return data, qid_to_sample, question_vocab, answer_vocab
   
    def __getitem__(self, index) -> DataSample:
        return self.data[index]

    def label_from_class(self, class_index: int) -> str:
        return self.a_vocab.itos(class_index)
    
    def word_in_vocab(self, word: str) -> bool:
        return self.q_vocab.exists(word)
    
    def get_name(self) -> str:
        # Needed for file caching
        return "CLEVR"

    def __len__(self):
        return len(self.data)

    def index_to_question_id(self, index) -> str:
        return self.data[index].question_id
    
    def class_idx_to_answer(self, class_idx: int) -> Union[str, None]:
        if isinstance(next(iter(self.idx2ans.keys())), int):
            if class_idx in self.idx2ans:
                return self.idx2ans[class_idx]
        else:
            if str(class_idx) in self.idx2ans:
                return self.idx2ans[str(class_idx)]
        return None
