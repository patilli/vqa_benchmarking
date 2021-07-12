import os
import re
import torch
import PIL
import cv2 as cv
from typing import Union
import json
from typing import List, Tuple, Dict
from .dataset import DataSample, DiagnosticDataset
from vqa_benchmarking_backend.utils.vocab import Vocabulary
import numpy as np
from vqa_benchmarking_backend.tokenizers.vqatokenizer import process_digit_article, process_punctuation
from tqdm.auto import tqdm

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

def answer_score(num_humans) -> float:
    """
    Calculates VQA score in [0,1] depending on number of humans having given the same answer
    """
    if num_humans == 0:
        return .0
    elif num_humans == 1:
        return .3
    elif num_humans == 2:
        return .6
    elif num_humans == 3:
        return .9
    else:
        return 1

class TextVQADataSample(DataSample):
    def __init__(self, question_id: str, question: str, answers: Dict[str, float], image_id: str, image_path: str, image_feat_path: str, image_transform = None) -> None:
        super().__init__(question_id, question, answers, image_id, image_path, image_feat_path, image_transform)
        self._question = preprocess_question(question)

    @property
    def image(self):
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

class TextVQADataset(DiagnosticDataset):
    def __init__(self, question_file: str, img_dir, img_feat_dir, idx2ans, transform=None, load_img_features=False):
        self.img_dir      = img_dir
        self.img_feat_dir = img_feat_dir
        self.transform   = transform
        self.load_img_features = load_img_features
        self.idx2ans = idx2ans

        self.data, self.qid_to_sample, self.q_vocab, self.a_vocab = self._load_data(question_file)

    def _load_data(self, question_file: str) -> Tuple[List[DataSample], Dict[str, DataSample], Vocabulary, Vocabulary]:
        data = []
        qid_to_sample = {}
        answer_vocab = Vocabulary(itos={}, stoi={})
        question_vocab = Vocabulary(itos={}, stoi={})
            # load questions
        ques = json.load(open(question_file))
        for question in tqdm(ques['data']):
            qid = str(question['question_id'])
            iid = str(question['image_id'])
            sample = TextVQADataSample(question_id=qid,
                                question=question['question'], 
                                answers={},
                                image_id=iid,
                                image_path=os.path.join(self.img_dir, f"{iid}.jpg"),
                                image_feat_path=os.path.join(self.img_feat_dir, f"{iid}.npz"))
            answers = {}
            for answer in question['answers']:
                answer_str = answer
                if not answer_str in answers:
                    answers[answer_str] = 0
                answers[answer_str] += 1
            for answer_str in answers:
                answers[answer_str] = answer_score(answers[answer_str])
            sample.answers = answers

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

    def __len__(self):
        return len(self.data)

    def index_to_question_id(self, index) -> str:
        return self.data[index].question_id

    def get_name(self) -> str:
        # Needed for file caching
        return 'TextVQA'

    def class_idx_to_answer(self, class_idx: int) -> Union[str, None]:
        if isinstance(next(iter(self.idx2ans.keys())), int):
            if class_idx in self.idx2ans:
                return self.idx2ans[class_idx]
        else:
            if str(class_idx) in self.idx2ans:
                return self.idx2ans[str(class_idx)]
        return None
