import os
import re
import torch
from typing import Union
import cv2 as cv
import json
from typing import List, Tuple, Dict
from vqa_benchmarking_backend.datasets.dataset import DataSample, DiagnosticDataset
from vqa_benchmarking_backend.utils.vocab import Vocabulary
import numpy as np
from vqa_benchmarking_backend.tokenizers.vqatokenizer import process_digit_article, process_punctuation
from tqdm import tqdm
import random


def preprocess_question(question: str) -> List[str]:
    """
    Removes punctuation and make everything lower case
    """
    return re.sub(
        r"([.,'!?\"()*#:;])",
        '',
        question.lower()
    ).replace('-', ' ').replace('/', ' ')

def load_img(path: str, transform = None) -> np.ndarray:
    """
    Loads an image using module ``cv2``
    """
    img = cv.imread(path)
    if transform:
        img = transform(img)
    return img

def load_img_feats(path: str) -> torch.FloatTensor:
    """
    Loads a numpy array containing image features
    """
    # Format:
    # f['info']: image_id, objects_id (object class id per ROI), objects_conf (propabily in [0,1] per object id), attrs_id (attribute id per ROI)
    # f["num_bbox"]: number of ROIs
    # f['x]: feature matrix (number of ROIs x feature_dim = 204)
    img_feats = np.load(path, allow_pickle=True)["x"]
    return torch.from_numpy(img_feats)

def preprocess_answer(answer: str) -> str:
    """
    Removes punctuation
    """
    answer = process_digit_article(process_punctuation(answer))
    answer = answer.replace(',', '')
    return answer
         
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


class VQADataSample(DataSample):
    """
    Class describing one data sample of the VQA2 dataset
    Inheriting from ``DataSample``
    """
    def __init__(self, question_id: str, question: str, answers: Dict[str, float], image_id: str, image_path: str, image_feat_path: str, image_transform = None) -> None:
        super().__init__(question_id, question, answers, image_id, image_path, image_feat_path, image_transform)
        self._question = preprocess_question(question)

    @property
    def image(self) -> np.ndarray:
        """
        Returns the image, if not present it loads it from ``self._image_path``
        """
        if isinstance(self._img, type(None)):
            self._img = load_img(self._image_path)
        return self._img

    @image.setter
    def image(self, image: np.ndarray):
        """
        Override image, resets image features since image was updated
        """
        self._img = image
        # reset image features, since image updated
        self._img_feats = None

        # pre-computed features
        # if isinstance(self._img_feats, type(None)):
        #    self._img_feats = load_img_feats(self._image_feat_path)

    @property
    def question_tokenized(self) -> List[str]:
        """
        Tokenize question by splitting it
        """
        return self._question.split()

    @property
    def question(self) -> str:
        """
        Return full question
        """
        return self._question

    @question.setter
    def question(self, question):
        """
        Reset tokens, token ids and embeddings since question updated
        """
        self._question = preprocess_question(question)
        # reset tokens, token ids and embeddings since question updated
        self._q_tokens = self.question_tokenized
        self._q_token_ids = None
        self._q_feats = None


class VQADataset(DiagnosticDataset):
    """
    Class describing the VQA2 dataset
    Inheriting from ``DiagnosticDataset``
    """
    def __init__(self, val_question_file: str, val_annotation_file: str,
                       answer_file: str,
                       img_dir, img_feat_dir,
                       name: str,
                       transform=None, load_img_features=False,
                       dataset_fraction: float = 0.05, # percentage of data to keep
                       random_seed: int = 12345
                       ):
        # self.idx_to_qstId = self.computeIdxToId(self.questions)
        self.name = name
        self.img_dir      = img_dir
        self.img_feat_dir = img_feat_dir
        self.transform   = transform
        self.load_img_features = load_img_features

        # load answers 
        if isinstance(answer_file, dict):
            self._class_idx_to_answer = answer_file
        else:
            with open(answer_file, 'r') as f:
                print('Loading answer / class mapping...')
                self._answer_to_class_idx, self._class_idx_to_answer = json.load(f)
                
        self.qid_to_sample = {}
        self.q_vocab = Vocabulary()
        self.q_vocab.add_token('PAD')
        self.q_vocab.add_token('UNK')
        self.q_vocab.add_token('CLS')

        print('Loading val data...')
        self.data = self._load_data(val_question_file, val_annotation_file, dataset_fraction, random_seed)
        print('Loaded', len(self.data), ' samples')
        print('Question vocab size:', len(self.q_vocab))
        print('Answer classes:', len(self._class_idx_to_answer))

    def __len__(self):
        return len(self.data)

    def _load_data(self, question_file: str, annotation_file: str, dataset_fraction: float, random_seed: int) -> Tuple[List[DataSample], Dict[str, DataSample], Vocabulary, Vocabulary]:
        """
        Loads data from VQA json files
        Returns:
            * data: list of ``VQADataSample``
            * qid_to_sample: mapping of question id to data sample
            * question_vocab: ``Vocabulary`` of all unique words occuring in the data
            * answer_vocab: ``Vocabulary`` of all unique answers
        """
        random.seed(random_seed)
        data = []

        with open(question_file, 'r') as f:
            # load questions
            ques = json.load(f)['questions']
            if dataset_fraction < 1.0:
                # draw fraction of dataset at random
                num_keep = int(len(ques) * dataset_fraction) 
                print(f"Keeping {dataset_fraction*100}%: {num_keep}/{len(ques)} samples")
                ques = random.sample(ques, k=num_keep)
                
            for question in tqdm(ques):
                qid = str(question['question_id'])
                iid = str(question['image_id'])
                while(len(iid)) < 12:
                    iid = "0" + iid
                sample = VQADataSample(question_id=qid,
                                    question=question['question'], 
                                    answers={},
                                    image_id=iid,
                                    image_path=os.path.join(self.img_dir, f"COCO_val2014_{iid}.jpg"),
                                    image_feat_path=os.path.join(self.img_feat_dir, f"COCO_val2014_{iid}.npz"))
                for token in sample.question_tokenized:
                    self.q_vocab.add_token(token)
                self.qid_to_sample[qid] = sample
        with open(annotation_file, 'r') as f:
            # load annotations
            anns = json.load(f)['annotations']
            for ann in tqdm(anns):
                qid = str(ann['question_id'])

                if qid not in self.qid_to_sample:
                    # answer not in subset, skip
                    continue

                answers = {}
                # calculate vqa score
                for answer in ann['answers']:
                    answer_str = preprocess_answer(answer['answer'])
                    if not answer_str in answers:
                        answers[answer_str] = 0
                    answers[answer_str] += 1
                for answer_str in answers:
                    answers[answer_str] = answer_score(answers[answer_str])
                self.qid_to_sample[qid].answers = answers
                data.append(self.qid_to_sample[qid])
        
        return data
   
    def __getitem__(self, index) -> DataSample:
        """
        Returns a data sample
        """
        return self.data[index]

    def get_name(self) -> str:
        """
        Returns the name of the dataset, required for file caching
        """
        return self.name

    def class_idx_to_answer(self, class_idx: int) -> Union[str, None]:
        """
        Get the answer string for a given class index from the ``self.idx2ans`` dictionary
        """
        if isinstance(next(iter(self._class_idx_to_answer.keys())), int):
            if class_idx in self._class_idx_to_answer:
                return self._class_idx_to_answer[class_idx]
        else:
            if str(class_idx) in self._class_idx_to_answer:
                return self._class_idx_to_answer[str(class_idx)]
        return None
