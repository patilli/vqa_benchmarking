Integrate new Datasets
=======================

This document provides a brief overview how to integrate a new benchmarking dataset.

We provide two classes that the new dataset needs to inherit from:

- ``DataSample``
- ``DiagnosticDataset``

Create new Data Samples
-------------------------

Each sample of a dataset is represented as an object of type ``vqa_benchmarking_backend.datasets.dataset.DataSample``.
It stores all the relevant information, like the id's for the question and image, the tokenized question, the corresponding answer, 
and the path to the image.

The following code block contains an exemplary ``DataSample``

.. code-block:: python

    from vqa_benchmarking_backend.datasets.dataset import DataSample

    class MyDataSample(DataSample):
        def __init__(self, 
                     question_id: str, 
                     question: str, 
                     answers: Dict[str, float], 
                     image_id: str, 
                     image_path: str) -> None:

            super().__init__(question_id, 
                             question, 
                             answers, 
                             image_id, 
                             image_path)
            # add your question preprocessing function
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

Create new Diagnostic Datasets
--------------------------------

An object of ``DiagnosticDataset`` requires the path to the image directory, a name for the dataset, and a dictionary 
that contains a mapping of classifier index to the natural language answer string.
From the ``__getitem__`` accessor, an instance of your custom ``vqa_benchmarking_backend.datasets.dataset.DataSample`` class (as created above in the class ``MyDataSample``) should be returned.
Here, the constructor loads all data using the ``_load_data`` method. You should create your own data loading function to match the data format for your dataset.
The ``data`` property should be a list with objects of ``MyDataSample`` for each data entry from the original data format.

The following code block contains an exemplary ``vqa_benchmarking_backend.datasets.dataset.DiagnosticDataset``.

.. code-block:: python

    from vqa_benchmarking_backend.datasets.dataset import DiagnosticDataset
    from vqa_benchmarking_backend.utils.vocab import Vocabulary

    class MyDataset(DiagnosticDataset):
        def __init__(self, 
                     question_file: str, 
                     img_dir: str,
                     idx2ans: Dict[int, str],
                     name: str) -> None:

            self.img_dir      = img_dir
            self.idx2ans      = idx2ans
            self.name         = name
    
            self.data, self.qid_to_sample, self.q_vocab, self.a_vocab = self._load_data(question_file)
    
        def _load_data(self, question_file: str) -> Tuple[List[DataSample], Dict[str, DataSample], Vocabulary, Vocabulary]:
            data = []
            qid_to_sample = {}
            answer_vocab = Vocabulary(itos={}, stoi={})
            question_vocab = Vocabulary(itos={}, stoi={})
            # load questions
            ques = json.load(open(question_file))
            for qid in tqdm(ques):
                iid = str(ques[qid]['imageId'])
                sample = MyDataSample(question_id=qid,
                                      question=ques[qid]['question'], 
                                      answers={ques[qid]['answer']: 1.0},
                                      image_id=iid,
                                      image_path=os.path.join(self.img_dir, f"{iid}.jpg"))
                answer_vocab.add_token(ques[qid]['answer'])
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
        
        def get_name(self) -> str:
            # Needed for file caching
            return self.name
    
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
