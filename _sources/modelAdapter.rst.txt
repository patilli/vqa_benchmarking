Write a Model Adapter
=====================

The interface between your model and the evaluation on our metrics and datasets is provided as a ``vqa_benchmarking_backend.datasets.dataset.DatasetModelAdapter`` .
An adapter wraps around a model and is required to return a probability distribution from its ``_forward()`` function.
During metric calculation, the ``_forward()`` function recieves a list of ``DataSample``s that need to be transformed to fit your models expected input.

Some general getter functions are required, e.g. its name ``get_name()``, output size ``get_output_size()``, and the model itself ``get_torch_module()`` .
The functions ``get_question_embedding()`` and ``get_image_embedding()`` should fill the properties ``question_features`` 
and ``image_features`` of a ``DataSample`` object respectively in order to enable caching and appyling noise onto the feature representations. 

Now that we have a created a ``DatasetModelAdapter``, we can start evaluating (see :ref:`Evaluate Metrics`) .

.. code-block:: python

    from vqa_benchmarking_backend.datasets.dataset import DatasetModelAdapter

    class MyModelAdapter(DatasetModelAdapter):
        """
        NOTE: when inheriting from this class, make sure to

            * move the model to the intended device
            * move the data to the intended device inside the _forward method
        """
        def __init__(self, 
                     device,
                     vocab: Vocabulary, 
                     ckpt_file: str = '',
                     name: str,
                     n_classes: int) -> None:

            self.device = device
            self.vocab = vocab
            self.name = name
            self.n_classes = n_classes
            self.vqa_model = myModel().to(device) # the pytorch instance of the VQA model
            self.vqa_model.load_state_dict(torch.load(ckpt_file, map_location=device)['state_dict'])

            gpu_id = int(device.split(':')[1]) # cuda:ID -> ID
            self.img_feat_extractor, self.img_feat_cfg = setup("bottomupattention/configs/bua-caffe/extract-bua-caffe-r101.yaml", 10, 100, gpu_id)  # in this example, we load an external image feature extractor


        def get_name(self) -> str:
            # Needed for file caching, has to be overriden
            return self.name

        def get_output_size(self) -> int:
            # number of classes in prediction, has to be overriden
            return self.n_classes

        def get_torch_module(self) -> torch.nn.Module:
            # return the pytorch VQA model, has to be overriden
            return self.vqa_model

        def question_token_ids(self, question_tokenized: List[str]) -> torch.LongTensor:
            # helper function to get token ids as input to our VQA model, custom to this example
            return torch.tensor([self.vocab.stoi(token) if self.vocab.exists(token) else self.vocab.stoi('UNK') for token in question_tokenized], dtype=torch.long)

        def get_question_embedding(self, sample: DataSample) -> torch.FloatTensor:
            # embed questions without full model forward-pass, has to be overriden
            if isinstance(sample.question_features, type(None)):
                sample.question_features = self.vqa_model.embedding(self.question_token_ids(sample.question_tokenized).to(self.device)).cpu()
            return sample.question_features

        def get_image_embedding(self, sample: DataSample) -> torch.FloatTensor:
            # embed images without full model forward-pass, has to be overriden
            # in this example, the feature extractor is external
            if isinstance(sample.image_features, type(None)):
                sample.image_features = extract_feat_in_memory(self.img_feat_extractor, sample._image_path, self.img_feat_cfg)['x'].cpu()
            return sample.image_features

        def _forward(self, samples: List[DataSample]) -> torch.FloatTensor:
            """
            Overwrite this function to run a forward-pass of a list of samples using your model.
            IMPORTANT: 
                * Make sure that the outputs are probabilities, not logits!
                * Make sure that the data samples are using the samples' question embedding field, if assigned (instead of re-calculating them, they could be modified from feature space methods)
                * Make sure that the data samples are moved to the intended device here
            """
            q_feats = pad_sequence(sequences=[self.get_question_embedding(sample).to(self.device) for sample in samples], batch_first=True) # extract question features
            img_feats = pad_sequence(sequences=[self.get_image_embedding(sample).to(self.device) for sample in samples], batch_first=True) # extract image features

            logits = self.vqa_model.forward(img_feats, q_feats) # run forward-pass for our VQA model
            probs = logits.softmax(dim=-1) # convert model outputs to probability distribution across answer space

            return probs
