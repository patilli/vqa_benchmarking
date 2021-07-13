Write a Model Adapter
=====================

The interface between your model and the evaluation on our metrics and datasets is provided as a ``DatasetModelAdapter`` .
An adapter wraps around a model and is required to return a probability distribution from its ``_forward()`` function.
The ``_forward()`` function gets a list of ``DataSample`` that needs to be processed to fit your models input parameters.

Some general getter functions are required, e.g. its name ``get_name()``, output size ``get_output_size()``, and the model itself ``get_torch_module()`` .
The functions ``get_question_embedding()`` and ``get_image_embedding()`` should fill the properties ``question_features`` 
and ``image_features`` of a ``DataSample`` object respectively in order to enable appyling noise onto the feature representations. 

Now that we have a created a ``DatasetModelAdapter`` we can start evaluating, see :ref:`Evaluate Metrics` .

.. code-block:: python

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
            self.vqa_model = myModel().to(device)
            self.vqa_model.load_state_dict(torch.load(ckpt_file, map_location=device)['state_dict'])

            gpu_id = int(device.split(':')[1]) # cuda:ID -> ID
            self.img_feat_extractor, self.img_feat_cfg = setup("bottomupattention/configs/bua-caffe/extract-bua-caffe-r101.yaml", 10, 100, gpu_id)


        def get_name(self) -> str:
            # Needed for file caching
            return self.name

        def get_output_size(self) -> int:
            # number of classes in prediction
            return self.n_classes

        def get_torch_module(self) -> torch.nn.Module:
            # return the model
            return self.vqa_model

        def question_token_ids(self, question_tokenized: List[str]) -> torch.LongTensor:
            return torch.tensor([self.vocab.stoi(token) if self.vocab.exists(token) else self.vocab.stoi('UNK') for token in question_tokenized], dtype=torch.long)

        def get_question_embedding(self, sample: DataSample) -> torch.FloatTensor:
            # embed questions without full model run-thtough
            if isinstance(sample.question_features, type(None)):
                sample.question_features = self.vqa_model.embedding(self.question_token_ids(sample.question_tokenized).to(self.device)).cpu()
            return sample.question_features

        def get_image_embedding(self, sample: DataSample) -> torch.FloatTensor:
            if isinstance(sample.image_features, type(None)):
                sample.image_features = extract_feat_in_memory(self.img_feat_extractor, sample._image_path, self.img_feat_cfg)['x'].cpu()
            return sample.image_features

        def _forward(self, samples: List[DataSample]) -> torch.FloatTensor:
            """
            Overwrite this function to connect a list of samples to your model.
            IMPORTANT: 
                * Make sure that the outputs are probabilities, not logits!
                * Make sure that the data samples are using the samples' question embedding field, if assigned (instead of re-calculating them, they could be modified from feature space methods)
                * Make sure that the data samples are moved to the intended device here
            """
            q_feats = pad_sequence(sequences=[self.get_question_embedding(sample).to(self.device) for sample in samples], batch_first=True)
            img_feats = pad_sequence(sequences=[self.get_image_embedding(sample).to(self.device) for sample in samples], batch_first=True)

            logits = self.vqa_model.forward(img_feats, q_feats)
            probs = logits.softmax(dim=-1)

            return probs
