Evaluate Metrics
================

To start the evaluation on a given list of metrics, you need to instantiate a dataset inherting from our ``DiagnosticDataset``.
The calculation starts by calling ``calculate_metrics``, and pass the model adapter, dataset, output directory and amount of trials as parameters.
The parameter ``trials`` refers to the number of monte carlo trials that are performed and averaged for respective metrics.

The following code block contains an example how a script could look like.

.. code-block:: python

    from vqa_benchmarking_backend.datasets.GQADataset import GQADataset
    from vqa_benchmarking_backend.metrics.metrics import calculate_metrics
    # or import your own dataset

    output_dir = '/path/to/my/ouput/dir'

    qsts_path = 'path/to/GQA/questions.json'
    img_dir   = 'path/to/GQA/images/'

    # file that contains a dict {idx: ans_str}
    idx2ans = load_idx_mapping()

    dataset = GQADataset(question_file=qsts_path, img_dir= img_dir, img_feat_dir='', idx2ans=idx2ans, name='GQA')

    # define a list with all metrics the model should be tested on 
    metrics = [
        'accuracy',
        'question_bias_imagespace',
        'image_bias_wordspace',
        'image_robustness_imagespace', 
        'image_robustness_featurespace',
        'question_robustness_featurespace',
        'sears',
        'uncertainty'
    ]
    
    calculate_metrics(adapter=model_adapter, dataset=dataset, output_path=output_dir, metrics=metrics, trials=7)
