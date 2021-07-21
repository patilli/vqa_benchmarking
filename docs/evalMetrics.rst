.. _Evaluate Metrics:

Evaluate Metrics
================

To start the evaluation on a given list of metrics, you need to instantiate a dataset inherting from our ``DiagnosticDataset``.
The calculation starts by calling ``calculate_metrics``, and pass the model adapter, dataset, output directory and amount of trials as parameters.
The parameter ``trials`` refers to the number of monte carlo trials that are performed and averaged for respective metrics.

The following code block contains an example how a script could look like.

.. code-block:: python

    from vqa_benchmarking_backend.datasets.GQADataset import GQADataset  # or import your own dataset
    from vqa_benchmarking_backend.metrics.metrics import calculate_metrics
   
    output_dir = '/path/to/my/ouput/dir' # set output directory for results. This should match the directory you are supplying to the webserver in webapp/server.py

    # directories containing the data
    qsts_path = 'path/to/GQA/questions.json' 
    img_dir   = 'path/to/GQA/images/'

    # file that contains a dictionary mapping from answer index to answer text: {idx: ans_str}
    idx2ans = load_idx_mapping()

    # instantiate dataset using data directories and index/answer mapping
    dataset = GQADataset(question_file=qsts_path, img_dir= img_dir, img_feat_dir='', idx2ans=idx2ans, name='GQA')

    # define a list with all metrics the model should be tested on. Remove as needed.
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
    
    # Run the metrics calculation. Once finished, start the webserver at webapp/server.py and the vue.js app using 'npm start' in webapp/ folder, then inspect the results in your webbrowser.
    calculate_metrics(adapter=model_adapter, dataset=dataset, output_path=output_dir, metrics=metrics, trials=7)
