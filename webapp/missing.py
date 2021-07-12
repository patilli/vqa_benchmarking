import sqlite3
import os

metrics_list = ['accuracy', 'image_bias_wordspace', 'question_bias_imagespace', 'image_robustness_featurespace', 'image_robustness_imagespace',
		'question_robustness_featurespace', 'sears', 'uncertainty']
model_list = ['BAN-8', 'MMNASNET-LARGE', 'MCAN-LARGE']
ds_list = ['CLEVR', 'GQA', 'GQA-HEAD', 'GQA-TAIL', 'GQA-OOD', 'OKVQA', 'TextVQA', 'VQA2', 'GQA-OOD-HEAD', "GQA-OOD-ALL", 'GQA-OOD-TAIL']


for model in model_list:
    for dataset in ds_list:
        db_file = f'outputs/{dataset}_{model}.db'
        # print('Chekcing', db_file)
        if not os.path.isfile(db_file):
            print(f"MISSING database for {model} {dataset}")
        else:
            conn = sqlite3.connect(db_file)
            conn.execute('PRAGMA synchronous = 0')
            conn.execute('PRAGMA journal_mode = OFF')

            for metric in metrics_list:
                exists = conn.execute(f"SELECT count(name) FROM sqlite_master WHERE type='table' AND name='{metric}'").fetchone()[0]
                if exists < 1:
                    print(f"MISSING metric {metric} for {model} {dataset}")
