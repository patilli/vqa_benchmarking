from vqa_benchmarking_backend.utils.vocab import Vocabulary
from vqa_benchmarking_backend.datasets.VQADataset import VQADataset
# from vqa_benchmarking_backend.datasets.GQADataset import GQADataset

from vqa_benchmarking_backend.metrics.metrics import calculate_metrics

from models.ban.model import BANAdapter

# load data
ds = VQADataset(val_question_file='./.data/val_questions.json', val_annotation_file='./.data/val_anns.json',
		answer_file='./.data/answer_dict.json',
		img_dir='/home/users2/vaethdk/vqa_data/datasets/OK-VQA/images/val2014', img_feat_dir='/home/users2/vaethdk/vqa_data/datasets/OK-VQA/images/features',
		name='VQA2')
# ds = GQADataset(question_file='', img_dir='', img_feat_dir='', )
vocab = Vocabulary.load('./.data/vqa_vocab.json')

# TODO remove testing only
# vocab.add_token('wheres')x
# vocab.add_token('whats')
# vocab.add_token('which')

# load models
ban = BANAdapter(device="cuda:0", vocab=vocab, ckpt_file='ban8.pt')
print('vqa model device', next(ban.vqa_model.parameters()).device)
print('img feat model device', next(ban.img_feat_extractor.parameters()).device)

# metrics

# 'accuracy',q
# 'question_bias_featurespace', 'question_bias_imagespace',
# 'image_bias_featurespace', 'image_bias_wordspace',
# 'image_robustness_imagespace', 'image_robustness_featurespace',
# 'question_robustness_wordspace', 'question_robustness_featurespace',
# 'sears',
# 'uncertainty'
calculate_metrics(adapter=ban, dataset=ds, metrics=['accuracy',
					            'question_bias_imagespace',
					            'image_bias_wordspace',
						    'image_robustness_imagespace', 'image_robustness_featurespace',
						     'question_robustness_featurespace', 'sears', 'uncertainty'
						    ], output_path='./output', trials=7, start_sample=0, max_samples=-1)


print("DONE")