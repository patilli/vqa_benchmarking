import json

from datasets.VQADataset import VQADataset
from vqa_benchmarking_backend.datasets.GQADataset import GQADataset

from vqa_benchmarking_backend.utils.vocab import Vocabulary
from vqa_benchmarking_backend.metrics.metrics import calculate_metrics

from models.ban.model import BANAdapter

# load data
with open('./.data/answer_dict.json', 'r') as f:
        print('Loading answer / class mapping...')
        answer_to_class_idx, class_idx_to_answer = json.load(f)
q_vocab = Vocabulary.load('./.data/vqa_vocab.json')

ds = GQADataset(question_file='/home/users2/vaethdk/vqa_data/datasets/GQA/questions/testdev_balanced_questions.json',
                img_dir='/home/users2/vaethdk/vqa_data/datasets/GQA/images', img_feat_dir='',
                idx2ans=class_idx_to_answer, ans2idx=None)

# load models
ban = BANAdapter(device="cuda:0", vocab=q_vocab, ckpt_file='ban8.pt')
print('vqa model device', next(ban.vqa_model.parameters()).device)
print('img feat model device', next(ban.img_feat_extractor.parameters()).device)

# metrics

# 'accuracy',
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
						    ], output_path='./output', trials=7, start_sample=0, max_samples=1000)


print("DONE")