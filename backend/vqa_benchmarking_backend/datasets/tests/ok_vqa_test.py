import sys
sys.path.insert(1, '/home/users0/tillipl/simtech/vqa-benchmarking/code/vqa-benchmarking-backend')
from datasets.VQADataset import VQADataset
import torch


img_dir   = '/home/users0/tillipl/simtech/vqa-benchmarking/datasets/OK-VQA/images/val2014'
qsts_path = '/home/users0/tillipl/simtech/vqa-benchmarking/datasets/OK-VQA/questions/OpenEnded_mscoco_val2014_questions.json'
anno_path = '/home/users0/tillipl/simtech/vqa-benchmarking/datasets/OK-VQA/questions/mscoco_val2014_annotations.json'

okVqa_dataset = VQADataset(question_file=qsts_path, annotation_file=anno_path, img_dir= img_dir, img_feat_dir='')

test = 0
