import sys
sys.path.insert(1, '/home/users0/tillipl/simtech/vqa-benchmarking/code/vqa-benchmarking-backend')
from datasets.GQADataset import GQADataset
from tqdm.auto import tqdm

img_dir   = '/home/users0/tillipl/simtech/vqa-benchmarking/datasets/GQA/images'
qsts_path = '/home/users0/tillipl/simtech/vqa-benchmarking/datasets/GQA/questions/testdev_balanced_questions.json'

gqa_dataset = GQADataset(question_file=qsts_path, img_dir= img_dir, img_feat_dir='')

for sample in tqdm(gqa_dataset):
    print(sample)
test = 0
