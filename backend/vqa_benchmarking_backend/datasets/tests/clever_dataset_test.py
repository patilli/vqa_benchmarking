import sys
sys.path.insert(1, '/home/users0/tillipl/simtech/vqa-benchmarking/code/vqa-benchmarking-backend')
from datasets.CLEVRDataset import CLEVRDataset
from tqdm.auto import tqdm

img_dir   = '/home/users0/tillipl/simtech/vqa-benchmarking/datasets/CLEVR/CLEVR_v1.0/images/val'
qsts_path = '/home/users0/tillipl/simtech/vqa-benchmarking/datasets/CLEVR/CLEVR_v1.0/questions/CLEVR_val_questions.json'

gqa_dataset = CLEVRDataset(question_file=qsts_path, img_dir= img_dir, img_feat_dir='')

for sample in tqdm(gqa_dataset):
    print(sample)
test = 0
