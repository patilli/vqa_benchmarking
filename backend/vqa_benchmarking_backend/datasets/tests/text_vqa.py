import sys
sys.path.insert(1, '/home/users0/tillipl/simtech/vqa-benchmarking/code/vqa-benchmarking-backend')
from datasets.TextVQADataset import TextVQADataset
from tqdm.auto import tqdm

img_dir   = '/home/users0/tillipl/simtech/vqa-benchmarking/datasets/Text-VQA/images/test_images'
qsts_path = '/home/users0/tillipl/simtech/vqa-benchmarking/datasets/Text-VQA/questions/TextVQA_0.5.1_test.json'

textVqa_dataset = TextVQADataset(question_file=qsts_path, img_dir= img_dir, img_feat_dir='')

for sample in tqdm(textVqa_dataset):
    print(sample)
test = 0
