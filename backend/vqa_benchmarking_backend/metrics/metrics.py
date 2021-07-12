from typing import List, Tuple
import sqlite3
import os
import random

from vqa_benchmarking_backend.datasets.dataset import DatasetModelAdapter, DiagnosticDataset, DataSample
from vqa_benchmarking_backend.metrics.bias import eval_bias, inputs_for_image_bias_featurespace, inputs_for_image_bias_wordspace, inputs_for_question_bias_featurespace, inputs_for_question_bias_imagespace
from vqa_benchmarking_backend.metrics.robustness import eval_robustness, inputs_for_image_robustness_featurespace, inputs_for_image_robustness_imagespace, inputs_for_question_robustness_featurespace, inputs_for_question_robustness_wordspace
from vqa_benchmarking_backend.metrics.sear import eval_sears, inputs_for_question_sears
from vqa_benchmarking_backend.metrics.uncertainty import certainty
from tqdm import tqdm
import torch


def _reduce_min(tensor: torch.FloatTensor):
    reduced = tensor.clone()
    while len(reduced.size()) > 1:
        reduced = reduced.min(dim=0)[0]
    return reduced


def _reduce_max(tensor: torch.FloatTensor):
    reduced = tensor.clone()
    while len(reduced.size()) > 1:
        reduced = reduced.max(dim=0)[0]
    return reduced


def _get_img_feature_range(adapter: DatasetModelAdapter, dataset: DiagnosticDataset, output_path: str, num_samples: int = 500) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    """
    Returns:
        Tuple
        * minimum feature values (per feature column) across dataset (FloatTensor: feature_dim)
        * maximum feature values (per feature column) across dataset (FloatTensor: feature_dim)
    """
    # store this in between sessions, so that it does not have to be recalculated for every run
    filename = os.path.join(output_path, f"{dataset.get_name()}_{adapter.get_name()}_imgfeat_range.pt")
    if os.path.isfile(filename):
        data = torch.load(filename)
        return data['min_feats'], data['max_feats'], data['std']
  
    print('Calculating image feature range...')
    if num_samples <= 0:
        sample_indices = range(len(dataset))
    else:
        sample_indices = random.sample(range(0, len(dataset)), num_samples)
    min_feats = None # feature_dim
    max_feats = None # feature_dim
    feats = []
    for sample_idx in tqdm(sample_indices):
        sample = dataset[sample_idx]
        embedding = adapter.get_image_embedding(sample).cpu()
        feats.append(embedding[-1])
        if isinstance(min_feats, type(None)):
            min_feats = _reduce_min(embedding)
            max_feats = _reduce_max(embedding)
        min_feats = torch.minimum(min_feats, _reduce_min(embedding))
        max_feats = torch.maximum(max_feats, _reduce_max(embedding))
    feats = torch.stack(feats, dim=0) # num_samples x feature_dim
    std = feats.std(dim=0)
    torch.save({'min_feats': min_feats, 'max_feats': max_feats, 'std': std}, filename)
    return min_feats, max_feats, std


def _get_question_feature_range(adapter: DatasetModelAdapter, dataset: DiagnosticDataset, output_path: str, num_samples: int = 500) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    """
    Returns:
        Tuple
        * minimum feature values (per feature column) across dataset (FloatTensor: feature_dim)
        * maximum feature values (per feature column) across dataset (FloatTensor: feature_dim)
    """
    # store this in between sessions, so that it does not have to be recalculated for every run
    filename = os.path.join(output_path, f"{dataset.get_name()}_{adapter.get_name()}_quesfeat_range.pt")
    if os.path.isfile(filename):
        data = torch.load(filename)
        return data['min_feats'], data['max_feats'], data['std']
  
    print('Calculating question feature range...')
    if num_samples <= 0:
        sample_indices = range(len(dataset))
    else:
        sample_indices = random.sample(range(0, len(dataset)), num_samples)
    min_feats = None # feature_dim
    max_feats = None # feature_dim
    feats = []
    for sample_idx in tqdm(sample_indices):
        sample = dataset[sample_idx]
        embedding = adapter.get_question_embedding(sample).cpu()
        feats.append(embedding[-1])
        if isinstance(min_feats, type(None)):
            min_feats = _reduce_min(embedding)
            max_feats = _reduce_max(embedding)
        min_feats = torch.minimum(min_feats, _reduce_min(embedding))
        max_feats = torch.maximum(max_feats, _reduce_max(embedding))
    feats = torch.stack(feats, dim=0) # num_samples x feature_dim
    std = feats.std(dim=0)
    torch.save({'min_feats': min_feats, 'max_feats': max_feats, 'std': std}, filename)
    return min_feats, max_feats, std


def _get_db_connection(output_path: str, adapter: DatasetModelAdapter, dataset: DiagnosticDataset) -> sqlite3.Connection:
    db_file_name = os.path.join(output_path, f"{dataset.get_name()}_{adapter.get_name()}.db")
    print("Opening DB at", db_file_name)
    conn = sqlite3.connect(db_file_name)
    # disable file caching because of our super slow network drives
    conn.execute('PRAGMA synchronous = 0')
    conn.execute('PRAGMA journal_mode = OFF')
    return conn


def _write_class_answer_mapping(db: sqlite3.Connection, adapter: DatasetModelAdapter, dataset: DiagnosticDataset):
    cur = db.cursor()
    cur.execute("SELECT count(name) FROM sqlite_master WHERE type='table' AND name='answers'")
    exists = cur.fetchone()[0]==1
    cur.close()
    if not exists:
        print("Writing answer mapping...")
        
        # create new table
        db.execute("CREATE TABLE answers(class INTEGER PRIMARY KEY, answer TEXT)")
        sql_insert = "INSERT INTO answers VALUES(?,?)"
        insert_values = []
        answer_to_class = {}
        for class_idx in range(adapter.get_output_size()):
            answer_str = dataset.class_idx_to_answer(class_idx)
            if answer_str:
                insert_values.append((class_idx, answer_str))
                answer_to_class[answer_str] = class_idx
        db.executemany(sql_insert, insert_values)
        db.commit()
    else:
        print('Found existing answer mapping')
        answer_to_class = {}
        for class_idx in range(adapter.get_output_size()):
            answer_str = dataset.class_idx_to_answer(class_idx)
            if answer_str:
                answer_to_class[answer_str] = class_idx
    
    cur = db.cursor()
    cur.execute("SELECT count(name) FROM sqlite_master WHERE type='table' AND name='ground_truth'")
    exists = cur.fetchone()[0]==1
    cur.close()
    if not exists:
        print("Writing ground truth mapping...")
        insert_values = []
        db.execute("CREATE TABLE ground_truth(question_id INTEGER, class TEXT, score REAL)")
        sql_insert = "INSERT INTO ground_truth VALUES(?,?,?)"
        for sample in dataset:
            for answer in sample.answers:
                insert_values.append((sample.question_id, answer, sample.answers[answer]))
        db.executemany(sql_insert, insert_values)
        db.commit()
    else:
        print("Found existing ground truth mapping")
     
def _write_qid_question_mapping(db: sqlite3.Connection, adapter: DatasetModelAdapter, dataset: DiagnosticDataset):
    cur = db.cursor()
    cur.execute("SELECT count(name) FROM sqlite_master WHERE type='table' AND name='questions'")
    if cur.fetchone()[0]==1:
        return # answer table exists already
    cur.close()

    print("Writing question mapping...")
    
    # create new table
    db.execute("CREATE TABLE questions(question_id INTEGER PRIMARY KEY, question TEXT, image_id TEXT)")
    sql_insert = "INSERT INTO questions VALUES(?,?,?)"
    insert_values = []
    for sample in dataset:
        insert_values.append((int(sample.question_id), sample.question, sample.image_id))
    db.executemany(sql_insert, insert_values)
    db.commit()
     

def _write_table(db: sqlite3.Connection, metric_name: str, data: dict, overwrite: bool = True):
    if len(data) == 0:
        return # don't write if there's nothing to write

    if overwrite:
        try:
            # delete table, if exists
            delete_table = f"DROP TABLE {metric_name};"
            db.execute(delete_table)
            print(f'Deleted old table {metric_name}')
        except:
            pass # table did not exist in the first place
    
    # create new table
    sql_table = f"CREATE TABLE {metric_name}(question_id INTEGER"
    sql_insert = f"INSERT INTO {metric_name} VALUES(?"
    if 'bias' in metric_name or 'robustness' in metric_name:
        sql_table += ", predicted_class TEXT, prediction_frequency REAL, score REAL"
        sql_insert += ", ?, ?, ?"
    elif metric_name == 'sears':
        sql_table += ", sear_1_predicted_class TEXT, sear_1_applied INTEGER, sear_1_flipped INTEGER"
        sql_table += ", sear_2_predicted_class TEXT, sear_2_applied INTEGER, sear_2_flipped INTEGER"
        sql_table += ", sear_3_predicted_class TEXT, sear_3_applied INTEGER, sear_3_flipped INTEGER"
        sql_table += ", sear_4_predicted_class TEXT, sear_4_applied INTEGER, sear_4_flipped INTEGER"
        sql_insert += ", ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?"
    elif metric_name == 'uncertainty':
        sql_table += ", predicted_class TEXT, prediction_fequency REAL, certainty_score REAL, entropy REAL"
        sql_insert += ", ?, ?, ?, ?"
    elif metric_name == 'accuracy':
        sql_table += ", top_1_class TEXT, top_1_prob REAL, top_1_accuracy REAL"
        sql_table += ", top_2_class TEXT, top_2_prob REAL, top_2_accuracy REAL"
        sql_table += ", top_3_class TEXT, top_3_prob REAL, top_3_accuracy REAL"
        sql_insert += ", ?, ?, ?, ?, ?, ?, ?, ?, ?"
    else:
        raise Exception('unknown metric name', metric_name)
    sql_table += ");"
    sql_insert += ");"
    if overwrite:
        db.execute(sql_table)

    # write data to table
    insert_values = []
    for question_id in data:
        if 'bias' in metric_name or 'robustness' in metric_name:
            score = data[question_id]['bias'] if 'bias' in metric_name else data[question_id]['robustness']
            for class_idx in data[question_id]['class_pred_counter']:
                insert_values.append((int(question_id), class_idx, data[question_id]['class_pred_counter'][class_idx], score))
        elif metric_name == 'sears':
            insert_values.append((int(question_id),
                                  data[question_id]['sear_1']['predicted_class'], data[question_id]['sear_1']['applied'], data[question_id]['sear_1']['flipped'],
                                  data[question_id]['sear_2']['predicted_class'], data[question_id]['sear_2']['applied'], data[question_id]['sear_2']['flipped'],
                                  data[question_id]['sear_3']['predicted_class'], data[question_id]['sear_3']['applied'], data[question_id]['sear_3']['flipped'],
                                  data[question_id]['sear_4']['predicted_class'], data[question_id]['sear_4']['applied'], data[question_id]['sear_4']['flipped']))
        elif metric_name == 'uncertainty':
            entropy = data[question_id]['entropy']
            for class_idx in data[question_id]['class_pred_counter']:
                insert_values.append((int(question_id), class_idx, data[question_id]['class_pred_counter'][class_idx], data[question_id]['class_certainty_scores'][class_idx], entropy))
        elif metric_name == 'accuracy':
            insert_values.append((int(question_id),
                                  data[question_id]['top_1_class'], data[question_id]['top_1_prob'], data[question_id]['top_1_accuracy'],
                                  data[question_id]['top_2_class'], data[question_id]['top_2_prob'], data[question_id]['top_2_accuracy'],
                                  data[question_id]['top_3_class'], data[question_id]['top_3_prob'], data[question_id]['top_3_accuracy']))                
        else:
            raise Exception('unknown metric name', metric_name)

    db.executemany(sql_insert, insert_values)
    db.commit()

@torch.no_grad()
def calculate_metrics(adapter: DatasetModelAdapter, dataset: DiagnosticDataset, metrics: List[str], output_path: str, trials: int = 15,
                      min_tokens: int = 3, max_tokens: int = 10, start_sample: int = 0, max_samples: int = -1):
    """
    Args:
        metrics: choice between ['accuracy',
                                 'question_bias_featurespace', 'question_bias_imagespace',
                                 'image_bias_featurespace', 'image_bias_wordspace',
                                 'image_robustness_imagespace', 'image_robustness_featurespace',
                                 'question_robustness_wordspace', 'question_robustness_featurespace',
                                 'sears',
                                 'uncertainty']

    """

    overwrite = start_sample == 0

    cache_accuracy = {}
    cache_question_bias_featurespace = {}
    cache_question_bias_imagespace = {}
    cache_image_bias_featurespace = {}
    cache_image_bias_wordspace = {}
    cache_image_robustness_imagespace = {}
    cache_image_robustness_featurespace = {}
    cache_question_robustness_featurespace = {}
    cache_sear_flips = {}
    cache_certainty = {}

    db = _get_db_connection(output_path=output_path, adapter=adapter, dataset=dataset)
    adapter.eval()

    if 'question_bias_featurespace' in metrics or 'image_robustness_featurespace' in metrics:
        min_img_feat_vals, max_img_feat_vals, img_feat_std = _get_img_feature_range(adapter, dataset, output_path)
    if 'image_bias_featurespace' in metrics or 'question_robustness_featurespace' in metrics:
        min_ques_feat_vals, max_ques_feat_vals, ques_feat_std = _get_question_feature_range(adapter, dataset, output_path)

    if overwrite:
        _write_class_answer_mapping(db, adapter, dataset)
        _write_qid_question_mapping(db, adapter, dataset)

    print("Calculating metrics...")
    counter = 0
    for sample_idx, sample in enumerate(tqdm(dataset)):
        if sample_idx < start_sample:
            continue # restart at specific index

        top_3_probs, top_3_classes = adapter.forward([sample]).squeeze().topk(k=3, dim=-1, sorted=True)
        original_pred_class = top_3_classes[0].item()
        pred_answer_1_text = dataset.class_idx_to_answer(original_pred_class)

        if 'accuracy' in metrics:
            pred_answer_2_text = dataset.class_idx_to_answer(top_3_classes[1].item())
            pred_answer_3_text = dataset.class_idx_to_answer(top_3_classes[2].item())
            cache_accuracy[sample.question_id] = {
                f'top_1_class': pred_answer_1_text, 'top_1_prob': top_3_probs[0].item(), 'top_1_accuracy': sample.answers[pred_answer_1_text] if pred_answer_1_text in sample.answers else 0.0,
                f'top_2_class': pred_answer_2_text, 'top_2_prob': top_3_probs[1].item(), 'top_2_accuracy': sample.answers[pred_answer_2_text] if pred_answer_2_text in sample.answers else 0.0,
                f'top_3_class': pred_answer_3_text, 'top_3_prob': top_3_probs[2].item(), 'top_3_accuracy': sample.answers[pred_answer_3_text] if pred_answer_3_text in sample.answers else 0.0,
            }

        if 'question_bias_featurespace' in metrics:
            inputs = inputs_for_question_bias_featurespace(current_sample=sample, min_img_feat_val=min_img_feat_vals, max_img_feat_val=max_img_feat_vals, trials=trials)
            preds = adapter.forward(inputs).cpu()
            class_pred_counter, bias = eval_bias(dataset=dataset, original_class_prediction=pred_answer_1_text, predictions=preds)
            cache_question_bias_featurespace[sample.question_id] = {'class_pred_counter': class_pred_counter, 'bias': bias}
            del preds
        if 'question_bias_imagespace' in metrics:
            inputs = inputs_for_question_bias_imagespace(current_sample=sample, dataset=dataset, trials=trials)
            preds = adapter.forward(inputs).cpu()
            class_pred_counter, bias = eval_bias(dataset=dataset, original_class_prediction=pred_answer_1_text, predictions=preds)
            cache_question_bias_imagespace[sample.question_id] = {'class_pred_counter': class_pred_counter, 'bias': bias}
            del preds
        
        if 'image_bias_featurespace' in metrics:
            inputs = inputs_for_image_bias_featurespace(current_sample=sample, min_question_feat_val=min_ques_feat_vals, max_question_feat_val=max_ques_feat_vals, min_tokens=min_tokens, max_tokens=max_tokens, trials=trials)
            preds = adapter.forward(inputs).cpu()
            class_pred_counter, bias = eval_bias(dataset=dataset, original_class_prediction=pred_answer_1_text, predictions=preds)
            cache_image_bias_featurespace[sample.question_id] = {'class_pred_counter': class_pred_counter, 'bias': bias}
            del preds 
        if 'image_bias_wordspace' in metrics:
            inputs = inputs_for_image_bias_wordspace(current_sample=sample, dataset=dataset, trials=trials)
            preds = adapter.forward(inputs).cpu()
            class_pred_counter, bias = eval_bias(dataset=dataset, original_class_prediction=pred_answer_1_text, predictions=preds)
            cache_image_bias_wordspace[sample.question_id] = {'class_pred_counter': class_pred_counter, 'bias': bias}
            del preds

        if 'image_robustness_imagespace' in metrics:
            inputs = inputs_for_image_robustness_imagespace(current_sample=sample, trials=trials//4, noise_types=['gaussian', 'poisson', 's&p', 'speckle'])
            preds = adapter.forward(inputs).cpu()
            class_pred_counter, robustness = eval_robustness(dataset=dataset, original_class_prediction=pred_answer_1_text, predictions=preds)
            cache_image_robustness_imagespace[sample.question_id] = {'class_pred_counter': class_pred_counter, 'robustness': robustness}
            del preds
        if 'image_robustness_featurespace' in metrics:
            inputs = inputs_for_image_robustness_featurespace(current_sample=sample, std=img_feat_std, trials=trials)
            preds = adapter.forward(inputs).cpu()
            class_pred_counter, robustness = eval_robustness(dataset=dataset, original_class_prediction=pred_answer_1_text, predictions=preds)
            cache_image_robustness_featurespace[sample.question_id] = {'class_pred_counter': class_pred_counter, 'robustness': robustness}
            del preds

        if 'question_robustness_featurespace' in metrics:
            inputs = inputs_for_question_robustness_featurespace(current_sample=sample, std=ques_feat_std, adapter=adapter, trials=trials)
            preds = adapter.forward(inputs).cpu()
            class_pred_counter, robustness = eval_robustness(dataset=dataset, original_class_prediction=pred_answer_1_text, predictions=preds)
            cache_question_robustness_featurespace[sample.question_id] = {'class_pred_counter': class_pred_counter, 'robustness': robustness}
            del preds

        if 'sears' in metrics:
            inputs = inputs_for_question_sears(current_sample=sample)
            sear_1_preds = adapter.forward([inputs[0]]).cpu() if inputs[0] else None
            sear_2_preds = adapter.forward([inputs[1]]).cpu() if inputs[1] else None
            sear_3_preds = adapter.forward([inputs[2]]).cpu() if inputs[2] else None
            sear_4_preds = adapter.forward([inputs[3]]).cpu() if inputs[3] else None
            cache_sear_flips[sample.question_id] = eval_sears(dataset=dataset,
                                                              sear_inputs=inputs,
                                                              sear_predictions=(sear_1_preds, sear_2_preds, sear_3_preds, sear_4_preds),
                                                              original_class_prediction=pred_answer_1_text)
            del sear_1_preds
            del sear_2_preds
            del sear_3_preds
            del sear_4_preds

        if 'uncertainty' in metrics:
            class_pred_counter, certainty_scores, entropy = certainty(dataset=dataset, adapter=adapter, sample=sample, trials=trials) # batch=1, batch=1
            cache_certainty[sample.question_id] = {'class_pred_counter': class_pred_counter, 'class_certainty_scores': certainty_scores, 'entropy': entropy}
        
        counter += 1
        if max_samples >= 0 and counter == max_samples:
            break

    print("Writing metrics to DB...")
    for metric in tqdm(metrics):
        if metric == 'accuracy':
            _write_table(db, metric, cache_accuracy, overwrite)
        elif metric == 'question_bias_featurespace':
            _write_table(db, metric, cache_question_bias_featurespace, overwrite)
        elif metric == 'question_bias_imagespace':
            _write_table(db, metric, cache_question_bias_imagespace, overwrite)
        elif metric == 'image_bias_featurespace':
            _write_table(db, metric, cache_image_bias_featurespace, overwrite)
        elif metric == 'image_bias_wordspace':
            _write_table(db, metric, cache_image_bias_wordspace, overwrite)
        elif metric == 'image_robustness_imagespace':
            _write_table(db, metric, cache_image_robustness_imagespace, overwrite)
        elif metric == 'image_robustness_featurespace':
            _write_table(db, metric, cache_image_robustness_featurespace, overwrite)
        elif metric == 'question_robustness_featurespace':
            _write_table(db, metric, cache_question_robustness_featurespace, overwrite)
        elif metric == 'sears':
            _write_table(db, metric, cache_sear_flips, overwrite)
        elif metric == 'uncertainty':
            _write_table(db, metric, cache_certainty, overwrite)
        else:
            raise Exception('Unknown metric', metric)

    db.commit()
    db.close()
