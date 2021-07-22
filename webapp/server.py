import sqlite3
from typing import List, Dict, Union
import json
import os
import argparse

import tornado.ioloop
import tornado.web
from tornado.web import RequestHandler


class CorsJsonHandler(RequestHandler):
    def set_default_headers(self):
        # allow requests from anywhere (CORS)
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "x-requested-with,access-control-allow-origin,authorization,content-type")
        self.set_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS, PUT, DELETE')

    def options(self):
        # no body
        self.set_status(204)
        self.finish()
 
    def prepare(self):
        # extract json
        if self.request.headers.get("Content-Type", "").startswith("application/json"):
            self.json_args = tornado.escape.json_decode(self.request.body)
        else:
            self.json_args = None


def get_database_connections(output_path: str) -> Dict[str, Dict[str, sqlite3.Connection]]:
    dataset_map = {}

    # get all databases in output directiory
    files = os.listdir(output_path)
    for file in files:
        if not file.endswith('.db'):
            continue # no sqlite file

        # recover model and dataset name from filename
        ds_model = file.split('.')[0] # remove extension
        ds_name, model_name = ds_model.split('_')
        if not ds_name in dataset_map:
            dataset_map[ds_name] = {}
    
        # disable file caching because of our super slow network drives
        conn = sqlite3.connect(os.path.join(output_path,file))
        conn.execute('PRAGMA synchronous = 0')
        conn.execute('PRAGMA journal_mode = OFF')
        dataset_map[ds_name][model_name] = conn
        print('Connecting to ', ds_name, model_name, conn)

    return dataset_map


def _get_avg_score_from_uniqe_question_ids(conn: sqlite3.Connection, table: str):
    return conn.execute(f"""
        SELECT avg(score) FROM  (
            SELECT score
            FROM {table}
            GROUP BY question_id
        );
    """).fetchone()[0]

def _get_avg_uncertainty(conn: sqlite3.Connection):
    return conn.execute("""SELECT 100.0 - round(100.0*avg(certainty), 2) FROM (
                                            SELECT
                                                sum(U.certainty_score) as certainty
                                            FROM 
                                                uncertainty as U
                                            INNER JOIN ground_truth as gt
                                                ON gt.question_id = U.question_id
                                            WHERE U.predicted_class == gt.class
                                            GROUP BY U.question_id
                                        )""").fetchone()[0]

def _get_avg_sears(conn: sqlite3.Connection):
    return conn.execute("""
        SELECT 1.0 - sum(CAST(sear_1_flipped + sear_2_flipped + sear_3_flipped + sear_4_flipped as REAL))/sum(CAST(sear_1_applied + sear_2_applied + sear_3_applied + sear_4_applied as REAL))
        FROM sears
    """).fetchone()[0] 


def get_overview_data(connections: Dict[str, Dict[str, sqlite3.Connection]], args):
    print("Creating cache for OverviewHandler...")
    cache = {"summary": [], "detail": {}}

    # load model info
    model_info = {}
    file = os.path.join(args.output_dir, 'model_info.json')
    with open(file, 'r') as f:
        model_info = json.load(f)

    # load dataset info
    dataset_list = set()
    model_list = set()
    for dataset in connections:
        dataset_list.add(dataset)
        for model in connections[dataset]:
            if model not in cache['detail']:
                cache['detail'][model] = []
            model_list.add(model)
            conn = connections[dataset][model]
            print(model, dataset)
            accuracy = 100. * conn.execute(f"SELECT AVG(top_1_accuracy) FROM accuracy;").fetchone()[0]
            biasImage = 100. * _get_avg_score_from_uniqe_question_ids(conn, 'image_bias_wordspace')
            biasQuestion = 100. * _get_avg_score_from_uniqe_question_ids(conn, 'question_bias_imagespace')
            robustnessImageFeaturespace = 100. * _get_avg_score_from_uniqe_question_ids(conn, 'image_robustness_featurespace')
            robustnessImageImagespace = 100. * _get_avg_score_from_uniqe_question_ids(conn, 'image_robustness_imagespace')
            robustnessQuestionFeaturespace = 100. * _get_avg_score_from_uniqe_question_ids(conn, 'question_robustness_featurespace')
            robustnessSears = 100. * _get_avg_sears(conn)
            for i in range(1, 5):
                applied_count, flipped_count = conn.execute(f"""
                    SELECT count(sear_{i}_flipped), sum(sear_{i}_flipped)  FROM 
                        (SELECT
                            sear_{i}_flipped
                        FROM sears
                        WHERE sear_{i}_applied == 1)
                    """).fetchall()[0]
            uncertainty = _get_avg_uncertainty(conn)

            cache['detail'][model].append({
                "model": {
                    "name": model,
                    "parameters": model_info[model]
                },
                "dataset": {
                    "name": dataset,
                    "type": "validation"
                },
                "metrics": {
                    "accuracy": accuracy,
                    "biasImage": biasImage,
                    "biasQuestion": biasQuestion,
                    "robustnessNoiseImageImagespace": robustnessImageImagespace,
                    "robustnessNoiseImageFeaturespace": robustnessImageFeaturespace,
                    "robustnessNoiseText": robustnessQuestionFeaturespace,
                    "robustnessSears": robustnessSears,
                    "uncertainty": uncertainty,
                }
            })

    for model in cache['detail']:
        avg_acc = []
        avg_biasImage = []
        avg_biasQuestion = []
        avg_robustnessImageFeaturespace = []
        avg_robustnessImageImagespace = []
        avg_robustnessQuestionFeaturespace = []
        avg_robustnessSears = []
        avg_uncertainty = []
        for ds_entry in cache['detail'][model]:
            avg_acc.append(ds_entry['metrics']['accuracy'])
            avg_biasImage.append(ds_entry['metrics']['biasImage'])
            avg_biasQuestion.append(ds_entry['metrics']['biasQuestion'])
            avg_robustnessImageFeaturespace.append(ds_entry['metrics']['robustnessNoiseImageFeaturespace'])
            avg_robustnessImageImagespace.append(ds_entry['metrics']['robustnessNoiseImageImagespace'])
            avg_robustnessQuestionFeaturespace.append(ds_entry['metrics']['robustnessNoiseText'])
            avg_robustnessSears.append(ds_entry['metrics']['robustnessSears'])
            avg_uncertainty.append(ds_entry['metrics']['uncertainty'])
        
        cache['summary'].append({
            "model": {
                "name": model,
                "parameters": model_info[model]
            },
            "dataset": {
                "name": dataset,
                "type": "validation"
            },
            "metrics": {
                "accuracy": sum(avg_acc)/len(avg_acc),
                "biasImage": sum(avg_biasImage)/len(avg_biasImage),
                "biasQuestion": sum(avg_biasQuestion)/len(avg_biasQuestion),
                "robustnessNoiseImageImagespace": sum(avg_robustnessImageImagespace)/len(avg_robustnessImageImagespace),
                "robustnessNoiseImageFeaturespace": sum(avg_robustnessImageFeaturespace)/len(avg_robustnessImageFeaturespace),
                "robustnessNoiseText": sum(avg_robustnessQuestionFeaturespace)/len(avg_robustnessQuestionFeaturespace),
                "robustnessSears": sum(avg_robustnessSears)/len(avg_robustnessSears),
                "uncertainty": sum(avg_uncertainty)/len(avg_uncertainty),
        } 
        })
    print("Done")
    return cache, list(dataset_list), list(model_list)


class InformationHandler(CorsJsonHandler):
    """
    Returns a list of available datasets, models and metrics
    """
    def initialize(self) -> None:
        self.dataset_list = self.application.settings.get('cache_dataset_list')
        self.model_list = self.application.settings.get('cache_model_list')
        self.metrics_list = self.application.settings.get('cache_metrics_list')
    
    def post(self):
        self.write({
            "information": {
                "datasets": self.dataset_list,
                "models": self.model_list,
                "metrics": self.metrics_list
            }
        })
        self.set_status(200)
        self.finish()



class OverviewHandler(CorsJsonHandler):
    def initialize(self):
        self.cache = self.application.settings.get('cache_overview')
           
    def post(self):
        self.write(self.cache)
        self.set_status(200)
        self.finish()




class MetricsDetailHandler(CorsJsonHandler):
    def initialize(self, connections: Dict[str, Dict[str, sqlite3.Connection]]):
        self.connections = connections

    def post(self):
        # get arguments
        dataset = self.json_args['dataset']
        model = self.json_args['model']
        metric = self.json_args['metric']

        # acces cache and write response
        x = []
        y = []
        x_title = ''
        y_title = ''
        avg = 0.0
        conn = self.connections[dataset][model]
        if metric == "accuracy":
            avg = 100. * conn.execute(f"SELECT AVG(top_1_accuracy) FROM accuracy;").fetchone()[0]
            values = conn.execute("""
                SELECT
                    CAST(round(100 * top_1_accuracy) as INTEGER)/10, count(*)
                FROM
                    accuracy
                GROUP BY top_1_accuracy;
            """).fetchall() # tuples: accuracy, count
            total_samples = sum([row[1] for row in values])
            for row in values:
                x.append(10. * row[0])
                y.append(100. * row[1] / total_samples)
            x_title = 'Accuracy in %'
            y_title = '% of Dataset'
        elif metric == 'sears':
            avg = 100. * _get_avg_sears(conn)
            x = [1, 2, 3, 4]
            for i in range(1, 5):
                value = conn.execute(f"""
                    SELECT CAST(sum(sear_{i}_flipped) as REAL)/CAST(sum(sear_{i}_applied) as REAL)
                    FROM sears
                    """).fetchone()[0]
                y.append(100.* value)
            x_title= 'SEAR Rule'
            y_title = '% flipped from applied'
        elif 'bias' in metric or 'robustness' in metric:
            avg = 100. * _get_avg_score_from_uniqe_question_ids(conn, metric)
            values = conn.execute(f"""
            SELECT
                CAST(round(100 * score) as INTEGER)/10,  count(*)
            FROM
                {metric}
            GROUP BY score; 
            """).fetchall()
            total_samples = sum([row[1] for row in values])
            for row in values:
                x.append(10. * row[0])
                y.append(100. * row[1]/total_samples)
            x_title = 'Bias in %'
            y_title = '% of Dataset'
        elif metric == 'uncertainty':
            avg = _get_avg_uncertainty(conn)
            values = conn.execute("""
            SELECT count(certainty),  100 - 10*certainty FROM 
                (SELECT
                    CAST(round(100.0 * sum(U.certainty_score)) as INTEGER)/10 as certainty
                FROM 
                    uncertainty as U
                INNER JOIN ground_truth as gt
                    ON gt.question_id = U.question_id
                WHERE U.predicted_class == gt.class
                GROUP BY U.question_id)
            GROUP BY certainty
            ORDER BY certainty DESC;
            """).fetchall()
            total_samples = sum([row[0] for row in values])
            for count, uncertainty in values:
                x.append(uncertainty)
                y.append(100. * count/total_samples)
            x_title = 'Uncertainty in %'
            y_title = '% of Dataset'
        else:
            raise Exception("unkown metric")
            
        
        self.write({
            "metric": {
                "name": metric,
                "model": {
                    "name": model,
                    "parameters": -1
                },
                "dataset": {
                    "name": dataset,
                    "type": "validation"
                },
                "plot": {
                    "x": x,
                    "y": y,
                    "x_title": x_title,
                    "y_title": y_title
                },
                "average": avg
            }
        })
        self.set_status(200)
        self.finish()


class FilterHandler(CorsJsonHandler):
    def initialize(self, connections: Dict[str, Dict[str, sqlite3.Connection]]):
        self.connections = connections

    def post(self):
        # get arguments
        dataset = self.json_args['dataset']
        model = self.json_args['model']
        metric = self.json_args['metric']
        minValue = float(self.json_args['minValue']) / 100.0
        maxValue = float(self.json_args['maxValue']) / 100.0

        conn = self.connections[dataset][model]
        if metric == 'accuracy':
            sql = f"""SELECT acc.question_id, q.question, GROUP_CONCAT(DISTINCT gt.class) , acc.top_1_class, acc.top_1_accuracy
                        FROM accuracy as acc
                        JOIN questions as q ON q.question_id == acc.question_id
                        JOIN ground_truth as gt ON gt.question_id == acc.question_id
                        WHERE acc.top_1_accuracy >= {minValue} AND acc.top_1_accuracy <= {maxValue}
                        GROUP BY acc.question_id
                        ORDER BY acc.top_1_accuracy ASC"""
        elif 'bias' in metric:
            sql = f"""SELECT bias.question_id, q.question, GROUP_CONCAT(DISTINCT gt.class), GROUP_CONCAT(DISTINCT bias.predicted_class), bias.score
                        FROM {metric} as bias
                        JOIN questions as q ON q.question_id == bias.question_id
                        JOIN ground_truth as gt ON gt.question_id == bias.question_id
                        WHERE bias.score >=  {minValue} AND bias.score <= {maxValue}
                        GROUP BY bias.question_id
                        ORDER BY bias.score DESC"""
        elif 'robustness' in metric:
            sql = f"""SELECT rob.question_id, q.question, GROUP_CONCAT(DISTINCT gt.class), GROUP_CONCAT(DISTINCT rob.predicted_class), rob.score
                        FROM {metric} as rob
                        JOIN questions as q ON q.question_id == rob.question_id
                        JOIN ground_truth as gt ON gt.question_id == rob.question_id
                        WHERE rob.score >= {minValue} AND rob.score <= {maxValue}
                        GROUP BY rob.question_id
                        ORDER BY rob.score DESC"""
        elif metric == 'sears':
            sql = f"""SELECT s.question_id, q.question, GROUP_CONCAT(DISTINCT gt.class), GROUP_CONCAT(s.sear_1_predicted_class || ','  || s.sear_2_predicted_class || ','  || s.sear_3_predicted_class || ','  || s.sear_4_predicted_class) as predictions, CAST(s.sear_1_flipped + s.sear_2_flipped + s.sear_3_flipped + s.sear_4_flipped as REAL)/CAST(s.sear_1_applied + s.sear_2_applied + s.sear_3_applied + s.sear_4_applied as REAL) as score
                        FROM sears as s
                        JOIN questions as q ON q.question_id == s.question_id
                        JOIN ground_truth as gt ON gt.question_id == s.question_id
                        WHERE (s.sear_1_applied + s.sear_2_applied + s.sear_3_applied + s.sear_2_flipped) > 0
                                        AND  CAST(s.sear_1_flipped + s.sear_2_flipped + s.sear_3_flipped + s.sear_4_flipped as REAL)/CAST(s.sear_1_applied + s.sear_2_applied + s.sear_3_applied + s.sear_4_applied as REAL) >= {minValue}
                                        AND  CAST(s.sear_1_flipped + s.sear_2_flipped + s.sear_3_flipped + s.sear_4_flipped as REAL)/CAST(s.sear_1_applied + s.sear_2_applied + s.sear_3_applied + s.sear_4_applied as REAL) <= {maxValue}
                        GROUP BY s.question_id
                        ORDER BY score DESC"""
        elif metric == 'uncertainty':
            sql = f"""SELECT unc.question_id, q.question, GROUP_CONCAT(DISTINCT gt.class), GROUP_CONCAT(DISTINCT unc.predicted_class), 1.0-unc.certainty_score as score
                        FROM uncertainty as unc
                        JOIN questions as q ON q.question_id == unc.question_id
                        JOIN ground_truth as gt ON gt.question_id == unc.question_id
                        WHERE 1.0-unc.certainty_score >= {minValue} AND 1.0-unc.certainty_score <= {maxValue}
                        GROUP BY unc.question_id
                        ORDER BY score DESC"""
        values = conn.execute(sql).fetchall()
        self.write({'samples': [
            {   
                'question_id': sample[0],
                'question': sample[1],
                'ground_truth': sample[2],
                'prediction_class': sample[3],
                'score': 100. * sample[4]
            }
            for sample in values
        ]})
        self.set_status(200)
        self.finish()


class SampleHandler(CorsJsonHandler):
    def initialize(self, connections: Dict[str, Dict[str, sqlite3.Connection]]):
        self.connections = connections

    def post(self):
        # get arguments
        dataset = self.json_args['dataset']
        model = self.json_args['model']
        questionId = self.json_args['questionId']

        conn = self.connections[dataset][model]

        sql = f"""SELECT q.question_id, q.question,  
					acc.top_1_class, acc.top_1_accuracy, acc.top_1_prob,
					acc.top_2_class, acc.top_2_accuracy, acc.top_2_prob,
					acc.top_3_class, acc.top_3_accuracy, acc.top_3_prob
                FROM questions as q
                JOIN accuracy as acc on acc.question_id == q.question_id
                WHERE q.question_id == {questionId}"""
        values = conn.execute(sql).fetchone()
        sample = {
            'question_id': values[0],
            'question': values[1],
            'predictions': {
                f'top_{i}': {
                    'answer': values[i*3+2],
                    'accuracy': values[i*3+1+2],
                    'probability': values[i*3+2+2]
                } for i in range(3)
            }
        }

        sql = f"""SELECT class, score FROM ground_truth WHERE question_id == {questionId}"""
        values = conn.execute(sql).fetchall()
        sample['ground_truth'] = []
        for gt in values:
            sample['ground_truth'].append({'answer': gt[0], 'score': gt[1]})
        
        for metric in ['image_bias_wordspace', 'image_robustness_featurespace', 'image_robustness_imagespace', 'question_bias_imagespace', 'question_robustness_featurespace']:
            sql = f"""SELECT  predicted_class, prediction_frequency, score
                        FROM {metric}
                        WHERE question_id == {questionId}"""
            values = conn.execute(sql).fetchall()
            sample[metric] = {'score': 100.0*values[0][2], 'predictions': []}
            for value in values:
                sample[metric]['predictions'].append({
                    'answer': value[0],
                    'frequency': value[1]
                })
        
        sql = f"""SELECT * FROM sears WHERE question_id == {questionId}"""
        values = conn.execute(sql).fetchone()
        sample['sears'] = {}
        for sear_idx in range(4):
            sample['sears'][f"sear_{sear_idx+1}"] = {
                'answer': values[sear_idx*3 + 1],
                "applied": int(values[sear_idx*3 + 1 + 1]),
                'flipped': int(values[sear_idx*3 + 2 + 1]) 
            }

        sql = f"""SELECT predicted_class, prediction_fequency, 1.0-certainty_score FROM uncertainty WHERE question_id == {questionId}"""
        values = conn.execute(sql).fetchall()
        sample['uncertainty'] = []
        for value in values:
            sample['uncertainty'].append({
                'answer': value[0],
                'frequency': value[1],
                'uncertainty': 100. * value[2]
            })
        
        sql = f"""SELECT image_id FROM questions WHERE question_id == {questionId}"""
        values = conn.execute(sql).fetchall()
        sample['imageId'] = values[0]

        self.write(sample)
        self.set_status(200)
        self.finish()                   



def make_app(args):
    connections = get_database_connections(args.output_dir)
    overview_cache, dataset_list, model_list = get_overview_data(connections, args)
    metrics_list = ['accuracy', 'image_bias_wordspace', 'question_bias_imagespace', 'image_robustness_featurespace', 'image_robustness_imagespace',
                    'question_robustness_featurespace', 'sears', 'uncertainty']

    print("Loaded datasets", dataset_list)
    print("Loaded model list", model_list)

    return tornado.web.Application([
        (r"/overview", OverviewHandler),
        (r"/metricsdetail", MetricsDetailHandler, dict(connections=connections)),
        (r"/information", InformationHandler),
        (r"/filter", FilterHandler, dict(connections=connections)),
        (r"/sample", SampleHandler, dict(connections=connections))
    ], cache_overview=overview_cache, cache_dataset_list=dataset_list, cache_model_list=model_list, cache_metrics_list=metrics_list
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VQA Benchmarking")

    parser.add_argument("--output_dir", type=str, default="outputs")

    args = parser.parse_args()

    app = make_app(args)
    app.listen(44123)
    print("Running")
    tornado.ioloop.IOLoop.current().start()