from copy import deepcopy
from vqa_benchmarking_backend.datasets.dataset import DataSample, DiagnosticDataset
from typing import List, Tuple, Union, Dict
import nltk
import torch


def _apply_SEAR_1(question_postagged: List[Tuple[str, str]]):
    # SEAR 1: WP VBZ -> WP's
    output = ""
    applied = False
    idx = 0
    while idx < len(question_postagged):
        word = question_postagged[idx][0]
        tag = question_postagged[idx][1]
        if tag == 'WP' and word.lower() != 'whom' and len(question_postagged) > idx + 1 and question_postagged[idx+1][1] == 'VBZ':
            # apply replace rule
            if idx > 0:
                output += ' '
            output += word + "'s"
            idx += 1  # skip VBZ
            applied = True
        else:
            if word == '``' or word == "''":  # NLTK likes to replace the first "-Token with `` and last with ''
                output += '"'
            else:
                if idx > 0 and word not in ["'s", "'"]:
                    # add space between words, except for 1st word and possesive ending: 's
                    output += ' '
                output += word
        idx += 1
    return output if applied else None


def _apply_SEAR_2(question_postagged: List[Tuple[str, str]]):
    # SEAR 2: What NOUN -> Which NOUN
    output = ""
    applied = False
    idx = 0
    while idx < len(question_postagged):
        word = question_postagged[idx][0]
        tag = question_postagged[idx][1]
        if word.lower() == 'what' and len(question_postagged) > idx + 1 and question_postagged[idx+1][1] in ['NN', 'NNS', 'NNP', 'NNPS']:
            # apply replace rule
            if idx > 0:
                output += ' '
            output += 'which'
            applied = True
        else:
            if word == '``' or word == "''":  # NLTK likes to replace the first "-Token with `` and last with ''
                output += '"'
            else:
                if idx > 0 and word not in ["'s", "'"]:
                    # add space between words, except for 1st word and possesive ending: 's
                    output += ' '
                output += word
        idx += 1
    return output if applied else None


def _apply_SEAR_3(question_tokenized: List[str]):
    # SEAR 3: color -> colour
    output = []
    applied = False
    for token in question_tokenized:
        if token == 'color':
            output.append('colour')
            applied = True
        else:
            output.append(token)
    return " ".join(output) if applied else None


def _apply_SEAR_4(question_postagged: List[Tuple[str, str]]):
    # SEAR 4: ADV VBZ -> ADV's
    output = ""
    applied = False
    idx = 0
    while idx < len(question_postagged):
        word = question_postagged[idx][0]
        tag = question_postagged[idx][1]
        if tag in ['WRB'] and word.lower() != 'when' and len(question_postagged) > idx + 1 and question_postagged[idx+1][0].lower() == 'is':
            # apply replace rule
            if idx > 0:
                output += ' '
            output += word + "'s"
            idx += 1  # skip VBZ
            applied = True
        else:
            if word == '``' or word == "''":  # NLTK likes to replace the first "-Token with `` and last with ''
                output += '"'
            else:
                if idx > 0 and word not in ["'s", "'"]:
                    # add space between words, except for 1st word and possesive ending: 's
                    output += ' '
                output += word
        idx += 1
    return output if applied else None


@torch.no_grad()
def inputs_for_question_sears(current_sample: DataSample) -> Tuple[Union[DataSample, None], Union[DataSample, None], Union[DataSample, None], Union[DataSample, None]]:
    """
    Creates inputs where semantically equivalent changes are applied to the input questions
    
    Returns: 
        A tuple with 4 entries of either type `DataSample` or with value `None`.        
        1st entry corresponds to SEAR 1, 2nd entry to SEAR 2, ... .
        Note: if a value in the tuple at index `i` is `None`, that means that SEAR `i` is not applicable.  
    """
    # pos-tagging
    q_tokenized = nltk.word_tokenize(text=current_sample.question)
    q_pos_tags = nltk.pos_tag(q_tokenized)

    candidates = []
    sear_1_input = _apply_SEAR_1(q_pos_tags)
    sear_2_input = _apply_SEAR_2(q_pos_tags)
    sear_3_input = _apply_SEAR_3(q_tokenized)
    sear_4_input = _apply_SEAR_4(q_pos_tags)

    if sear_1_input and sear_1_input != current_sample.question:
        sear_1_candidate = deepcopy(current_sample)
        sear_1_candidate.question = sear_1_input
    else:
        sear_1_candidate = None
    candidates.append(sear_1_candidate)

    if sear_2_input and sear_2_input != current_sample.question:
        sear_2_candidate = deepcopy(current_sample)
        sear_2_candidate.question = sear_2_input
    else:
        sear_2_candidate = None
    candidates.append(sear_2_candidate)

    if sear_3_input and sear_3_input != current_sample.question:
        sear_3_candidate = deepcopy(current_sample)
        sear_3_candidate.question = sear_3_input
    else:
        sear_3_candidate = None
    candidates.append(sear_3_candidate)

    if sear_4_input and sear_4_input != current_sample.question:
        sear_4_candidate = deepcopy(current_sample)
        sear_4_candidate.question = sear_4_input
    else:
        sear_4_candidate = None
    candidates.append(sear_4_candidate)

    return candidates


@torch.no_grad()
def eval_sears(dataset: DiagnosticDataset, 
               sear_inputs: Tuple[Union[DataSample, None], Union[DataSample, None], Union[DataSample, None], Union[DataSample, None]],
               sear_predictions: Tuple[Union[torch.FloatTensor, None]],
               original_class_prediction: str) -> Dict[str, dict]:
    """
    Evalutate predictions generated with `inputs_for_question_sears`.

    Args:
        sear_inputs: the 4 outputs generated by `inputs_for_question_sears`
        predictions List[(1 x answer space)] of length 4: Model predictions for SEAR questions or None if SEAR input was None. (probabilities)      

    Returns:
        dictionary with information per sear, e.g.
            sear_4: {
                'predicted_class': 10,
                'flipped': False,
                'applied': True
            }
    """
    assert len(sear_inputs) == 4, f'sear_inputs should have length 4, got {len(sear_inputs)}'

    sear_flips = {}
    for sear_idx, sear_input in enumerate(sear_inputs):
        sear_name = f"sear_{sear_idx + 1}"
        sear_flips[sear_name] = {'predicted_class': '', 'flipped': False, 'applied': False}
        if sear_input:
            top_pred_class = sear_predictions[sear_idx].squeeze().argmax(dim=-1).item() # scalar
            top_answer = dataset.class_idx_to_answer(top_pred_class)
            sear_flips[sear_name]['predicted_class'] = top_answer
            sear_flips[sear_name]['applied'] = True
            if top_answer != original_class_prediction:
                sear_flips[sear_name]['flipped'] = True

    return sear_flips
    