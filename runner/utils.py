import os
import random
import logging
import torch
import numpy as np
import csv
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def get_phase_dataset():
    dataset = {}
    dataset['phased1'] = ['ekman', 'group', 'original']
    dataset['phased2'] = ['fakeNews', 'hateval2019']
    return dataset

def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

def get_steps(model_name_or_path):
    parts = model_name_or_path.split('-')
    steps = parts[-1] if parts[-1].isdigit() else 0
    return int(steps)

def set_local_fp(args):
    steps = get_steps(args.model_name_or_path)
    result_file = args.output_dir + f'/result-{steps}.txt'
    f_result = open(result_file, "a+")
    powerset_file = args.output_dir + f'/power_set-{steps}.txt'
    f_powerset = open(powerset_file, "a+")
    labels_file = args.output_dir + f'/labels_all-{steps}.txt'
    f_labels = open(labels_file, "a+")
    return f_result, f_powerset, f_labels

def get_global_csv_path(taxonomy):
    return os.path.join("ckpt", taxonomy) + "/global.csv"

def set_global_fp(args):
    global_csv =  get_global_csv_path(args.taxonomy)# ckpt/[phased2]/global.csv
    f_global = open(global_csv, "a+")
    return f_global

def compute_metrics(args, labels, preds):
    if len(preds) != len(labels):
        print(f'compute_metrics in utils.py: len(preds) != len(labels)')
        return
    result = dict()
    result["accuracy"] = accuracy_score(labels, preds)
    result["weighted_precision"], result["weighted_recall"], result[
        "weighted_f1"], _ = precision_recall_fscore_support(
        labels, preds, average="weighted")
    print(result["accuracy"], result["weighted_precision"], result["weighted_recall"], result["weighted_f1"])
    for k, v in result.items():
        result[k] = round(v, 2)
    return result

def phased1_or_phased2_none_to_labels(args, result, f_labels):
    print(get_labels_metadata(args, "phased1_or_phased2_none"), file = f_labels)
    print(result["accuracy"], result["weighted_precision"], 
        result["weighted_recall"], result["weighted_f1"], file = f_labels)

def phased1_or_phased2_none_to_global(args, result, f_global):
    row = [get_labels_metadata(args, "phased1_or_phased2_none"),
            result["accuracy"], result["weighted_precision"], 
            result["weighted_recall"], result["weighted_f1"]]
    write = csv.writer(f_global)
    write.writerow(row)

def calculate_guilty_non_guilty(preds, do_print):
    if do_print == False:
        return
    ng, g = 0, 0
    for i in preds:
        if np.argmax(i) == 0:
            ng += 1
        else:
            g += 1
    print(f"non_guilty:{ng}, guilty:{g}")

def phased2_binary_to_result(args, taxamony, preds, f_result, phased1_model_name):
    print(get_labels_metadata(args, "phased2_binary", taxamony, 
        phased1_model_name), file = f_result)
    for i in range(len(preds)):
        print(np.argmax(preds[i]), end = " ", file = f_result)
    print('', file = f_result)

def phased2_binary_to_labels(args, taxamony, 
    phased1_model_name, result, f_labels):
    print(get_labels_metadata(args, "phased2_binary", taxamony, 
        phased1_model_name), file = f_labels)
    print(result["accuracy"], result["weighted_precision"], 
        result["weighted_recall"], result["weighted_f1"], file = f_labels)

def phased2_binary_to_global(args, taxamony, 
    phased1_model_name, result, f_global):
    print("check")
    row = [get_labels_metadata(args, "phased2_binary", taxamony, phased1_model_name),
            result["accuracy"], result["weighted_precision"], 
            result["weighted_recall"], result["weighted_f1"]]
    write = csv.writer(f_global)
    write.writerow(row)


def write_label_to_file(args, f_label_text):
    examples = []
    with open(os.path.join(args.data_dir, "zh-cn_test.csv")
        , "r") as f_r:
        examples = f_r.readlines()
    print("labels:", file = f_label_text)
    for example in examples:
        print(example.split('\t')[1], end = " ", 
            file = f_label_text)
    print("", file = f_label_text)

def write_text_to_file(args, f_label_text):
    examples = []
    with open(os.path.join(args.data_dir, "zh-cn_test.csv")
        , "r") as f_r:
        examples = f_r.readlines()
    print("examples:", file = f_label_text)
    for example in examples:
        print(example.split('\t')[0], end = "\n", 
            file = f_label_text)

def write_end_separation_to_labels_all(args):
    steps = get_steps(args.model_name_or_path)
    file_name = args.output_dir + f'/labels_all-{steps}.txt'
    f_labels = open(file_name, "a")
    print("--------", file = f_labels)
    print("", file = f_labels)

def write_label_text_to_file(args):
    steps = get_steps(args.model_name_or_path)
    file_name = args.output_dir + f'/result-{steps}.txt'
    f_label_text = open(file_name, "a")
    write_label_to_file(args, f_label_text)
    write_text_to_file(args, f_label_text)

def write_label_metadata_to_file(args):
    steps = get_steps(args.model_name_or_path)
    file_name = args.output_dir + f'/result-{steps}.txt'
    f_label_text = open(file_name, "a+")
    print("acc, prec, recall, f1:", file = f_label_text)

def phased2_powerset_emotion_to_result(args, preds, taxamony, 
    phased1_model_name, f_result):
    print(get_labels_metadata(args, "phased2_powerset", taxamony, 
        phased1_model_name), file = f_result)
    for pred in preds:
        print(np.argmax(pred), end = " ", file = f_result)
    print("", file = f_result)

def get_emotion_num(preds):
    emotion_num = {}
    for pred in preds:
        if np.argmax(pred) in emotion_num:
            emotion_num[np.argmax(pred)] += 1
        else:
            emotion_num[np.argmax(pred)] = 1
    return emotion_num

def get_guilty_emotion_num(labels, preds):
    if len(preds) != len(labels):
        print(f'get_guilty_emotion_num in utils.py: len(preds) != len(labels)')
        return
    guilty_emotion_num = {}
    for i in range(len(preds)):
        if np.argmax(labels[i]) == 1:
            if np.argmax(preds[i]) in guilty_emotion_num:
                guilty_emotion_num[np.argmax(preds[i])] += 1
            else:
                guilty_emotion_num[np.argmax(preds[i])] = 1
    return guilty_emotion_num

def phased2_powerset_guilty_to_powerset(args, emotion_num, taxamony, 
    phased1_model_name, guilty_emotion_num, f_powerset):
    print(get_labels_metadata(args, "phased2_powerset", taxamony, 
        phased1_model_name), file = f_powerset)
    print("the guilty ratio of emotion", 
        file = f_powerset)
    for [k,v] in sorted(emotion_num.items(), 
        key = lambda x : x[1], reverse = True):
        if k in guilty_emotion_num:
            print(k, v, round(guilty_emotion_num[k] / v, 2), file = f_powerset)
        else:
            print(k, v, 0, file = f_powerset)
    print("the guilty num of emotion", file = f_powerset)
    for [k,v] in sorted(guilty_emotion_num.items(), 
        key = lambda x : x[1], reverse = True):
        print(k, v, file = f_powerset)
    print("---------", file = f_powerset)
    print("", file = f_powerset)

def get_guilty_ids(taxamony):
    if taxamony == "group":
        return [1]
    if taxamony == "ekman":
        return [0, 1, 5]
    if taxamony == "original":
        return [6, 9, 14, 25]

def get_guilty_name(taxamony):
    if taxamony == "group":
        return "[negative]"
    if taxamony == "ekman":
        return "[disgust, fear, sadness]"
    if taxamony == "original":
        return "[confusion, disappointment, fear, sadness]"

def preds_to_binary(preds, taxamony):
    guilty_ids = get_guilty_ids(taxamony)
    binary = np.empty((0, 2), int)
    for pred in preds:
        if np.argmax(pred) not in guilty_ids:
            binary = np.append(binary, np.array([[1, 0]]), axis=0)
        else:
            binary = np.append(binary, np.array([[0, 1]]), axis=0)
    return binary

def truncate_labels_size(labels, classify_num):
    return np.delete(labels, 
        [range(classify_num, np.size(labels, axis = 1))], 1)

def phased2_powerset_to_labels(args, taxamony, 
    phased1_model_name, result, f_labels):
    print(get_labels_metadata(args, "phased2_powerset", taxamony, 
        phased1_model_name), file = f_labels)
    print(result["accuracy"], result["weighted_precision"], 
        result["weighted_recall"], result["weighted_f1"], 
        file = f_labels)

def phased2_powerset_to_global(args, taxamony, 
    phased1_model_name, result, f_global):
    row = [get_labels_metadata(args, "phased2_powerset", taxamony, phased1_model_name),
            result["accuracy"], result["weighted_precision"], 
            result["weighted_recall"], result["weighted_f1"]]
    write = csv.writer(f_global)
    write.writerow(row)

def get_labels_metadata(args, labels_type, taxamony = None, phased1_model_name = None):
    if labels_type == "phased1_or_phased2_none":
        return f'{args.data_language} {args.model_name_or_path} {args.last_layer_mode}'
    if labels_type == "phased2_binary":
        return f'{args.data_language} {taxamony} {phased1_model_name} {args.last_layer_mode}'
    if labels_type == "phased2_powerset":
        guilty_name = get_guilty_name(taxamony)
        return f'{args.data_language} {taxamony} {phased1_model_name} {args.last_layer_mode} with {guilty_name}'

def write_result_files(args, result, preds, labels):
    f_result, f_powerset, f_labels = set_local_fp(args)
    f_global = set_global_fp(args)
    data = get_phase_dataset()
    if args.taxonomy not in data["phased2"] or args.last_layer_mode == "none":
        phased1_or_phased2_none_to_labels(args, result, f_labels)
        phased1_or_phased2_none_to_global(args, result, f_global)
        return
    # args.taxonomy is phased2 and (last_layer_mode is binary or power_set)
    taxamony = args.model_name_or_path.split('/')[1]
    phased1_model_name = args.model_name_or_path.split('/')[2]
    if args.last_layer_mode == "binary":
        calculate_guilty_non_guilty(preds, True)
        phased2_binary_to_result(args, taxamony, preds, f_result, phased1_model_name)
        phased2_binary_to_labels(args, taxamony, 
            phased1_model_name, result, f_labels)
        phased2_binary_to_global(args, taxamony, 
            phased1_model_name, result, f_global)
        return
    elif args.last_layer_mode == "power_set":
        phased2_powerset_emotion_to_result(args, preds, taxamony, 
            phased1_model_name, f_result)
        emotion_num = get_emotion_num(preds)
        guilty_emotion_num = get_guilty_emotion_num(labels, preds)
        phased2_powerset_guilty_to_powerset(args, emotion_num, taxamony, 
            phased1_model_name, guilty_emotion_num, f_powerset)
        # preds, labels to binary
        binary_preds = preds_to_binary(preds, taxamony)
        binary_labels = truncate_labels_size(labels, 2)
        # compute matrix again
        result = compute_metrics(args, binary_labels, binary_preds)
        phased2_powerset_to_labels(args, taxamony, 
            phased1_model_name, result, f_labels)
        phased2_powerset_to_global(args, taxamony, 
            phased1_model_name, result, f_global)
        return

