import argparse
import json
import logging
import os
import glob
import shutil
import warnings
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
from attrdict import AttrDict
from transformers import (
    BertConfig,
    BertTokenizer,
    XLMRobertaConfig,
    XLMRobertaTokenizer,
    AdamW,
    get_linear_schedule_with_warmup
)
from model import BertForMultiLabelClassification
from utils import (
    init_logger,
    set_seed,
    get_phase_dataset,
    compute_metrics,
    write_result_files,
    write_end_separation_to_labels_all
)
from data_loader import (
    load_and_cache_examples,
    GoEmotionsProcessor
)
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def get_cli_args():
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument("--taxonomy", type=str, required=True, 
        help="options: original, ekman, group, fakeNews")
    cli_parser.add_argument("--num_train_epochs", type=int)
    cli_parser.add_argument("--model_name_or_path", type=str)
    cli_parser.add_argument("--tokenizer_name_or_path", type=str)
    cli_parser.add_argument("--data_language", type=str)
    cli_parser.add_argument("--pretrain_language", type=str,
        help="only used in fakeNews with no pretrain model")
    cli_parser.add_argument("--last_layer_mode", type=str,
        help="options: none, binary, power_set")
    return cli_parser.parse_args()

def print_cli_args(cli_args, execute):
    if not execute:
        return
    for k, v in vars(cli_args).items():
        print(k, v)

def print_args(args, execute):
    if not execute:
        return
    print("--- print args begin ---")
    for k, v in args.items():
        print(k, v)
    print("--- print args done ---")

def overwrite(args, cli_args):
    for k, v in vars(cli_args).items():
        if v is None:
            continue
        args[k] = v

def setup_args():
    # Read from config file and make args
    cli_args = get_cli_args()
    print_cli_args(cli_args, False)
    # load json file
    args = AttrDict()
    with open(os.path.join("config", f'{cli_args.taxonomy}.json')) as f:
        args = AttrDict(json.load(f))
    print_args(args, False)
    # overwrite args if cli_args exists
    overwrite(args, cli_args)
    print_args(args, False)
    return args

def special_rules(args):
    # 1. not train if last_layer_mode is "power_set"
    if args.last_layer_mode == "power_set":
        args.do_train = False

def set_output_dir_name(args):
    data = get_phase_dataset()
    #change output_dir name
    if args.taxonomy not in data["phased2"]:
        model_pretrain_name = args.model_name_or_path.split('/')[-1]
        args.output_dir = f'{model_pretrain_name}_{args.data_language}'
    elif args.taxonomy in data["phased2"] and args.last_layer_mode == "none":
        model_pretrain_name = args.model_name_or_path.split('/')[-1]
        args.output_dir = f'{model_pretrain_name}_{args.pretrain_language}_{args.data_language}'
    # args.taxonomy in data["phased2"] and args.last_layer_mode == "binary" or "power_set":
    # model_pretrain_name = f'{model_pretrain_name}_{args.pretrain_language}  
    else:
        model_pretrain_name = args.model_name_or_path.split('/')[-2]
        args.output_dir = f'{model_pretrain_name}_{args.data_language}'
    args.output_dir = os.path.join(args.ckpt_dir, args.output_dir)

def add_language_in_train_dev_test_files(args):
    args.train_file = f'{args.data_language}_{args.train_file}'
    args.dev_file = f'{args.data_language}_{args.dev_file}'
    args.test_file = f'{args.data_language}_{args.test_file}'

def remove_past_checkpoints(args):
    for c in glob.glob(args.output_dir + "/**/", recursive=True):
        file_name = c.split("/")[-2]
        if file_name[:10] == "checkpoint":
            shutil.rmtree(c)

def get_label_list(args):
    processor = GoEmotionsProcessor(args)
    return processor.get_labels()

def load_config(args, label_list):
    if "xlm-roberta" in args.model_name_or_path:
        config = XLMRobertaConfig.from_pretrained(
            args.model_name_or_path,
            num_labels=len(label_list),
            finetuning_task=args.task,
            id2label={str(i): label for i, label in enumerate(label_list)},
            label2id={label: i for i, label in enumerate(label_list)},
        )
    else:
        config = BertConfig.from_pretrained(
            args.model_name_or_path,
            num_labels=len(label_list),
            finetuning_task=args.task,
            id2label={str(i): label for i, label in enumerate(label_list)},
            label2id={label: i for i, label in enumerate(label_list)},
        )
    return config


def load_tokenizer(args):
    if "xlm-roberta" in args.tokenizer_name_or_path:
        tokenizer = XLMRobertaTokenizer.from_pretrained(args.tokenizer_name_or_path)
    else:
        tokenizer = BertTokenizer.from_pretrained(args.tokenizer_name_or_path)
    return tokenizer


def load_model_no_strict(args, config):
    model = BertForMultiLabelClassification.from_pretrained(
        args.model_name_or_path,
    )
    state_dict = model.state_dict()
    del state_dict["classifier.weight"]
    del state_dict["classifier.bias"]
    model = BertForMultiLabelClassification(
        config = config
    )
    model.load_state_dict(state_dict, strict=False)
    return model

def cpu_or_gpu(args, model):
    args.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    model.to(args.device)

def load_dataset(args, tokenizer):
    train_dataset = load_and_cache_examples(args, tokenizer, mode="train") if args.train_file else None
    dev_dataset = load_and_cache_examples(args, tokenizer, mode="dev") if args.dev_file else None
    test_dataset = load_and_cache_examples(args, tokenizer, mode="test") if args.test_file else None
    return train_dataset, dev_dataset, test_dataset

def print_train_result(args, global_step, avg_loss, do_print):
    if do_print == False:
        return
    print(f' global_step = {global_step}, average loss = {avg_loss}')
    
def get_train_dataloader(train_dataset, args):
    train_sampler = RandomSampler(train_dataset)
    return DataLoader(train_dataset, 
        sampler = train_sampler, 
        batch_size = args.train_batch_size)

def get_optimizer(args, model):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    return AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

def get_scheduler(args, optimizer, t_total):
    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps = int(t_total * args.warmup_proportion),
        num_training_steps = t_total
    )

def print_train_info(args, t_total, train_dataset_len):
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", train_dataset_len)
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Total train batch size = %d", args.train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Logging steps = %d", args.logging_steps)
    logger.info("  Save steps = %d", args.save_steps)

def get_loss(model, args, batch):
    batch = tuple(t.to(args.device) for t in batch)
    if "xlm-roberta" in args.model_name_or_path:
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "labels": batch[2]
        }
    else:
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "token_type_ids": batch[2],
            "labels": batch[3]
        }
    outputs = model(**inputs)
    loss = outputs[0]
    return loss

def print_save_checkpoint(global_step, output_dir):
    print(f'Save model checkpoint at step: {global_step}, output_dir: {output_dir}\n')

def train_iter(model, args, step, batch, global_step, 
    tr_loss, train_dataloader_len, need_scheduler
    , optimizer, scheduler, tokenizer, load_scheduler_state):
    model.train()
    loss = get_loss(model, args, batch)
    if args.gradient_accumulation_steps > 1:
        loss = loss / args.gradient_accumulation_steps
    loss.backward()
    tr_loss += loss.item()
    if (step + 1) % args.gradient_accumulation_steps == 0 or (
        train_dataloader_len <= args.gradient_accumulation_steps
        and (step + 1) == train_dataloader_len
    ):
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        if need_scheduler:
            print("scheduler.step()")
            scheduler.step()
        model.zero_grad()
        global_step += 1
        if args.save_steps > 0 and \
            global_step % args.save_steps == 0:
            # Save model checkpoint
            output_dir = os.path.join(args.output_dir, f'checkpoint-{global_step}')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )
            model_to_save.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            torch.save(args, os.path.join(output_dir, "training_args.bin"))
            print_save_checkpoint(global_step, output_dir)
            if args.save_optimizer:
                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            if need_scheduler and load_scheduler_state:
                torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
    return global_step, tr_loss

def train(args, model, tokenizer, train_dataset, 
        dev_dataset = None, test_dataset = None):
    # To Do: parameterize need_scheduler
    need_scheduler = False
    load_scheduler_state = False

    train_dataloader = get_train_dataloader(train_dataset, args)
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = \
            args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    optimizer = get_optimizer(args, model)
    scheduler = get_scheduler(args, optimizer, t_total)
    if args.save_optimizer:
        if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")):
            optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
    if need_scheduler and load_scheduler_state:
        if os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt")):
            scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))
    
    # train info
    print_train_info(args, t_total, len(train_dataset))

    global_step = 0
    tr_loss = 0.0
    model.zero_grad()
    for _ in trange(int(args.num_train_epochs)
        , desc="Epoch", disable=False):
        for step, batch in enumerate(tqdm(train_dataloader, 
            desc="Iteration", disable=False)):

            global_step, tr_loss = \
               train_iter(model, args, step, batch, global_step, 
               tr_loss, len(train_dataloader), need_scheduler, 
               optimizer, scheduler, tokenizer, load_scheduler_state)
            print_train_result(args, global_step, round(tr_loss / global_step, 2), False)
            
            if args.max_steps > 0 and global_step > args.max_steps:
                break                
        if args.max_steps > 0 and global_step > args.max_steps:
            break
    print(f'global_step: {global_step}')
    return global_step, tr_loss / global_step

def get_checkpoints(args):
    checkpoints = []
    data = get_phase_dataset()
    if args.last_layer_mode == "binary" or args.taxonomy not in data["phased2"]:
        checkpoints = list(os.path.dirname(c) for c in glob.glob(args.output_dir + "/**/" + "pytorch_model.bin", recursive=True))
    elif args.last_layer_mode == "power_set" or  args.last_layer_mode == "none":
        checkpoints = list([args.model_name_or_path])
    if not args.eval_all_checkpoints:
        checkpoints = checkpoints[-1:]
    return checkpoints

def print_checkpoints(checkpoints, do_print):
    if do_print == False:
        return
    if not hasattr(checkpoints, '__iter__'):
        print('checkpoints is not iterable')

    print('checkpoint in checkpoints:')
    for checkpoint in checkpoints:
        print(checkpoint, end = ' ')
    print('')

# def store_checkpoints_list(results):
#     output_eval_file = os.path.join(args.output_dir, "eval_results_{}_{}.txt".format(args.num_train_epochs, args.tokenizer_name_or_path.split('/')[-1]))
#     with open(output_eval_file, "w") as f_w:
#         f_w.write("steps, loss, acc, prec, recall, f1\n")
#         for idx in range(len(global_steps)):
#             f_w.write("|{}|{:.2f}|{:.2f}|{:.2f}|{:.2f}|{:.2f}|\n".format(global_steps[idx] * args.save_steps, loss[idx], accuracy[idx], prec[idx], recall[idx], f1[idx]))

#     Data1, = plt.plot(global_steps, loss, 'r.-', label='loss') 
#     Data2, = plt.plot(global_steps, accuracy, 'g-*', label='accuracy') 
#     Data3, = plt.plot(global_steps, prec, 'b-,', label='precesion')
#     Data4, = plt.plot(global_steps, recall, 'y-+', label='recall')
#     Data5, = plt.plot(global_steps, f1, 'm-+', label='f1')
#     plt.legend(handles=[Data1, Data2, Data3, Data4, Data5])
#     plt.xlabel(f'{args.model_name_or_path}, {args.save_steps} for 1 unit', fontsize=16)
#     # plt.ylabel("recall", fontsize=20)
#     output_result_file = os.path.join(args.output_dir, "eval_results_{}_{}.png".format(args.num_train_epochs, args.tokenizer_name_or_path.split('/')[-1]))
#     plt.savefig(output_result_file, bbox_inches='tight')
#     plt.show()

def print_evaluate_metadata(mode, global_step, 
    eval_dataset_len, args, do_print):
    if do_print == False:
        return
    if global_step != None:
        logger.info("***** Running evaluation on {} dataset ({} step) *****".format(mode, global_step))
    else:
        logger.info("***** Running evaluation on {} dataset *****".format(mode))
    logger.info("  Num examples = {}".format(eval_dataset_len))
    logger.info("  Eval Batch size = {}".format(args.eval_batch_size))

def eval_iter(args, model, batch, eval_loss, 
    nb_eval_steps, preds, out_label_ids):
    model.eval()
    batch = tuple(t.to(args.device) for t in batch)
    with torch.no_grad():
        if "xlm-roberta" in args.model_name_or_path:
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[2]
            }
        else:
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "labels": batch[3]
            }
        outputs = model(**inputs)
        tmp_eval_loss, logits = outputs[:2]
        eval_loss += tmp_eval_loss.mean().item()
    nb_eval_steps += 1
    if preds is None:
        preds = 1 / (1 + np.exp(-logits.detach().cpu().numpy()))  # Sigmoid
        out_label_ids = inputs["labels"].detach().cpu().numpy()
    else:
        preds = np.append(preds, 1 / (1 + np.exp(-logits.detach().cpu().numpy())), axis=0)  # Sigmoid
        out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
    return eval_loss, nb_eval_steps, preds, out_label_ids

def clean_neutral_ambigous_in_preds(args, preds):
    data = get_phase_dataset()
    if args.taxonomy in data["phased2"] and \
        (args.last_layer_mode == "power_set" or args.last_layer_mode == "binary"):
        for i in ["neutral", "ambiguous"]:
            label_id = 0
            label_dir = ''
            if args.last_layer_mode == "power_set":
                label_dir = 'data/' + args.model_name_or_path.split('/')[1]
            elif args.last_layer_mode == "binary":
                label_dir = args.data_dir
            with open(os.path.join(label_dir, args.label_file), "r") as f:
                for line in f:
                    if line.split("\n")[0] == i:
                        break
                    label_id += 1
            # print(label_id, np.size(preds, 1))
            if label_id < np.size(preds, 1):
                preds[:, label_id] = 0
    return preds
        
def make_mode_subdir(args, mode, do_make):
    if do_make == False:
        return
    output_dir = os.path.join(args.output_dir, mode)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def evaluate(args, model, eval_dataset
    , mode, global_step = None):
    print("do eval")
    results = {}
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, 
        sampler = eval_sampler, batch_size = args.eval_batch_size)
    print_evaluate_metadata(mode, global_step, 
        len(eval_dataset), args, False)
    eval_loss, nb_eval_steps = 0.0, 0
    preds, out_label_ids = None, None
    for batch in tqdm(eval_dataloader, desc="Evaluating", disable=False):
        eval_loss, nb_eval_steps, preds, out_label_ids = \
            eval_iter(args, model, batch, eval_loss, 
                nb_eval_steps, preds, out_label_ids)
    eval_loss = eval_loss / nb_eval_steps
    results = { "loss": round(eval_loss, 2) }
    preds = clean_neutral_ambigous_in_preds(args, preds)
    preds = (preds == preds.max(axis = 1, keepdims = 1)).astype(int)

    result = compute_metrics(args, out_label_ids, preds)
    write_result_files(args, result, preds, out_label_ids)
    write_end_separation_to_labels_all(args)
    results.update(result)
    make_mode_subdir(args, mode, False)
    return results

def evaluate_checkpoints(args, test_dataset):
    if args.do_eval== False:
        return
    checkpoints = get_checkpoints(args)
    print_checkpoints(checkpoints, True)
    
    global_steps, loss, accuracy, prec, recall, f1 = \
        [0] * len(checkpoints), [0] * len(checkpoints), [0] * len(checkpoints), [0] * len(checkpoints), [0] * len(checkpoints), [0] * len(checkpoints)
    idx = 0
    for checkpoint in checkpoints:
        global_step = checkpoint.split("-")[-1]
        model = BertForMultiLabelClassification.from_pretrained(checkpoint)
        model.to(args.device)
        result = evaluate(args, model, test_dataset, mode="test", global_step = global_step)
        global_steps[idx], loss[idx], accuracy[idx], prec[idx], recall[idx], f1[idx] = \
            idx + 1, result["loss"], result["accuracy"], result["weighted_precision"], result["weighted_recall"], result["weighted_f1"]
        idx += 1
        # the example is in tmp_file dir and store_results only use in phased1 training
        # if args.store_results:
        #     store_checkpoints_list(global_steps, loss, accuracy, prec, recall, f1)

def main():
    print("run.py main()")
    args = setup_args()
    special_rules(args)
    set_output_dir_name(args)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    add_language_in_train_dev_test_files(args)
    if args.remove_past_checkpoints == True:
        remove_past_checkpoints(args)
    # init_logger()
    set_seed(args)
    label_list = get_label_list(args)
    print('after get_label_list')
    config = load_config(args, label_list)
    print('after load_config')
    tokenizer = load_tokenizer(args)
    print('after load_tokenizer')
    model = load_model_no_strict(args, config)
    print('after load_model_no_strict')
    cpu_or_gpu(args, model)
    print('after cpu_or_gpu')
    train_dataset, dev_dataset, test_dataset = \
        load_dataset(args, tokenizer)
    print(f'train_dataset: {len(list(train_dataset))}')
    if args.do_train:
        global_step, avg_loss = train(args, model, tokenizer, 
            train_dataset, dev_dataset, test_dataset)
        print_train_result(args, global_step, avg_loss, False)
    evaluate_checkpoints(args, test_dataset)

if __name__ == '__main__':
    main()

