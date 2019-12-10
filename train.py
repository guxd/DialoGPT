#  Copyright (c) Microsoft Corporation. 
#  Licensed under the MIT license. 
'''
 * @Desc: train GPT2 from scratch/ fine tuning. Modified based on Huggingface GPT-2 implementation
'''

import json
import os
import sys
import argparse
import logging
import time
import tqdm
import datetime
import torch
from collections import defaultdict
import numpy as np
from os.path import join

from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, AdamW
from transformers import get_linear_schedule_with_warmup

from data_loader import END_OF_TEXT_TOKEN
from data_loader import BucketingDataLoader, DynamicBatchingLoader, DistributedBucketingDataLoader
from data_loader import (InputFeatures, InputFeatures_train, RedditExample)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

INF = 100000000
EVAL_STEP = 100000

########################################################################################################
###### Train Utils ###################
SEQ_LENGTH_SHRINK_PROP = 0.9

def boolean_string(s):
    if s.lower() not in {'false', 'true'}: raise ValueError('Not a valid boolean string')
    return s.lower() == 'true'

def get_eval_list_same_length(input_file, tokenizer, max_batch_size, norm=True):
    examples = []
    with open(input_file, 'r', encoding="utf-8") as f:
        content = [l.split('\t') for l in f.read().splitlines()]

    context, response = [c[0] for c in content], [c[1:] for c in content]
    i = 0
    for src, tgt_all in zip(context, response):
        for tgt in tgt_all:
            if norm:
                src_line = ' '.join(src.strip().split())
                tgt_line = ' '.join(tgt.strip().split())
            else:
                src_line = src.strip()
                tgt_line = tgt.strip()
            examples.append(RedditExample(i, src_line, tgt_line))
            i += 1
    def featurize(example):
        conv_id = example.conv_id
        context_id = tokenizer.encode(example.context)
        end_of_text_id = tokenizer.encoder[END_OF_TEXT_TOKEN]
        response_id = tokenizer.encode(example.response)
        input_ids = context_id + [end_of_text_id]
        lm_labels = response_id
        position_ids = list(range(len(input_ids)))
        token_type_id = [0] * len(input_ids)

        return InputFeatures(conv_id, input_ids, position_ids, token_type_id, lm_labels, len(context_id), len(response_id))

    def batch_feature_same_len(features):
        input_ids = torch.stack([torch.tensor(f.choices_features['input_ids'], dtype=torch.long) for f in features])
        position_ids = torch.stack([torch.tensor(f.choices_features['position_ids'], dtype=torch.long) for f in features])
        token_type_ids = torch.stack([torch.tensor(f.choices_features['token_type_ids'], dtype=torch.long) for f in features])
        labels = torch.nn.utils.rnn.pad_sequence([torch.tensor(f.lm_labels, dtype=torch.long) for f in features],
            batch_first=True, padding_value=-1)
        context_len = torch.tensor([f.context_len for f in features], dtype=torch.long)
        response_len = torch.tensor([f.response_len for f in features], dtype=torch.long)
        return (input_ids, position_ids, token_type_ids, labels, context_len, response_len)

    features = [featurize(e) for e in examples]
    dataloader_pre = defaultdict(list)
    for f in features:
        dataloader_pre[f.context_len].append(f)

    dataloader = []
    for l in sorted(dataloader_pre):
        f = batch_feature_same_len(dataloader_pre[l])
        if len(f[0]) <= max_batch_size:
            dataloader.append(f)
        else:
            start_index = 0
            while True:
                dataloader.append([ff[start_index:start_index + max_batch_size] for ff in f])
                start_index += max_batch_size
                if start_index >= len(f[0]): break
    return dataloader

#### Eval Utils ######

#from pycocoevalcap.bleu.bleu import Bleu
EOS_ID = 50256

def cal_BLEU_4(generated, reference, is_corpus=False):
    BLEUscore = [0.0, 0.0, 0.0, 0.0]
    for idx, g in enumerate(generated):
        if is_corpus:
            score, scores = Bleu(4).compute_score(reference, {0: [g]})
        else:
            score, scores = Bleu(4).compute_score({0: [reference[0][idx]]}, {0: [g]})
        for i, s in zip([0, 1, 2, 3], score): BLEUscore[i] += s
    BLEUscore[0] = BLEUscore[0]/len(generated)
    BLEUscore[1] = BLEUscore[1]/len(generated)
    BLEUscore[2] = BLEUscore[2]/len(generated)
    BLEUscore[3] = BLEUscore[3]/len(generated)
    return BLEUscore

def cal_entropy(generated):
    etp_score = [0.0, 0.0, 0.0, 0.0]
    div_score = [0.0, 0.0, 0.0, 0.0]
    counter = [defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int)]
    for gg in generated:
        g = gg.rstrip().split()
        for n in range(4):
            for idx in range(len(g)-n):
                ngram = ' '.join(g[idx:idx+n+1])
                counter[n][ngram] += 1
    for n in range(4):
        total = sum(counter[n].values()) + 1e-10
        for v in counter[n].values():
            etp_score[n] += - (v+0.0) / total * (np.log(v+0.0) - np.log(total))
        div_score[n] = (len(counter[n].values())+0.0) / total
    return etp_score, div_score

#######################################################################################################################

def train(args, train_dataloader, model, tokenizer, train_logger, eval_logger):
    
    t_total = args.num_optim_steps
        
    no_decay = ['bias', 'ln']   # no decay for bias and LayerNorm (ln)
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
        
    global_step = 0
    step = 0
    epoch = 0

    if args.continue_from:
        global_step = args.continue_from
        step = global_step*2 - 1

    if args.local_rank != -1: n_gpu = 1
    if args.local_rank == -1 or torch.distributed.get_rank() == 0:
        pbar = tqdm.tqdm(total=args.num_optim_steps, desc=f"training") if args.pbar else None

    while True:
        model.train()
        (tr_loss, nb_tr_examples, nb_tr_steps) = 0.0, 0, 0
        n_token_real, n_token_total = 0, 0
        train_start_time_epoch = time.time()
        for batch in train_dataloader:
            batch = tuple(t.to(args.device) for t in batch)
            input_ids, position_ids, token_ids, label_ids, *_ = batch
            if args.no_token_id: token_ids = None
            loss, *_ = model(input_ids, None, None, token_ids, position_ids, None, None, label_ids)
       
            if args.n_gpu > 1:
                loss = loss.mean()
            loss = loss / (args.train_batch_size / input_ids.shape[0])
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += float(loss.item()) * (args.train_batch_size / input_ids.shape[0])
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            mean_loss = tr_loss / nb_tr_steps

            n_token_total += input_ids.shape[0] * input_ids.shape[1]
            n_token_real += (input_ids != 0).sum().item()

            # gradient update
            step += 1
            if step % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # Print log info to file
                if args.local_rank != -1:
                    n_token_real_all_proc = sum(all_gather_list(n_token_real))
                    n_token_total_all_proc = sum(all_gather_list(n_token_total))
                else:
                    n_token_real_all_proc = n_token_real
                    n_token_total_all_proc = n_token_total

                if args.local_rank == -1 or torch.distributed.get_rank() == 0:
                    epoch_time = time.time() - train_start_time_epoch
                    if pbar is not None:
                        pbar.set_postfix_str(
                            f"tok/s: {n_token_real_all_proc//epoch_time//1000}k epoch: {epoch}")
                        pbar.update(1)
                    print(f'{epoch+1},{global_step+1},{step+1},{mean_loss},\
                          {n_token_real_all_proc},{n_token_total_all_proc},{epoch_time}',
                        file=train_logger)

                if global_step % args.valid_step == 0:
                    if args.local_rank == -1 or torch.distributed.get_rank() == 0:
                        # only rank 0 process evaluate
                        torch.save(
                            {k: (v.cpu() if v is not None else None)  # save to cpu tensors
                             for k, v in model.state_dict().items()},
                            join(output_dir, f'GP2-pretrain-step-{global_step}.pkl'))

                        eval_loss = evaluate(model, tokenizer, epoch, args)
                        # enable generation step evaluation for now
                        # gen_response = generation(model, tokenizer, epoch, args)
                        '''
                        # probably use beam search only for test set
                        if False:
                            gen_response_beam = generation(model, tokenizer, epoch, args, use_beam_search=True, beam_width=3)
                        '''
                        print('{},{},{},{},{}'.format(epoch+1, global_step+1, step+1, eval_loss), file=eval_logger)
                        logger.info('current learning rate: '+ str(optimizer.param_groups[0]['lr']))
                        model.train()
                if global_step >= args.num_optim_steps: break

        if global_step >= args.num_optim_steps: break
        epoch += 1


    if args.local_rank == -1 or torch.distributed.get_rank() == 0:
        if pbar is not None: pbar.close()
        train_logger.close()
        eval_logger.close()    


def evaluate(model, tokenizer, epoch_id, args):
    # use the same signature with eval_model_generation
    logger.info('compute eval model loss, using eval mode, please change it back to train after calling this function')
    model.eval()
    
    eval_dataloader = DynamicBatchingLoader(args.eval_input_file, tokenizer, args.normalize_data, args.eval_batch_size, args.max_seq_length)
        
    tot_loss = []
    tot_sample = []
    with torch.no_grad():
        for step, batch in enumerate(eval_dataloader):
            batch = tuple(t.to(args.device) for t in batch)
            input_ids, position_ids, token_ids, label_ids, src_len, _ = batch
            if args.no_token_id: token_ids = None
            n_sample = input_ids.shape[0]
            loss = model(input_ids, position_ids, token_ids, label_ids)
            tot_loss.append(loss.mean().item() * n_sample)
            tot_sample.append(n_sample)
    print(f"\n Epoch {epoch_id}: Val loss {np.sum(tot_loss) / np.sum(tot_sample)}  ")
    return np.sum(tot_loss) / np.sum(tot_sample)

def generation(model, tokenizer, epoch, args):
    gen_dataloader = get_eval_list_same_length(args.eval_input_file, tokenizer, args.eval_batch_size, True)
    return ''

##############################################################################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='gpt2', help='pretrained model name or path to local checkpoint')
    parser.add_argument("--train_input_file", type=str, default='data/train.128len.db')
    parser.add_argument("--eval_input_file", type=str, default='./data/dummy_data.tsv')
    parser.add_argument("--output_dir", type=str, default='output')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_seq_length", type=int, default=128)

    parser.add_argument("--skip_eval", action='store_true', help='If true, skip evaluation.')
    
    parser.add_argument("--continue_from", type=int, default=0)

    parser.add_argument("--train_batch_size", type=int, default=4, help="batch size now means per GPU per step")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="to increase effective batch size and reduce synchronization")
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--num_optim_steps", type=int, default=1000000, help="new API specifies num update steps")
    parser.add_argument("--valid_step", type=int, default=10000, help="how many optim steps between validations")
    parser.add_argument("--warmup_proportion", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=16000)

    parser.add_argument("--normalize_data", type=boolean_string, default=True)
    parser.add_argument("--fp16", type=boolean_string, default=True)
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--lr_schedule", type=str, choices=['noam', 'noamwd', 'BERT', 'None'], default='noam')
    parser.add_argument("--loss_scale", type=float, default=0)
    parser.add_argument("--no_token_id", type=boolean_string, default=True)
    
    parser.add_argument("--log_dir", type=str)
    parser.add_argument('--pbar', type=boolean_string, default=True, help='turn on progress bar')

    # distributed
    parser.add_argument('--local_rank', type=int, default=-1, help='for torch.distributed')

    args = parser.parse_args()

    assert args.train_batch_size % args.gradient_accumulation_steps == 0, 'batch size % gradient accumulation steps != 0!'
    args.train_batch_size = (args.train_batch_size// args.gradient_accumulation_steps)
    logger.info(f'train batch size = {args.train_batch_size*args.gradient_accumulation_steps}, '
                'new train batch size (after gradient accumulation) = {args.train_batch_size}')

    if args.local_rank == -1:
        logger.info(f'CUDA available? {str(torch.cuda.is_available())}')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
        args.device, args.n_gpu = device, n_gpu
    else:
        # distributed training
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
        n_gpu = torch.distributed.get_world_size()
        args.device, args.n_gpu = device, 1
        logger.info(f"device: {device} n_gpu: {n_gpu}, distributed training: {bool(args.local_rank != -1)},16-bits training: {args.fp16}")

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    if n_gpu > 0: torch.cuda.manual_seed_all(args.seed)

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d%H%M%S')
    output_dir = join(args.output_dir, 'GPT2.{}.{}.{}gpu.{}'.format(args.learning_rate, args.train_batch_size, n_gpu, timestamp))
    log_dir = args.log_dir if args.log_dir is not None and len(args.log_dir) > 0 else output_dir
    if args.local_rank == -1 or torch.distributed.get_rank() == 0: 
        os.makedirs(output_dir, exist_ok=True)
        train_logger = open(join(log_dir, 'train_log.txt'), 'a+', buffering=1)
        eval_logger = open(join(log_dir, 'eval_log.txt'), 'a+', buffering=1)
        print('epoch,global_step,step,mean_loss,n_token_real,n_token_total,epoch_time', file=train_logger)
        print('epoch,global_step,step,eval_loss', file=eval_logger)

    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name_or_path)
    config = GPT2Config.from_pretrained(args.model_name_or_path)
    
    if args.local_rank == -1:
        train_dataloader = BucketingDataLoader(args.train_input_file, args.train_batch_size, args.max_seq_length)
    else:
        train_dataloader = DistributedBucketingDataLoader(
            torch.distributed.get_rank(), torch.distributed.get_world_size(), 
            args.train_input_file, args.train_batch_size, args.max_seq_length)

    model = GPT2LMHeadModel.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config)
    model = model.to(args.device)
    
    global_step, tr_loss = train(args, train_dataloader, model, tokenizer, train_logger, eval_logger)
    
    
if __name__ == "__main__":
    main()