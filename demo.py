#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT license.
#
# Please assign the DATA_FOLDER before running this scripts, the data, pre-trained model, fine-tuned model will be
# downloaded automatically to DATA_FOLDER

import os,sys
import logging
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default = './data', help = 'data foler')
parser.add_argument('--data_type', type=str, default='small', help='choose from dummy, small and full')
args = parser.parse_args()

assert args.data_type in ['dummy', 'small', 'full'] , 'The specified data option is not support!'

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

#########################################################################
# Download Data
#########################################################################
import subprocess

logger.info('Downloading and Extracting Data...')
myCmd = os.popen('cd reddit_extractor; make -j 8; cd ..').read()
cmd = 'gzip -d ./train.tsv.gz'
ret = subprocess.run(cmd.split(' '), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=args.data_path)

#########################################################################
# Prepare Data
#########################################################################

logger.info('Preparing Data...')
data_path = os.path.join(args.data_path, 'train.tsv')
MAX_LEN = 128
cmd = ['python', 'prepare_data.py', '--corpus', data_path, '--max_seq_len', f'{MAX_LEN}']
print(' '.join(cmd))
ret = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

#data_db = data_path[:-4] +'.db'
data_db = f'{data_path[:-4]}.{MAX_LEN}len.db'

logger.info('Done!\n')

#########################################################################
# Train !
#########################################################################
logger.info('Generating training CMD!')
logger.info('If there is any problem, please copy (modify) and run command below')
logger.info('#########################################################################')
train_cmd = [
    'python',
    'train.py',
    '--model_name_or_path', 'gpt2',
    '--train_input_file', data_db ,  # file from last step
    '--eval_input_file', './data/dummy_data.tsv',   # dummy test data
    '--output_dir', os.path.join(args.model_path, 'output_model'),
    '--seed', '42',
    '--max_seq_length', '128',
    '--train_batch_size', '512',
    '--gradient_accumulation_steps', '8',
    '--eval_batch_size', '64',
    '--learning_rate', '1e-5',
    '--num_optim_steps', '10000',
    '--valid_step', '5000',
    '--warmup_steps', '4000',
    '--normalize_data', 'true',
    '--fp16', 'true',
    '--lr_schedule', 'noam',
    '--loss_scale', '0.0',
    '--no_token_id', 'true',
    '--pbar', 'true'
]

print(' '.join(train_cmd))
logger.info('#########################################################################')
f_log = open('./output.log', 'wb') 
process = subprocess.Popen(train_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
for line in iter(process.stdout.readline, b''): 
    sys.stdout.write(line.decode(sys.stdout.encoding)) 
    f_log.write(line)
f_log.close()
logger.info('Done!\n')
