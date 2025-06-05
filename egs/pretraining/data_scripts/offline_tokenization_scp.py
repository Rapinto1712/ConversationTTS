# Author: # UniAudio Teams

import sys
import torch
import argparse
import logging
import tarfile
import mmap
import pickle
import librosa
from io import BytesIO
import io
from tools.tokenizer.MimiCodec.mimi_tokenizer import MimiTokenizer
from tools.tokenizer.MusicSemantic.Semantic_tokenizer import SemanticTokenizer
import json
import soundfile as sf
import torchaudio

def get_parser():
    parser = argparse.ArgumentParser(
        description="convert a data list, do tokenization and save as a torch .pt file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-file", type=str, default=None, help="text file in the format <exampe_id> <content>")
    parser.add_argument("--output-file", type=str, help="dict")
    parser.add_argument("--tokenizer", type=str, choices=['audio', 'g2p', 'Music', 'semantic', 'mimi', 'ssl'], help="what tokenizer to use")
    parser.add_argument("--rank", type=int, help="local GPU rank, if applicable")
    parser.add_argument("--batch-size", type=int, default=1, help="for batch tokenization")
    return parser

def main(args):
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.DEBUG,
        format=f"%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    )
    args = get_parser().parse_args(args)
    args.rank -= 1 # run.pl starts from 1 but the exact jobid / gpuid starts from 0   
    max_gpu = 4 # torch.cuda.device_count()
    logging.info(f"max gpu {max_gpu}")
    args.rank = (args.rank % max_gpu) #

    if args.tokenizer in ['audio', 'Music', 'encodec', 'mimi', 'ssl']:
        device = torch.device(f"cuda:{args.rank}")
    else:
        device = torch.device('cpu')
    logging.info(f"Using device: {device}")

    # GPU tokenizers 
    if args.tokenizer == "Music":
        tokenizer = SemanticTokenizer(device=device)
    elif args.tokenizer == 'mimi':
        tokenizer = MimiTokenizer(device=device)
    # CPU tokenizers
    else:
        raise NotImplementedError
    #tokenizer = tokenizer.to(device)
    logging.info('tokenizer built')
    data_dict = {}
    import time
    st_time = time.time()
    # assert not (args.input_file is not None)
    # TODO: support batch inference
    if args.input_file is not None:
        iterator = open(args.input_file)
        s_cnt = 0
        for i, line in enumerate(open(args.input_file)):
            try:
                line = line.strip().split()
                key, value_path = line[0], " ".join(line[1:])
                values = tokenizer.tokenize(value_path)
                # print('values ', values, values.shape)
                # assert 1==2
                data_dict[key] = values #.to(torch.int32)
                s_cnt += 1
                if i > 0 and i % 1000 == 0:
                    logging.info(f"processed {s_cnt} examples")
            except Exception as e:
                logging.error(f"an error instance: {line}, {e}")
    else:
        # kaldiio format
        iterator = ReadHelper('scp:'+args.wav_scp, args.segments)
        count = 0
        for key, (sr, value) in iterator:
            value = torch.from_numpy(value.copy())  / 32768 # [channel, samples]
            value = value.unsqueeze(0)
            value = tokenizer.tokenize(value)
            data_dict[key] = value
            if count > 0 and count % 100 == 0:
                logging.info(f"processed {count} examples")
            count += 1
    torch.save(data_dict, args.output_file)
    ed_time = time.time()
    print('ed_time-st_time ', ed_time-st_time)
    logging.info(f"processed {ed_time-st_time} seconds")

if __name__ == "__main__":
    main(sys.argv[1:])
