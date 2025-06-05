''' transfer the decode the text ID into text, then, add [id] tag. Lastly, convert it into tokens
'''
import json
import os
import sys
import torch
import torch.nn.functional as F
import argparse
import logging
from collections import defaultdict
import torchaudio
import random
#from tools.tokenizer.MimiCodec.mimi_tokenizer import MimiTokenizer
from tools.tokenizer.Text2ID.text_tokenizer import TextTokenizer


def get_parser():
    parser = argparse.ArgumentParser(
        description="convert a data list, do tokenization and save as a torch .pt file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-file", type=str, default=None, help="text.pt")
    parser.add_argument("--output-file", type=str, help="the output .pt file")
    parser.add_argument("--checkpoint_dir", type=str, help='the path of')
    parser.add_argument("--rank", type=int, help="local GPU rank, if applicable")
    return parser

def main(args):
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.DEBUG,
        format=f"%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    )
    args = get_parser().parse_args(args)
    args.rank -= 1 # run.pl starts from 1 but the exact jobid / gpuid starts from 0   
    max_gpu = torch.cuda.device_count()
    args.rank = (args.rank % max_gpu)
    data_dict = defaultdict(dict)
    text_tokenizer = TextTokenizer(checkpoint_dir=args.checkpoint_dir)
    logging.info('Text tokenizer built')
    result = []
    cnt = 0
    import time
    st_time = time.time()
    text_tokens = torch.load(args.input_file)
    for key in text_tokens.keys():
        org_text = text_tokenizer.decode(text_tokens[key])
        id_ = random.randint(0, 999)
        content = f'[{id_}]' + org_text
        ids = text_tokenizer.tokenize(content)
        # print('name, content, ids', name, content, ids)
        # assert 1==2
        ids = torch.Tensor(ids).to(torch.int32)
        cnt += 1
        data_dict[key] = ids
    torch.save(data_dict, args.output_file)
    ed_time = time.time()
    logging.info(f"processed {ed_time-st_time} seconds")


if __name__ == "__main__":
    main(sys.argv[1:])
    