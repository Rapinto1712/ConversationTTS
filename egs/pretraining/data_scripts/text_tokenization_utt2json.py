''' transfer the text-only dataset into text token ID
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
#from tools.tokenizer.MimiCodec.mimi_tokenizer import MimiTokenizer
from tools.tokenizer.MimiCodec.mimi_tokenizer import MimiTokenizer
from tools.tokenizer.Text2ID.text_tokenizer import TextTokenizer
from utils.dataloader import Collate_Fn_Factory
FRAME_RATE = 12.5

def get_parser():
    parser = argparse.ArgumentParser(
        description="convert a data list, do tokenization and save as a torch .pt file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-file", type=str, default=None, help="utt2json in the format <exampe_id> <json_path>")
    parser.add_argument("--input-audio", type=str, default=None, help="the input audio codec .pt file")
    parser.add_argument("--output-file", type=str, help="the output .pt file")
    parser.add_argument("--checkpoint_dir", type=str, help='the path of')
    parser.add_argument("--rank", type=int, help="local GPU rank, if applicable")

    parser.add_argument("--text-empty-token", type=int, default=0, help="Token ID for empty text")
    parser.add_argument("--text-pad-token", type=int, default=128002, help="Token ID for text padding")
    parser.add_argument("--semantic-empty-token", type=int, default=0, help="Token ID for empty semantic")
    parser.add_argument("--semantic-pad-token", type=int, default=2050, help="Token ID for semantic padding")
    parser.add_argument("--semantic-eos", type=int, default=0, help="Token ID for semantic end-of-sequence")
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
    data_dict = {}
    text_tokenizer = TextTokenizer(checkpoint_dir=args.checkpoint_dir)
    logging.info('Text tokenizer built')
    if args.input_audio is not None:
        audio_codes_dict = torch.load(args.input_audio)
        parallel_number = list(audio_codes_dict.values())[0].shape[0] + 1
        tokenizers = {
            "audio": MimiTokenizer(),
            "text": text_tokenizer
        }
        collate_fn = Collate_Fn_Factory(
            tokenizers=tokenizers, 
            parallel_number=parallel_number,
            text_empty_token=args.text_empty_token,
            text_pad_token=args.text_pad_token,
            semantic_empty_token=args.semantic_empty_token,
            semantic_pad_token=args.semantic_pad_token,
            semantic_eos=args.semantic_eos
            )
    import time
    st_time = time.time()
    with open(args.input_file, "r") as utt2json:
        for line in utt2json:
            name, json_path = line.strip().split(' ')
            with open(json_path, "r") as f:
                session = json.load(f)

                # Interleaved pattern
                if args.input_audio is not None:
                    audio_codes = audio_codes_dict[name]
                    audio_codes = tokenizers['audio'].tokenize2(audio_codes)    # [N, T] -> [T, N]
                    turn_ids = []
                    start = 0
                    with open(json_path, "r") as f:
                        session = json.load(f)
                    
                        for idx, itm in enumerate(session["segments"]):
                            speaker_id = int(itm["speaker"].split("_")[-1])
                            text = f"[{speaker_id}]{itm['text']}"
                            if len(turn_ids) > 0:
                                text = " " + text
                            # Tokenize text
                            text_ids = text_tokenizer.tokenize(text)   # list(t_text), with bos and eos
                            text_ids = torch.Tensor(text_ids)
                            text_ids = collate_fn.text_pad(text_ids)
                            turn_ids.append(text_ids)   # [T, N]

                            # Segment audio
                            # NOTICE: all timestamps are absolute, so should be subtracted by session["start"]
                            if idx == len(session["segments"]) - 1:
                                end = itm["end"] - session["start"]
                            else:
                                end = (itm["end"] + session["segments"][idx+1]["start"]) / 2  - session["start"]
                            end = int(end * FRAME_RATE)
                            audio_ids = audio_codes[start:end]
                            start = end
                            eos_frame = torch.ones(1, audio_ids.shape[1]) * collate_fn.semantic_eos # add eos tokens
                            audio_ids = torch.cat([audio_ids, eos_frame], dim=0) # (T+1,4)
                            audio_ids = collate_fn.audio_pad(audio_ids)
                            turn_ids.append(audio_ids)
                    ids = torch.cat(turn_ids, dim=0).T  # [N, T]
                # Text only pattern
                else:
                    text = ""
                    for itm in session["segments"]:
                        speaker_id = int(itm["speaker"].split("_")[-1])
                        text += f" [{speaker_id}]{itm['text']}"
            
                    text = text.strip()
                    ids = text_tokenizer.tokenize(text)
            
            ids = torch.Tensor(ids).to(torch.int32)
            data_dict[name] = ids
    torch.save(data_dict, args.output_file)
    ed_time = time.time()
    logging.info(f"processed {ed_time-st_time} seconds")


if __name__ == "__main__":
    main(sys.argv[1:])
    