import argparse
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    parser = argparse.ArgumentParser()
    # args for randomness
    parser.add_argument('--seed', type=int, default=2048, help='seed for initializing training. ')
    parser.add_argument('--cudnn_deterministic', default=False, action='store_true', help='set cudnn.deterministic True')
    # args for data
    parser.add_argument('--train_data_jsons', type=str, nargs="+", help="list of train data jsons, separated by comma,")
    parser.add_argument('--valid_data_jsons', type=str, nargs="+", help="list of valid data jsons, separated by comma,")
    parser.add_argument('--batch_scale', type=int, default=1000, help="summed sequence length of each batch")
    parser.add_argument('--max_length', type=int, default=1000, help="maximum length of each example sequence. -1 means no constraint. The real allowed length may exceed this slightly")
    parser.add_argument('--min_length', type=int, default=100, help="minimum length of each example sequence. -1 means no constraint. The real allowed length may exceed this slightly")
    parser.add_argument('--n_worker', type=int, default=4, help='number of loading workers for each GPU')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--training_dtype', type=int, default=-1)
    parser.add_argument('--minibatch_debug', type=int, default=-1, help="if > 0, chuncate the data iterator for debug")
    
    # args for local model
    parser.add_argument('--audio_card', type=int, default=2050, help='the audio token space of LLM')

    # args for save model and log: 
    parser.add_argument('--parallel_number', type=int, default=9, help='the number of training streaming')
    parser.add_argument('--empty_token', type=int, default=1024, help='the empty token for semantic')
    parser.add_argument('--pad_token', type=int, default=1025, help='the pading token for semantic')
    parser.add_argument('--semantic_eos', type=int, default=10000, help='the eos token for semantic')
    parser.add_argument('--text_empty_token', type=int, default=1024, help='the number of training streaming')
    parser.add_argument('--text_pad_token', type=int, default=1025, help='the number of training streaming')
    parser.add_argument('--exp_dir', type=str, default='./log', help='directory of this experiment')
    parser.add_argument('--model_config', type=str, default='configs/llama3.yaml', help='the config file for LLM')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints/meta-llama/Meta-Llama-3-8B-Instruct/lit_model.pth')
    parser.add_argument('--mimi_codec_path', type=str, default="checkpoints/moshi/tokenizer-e351c8d8-checkpoint125.safetensors", help='the path of mimi codec')
    parser.add_argument('--print_freq', type=int, default=100, help='the print frequency')
    parser.add_argument('--save_interval', type=int, default=10000, help='save a checkpoint within an epoch')
    parser.add_argument('--resume', type=str, default=None, help='whether re-train model')

    # args for training / optimization
    parser.add_argument('--n_epoch', type=int, default=500, help='Total training epoch')
    parser.add_argument('--grad_accum', type=int, default=1, help='help to simulate large batch')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='The learning rate for training')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='clip gradients at this value, or disable if == 0.0')
    parser.add_argument('--warmup_steps', type=int, default=10000, help="step of warmup")
    parser.add_argument('--schedule', type=str, default='cosine', help="the schedule strategy of training")
    parser.add_argument("--weight-decay", type=float, default=5e-2, help="Weight decay to use")
    parser.add_argument("--beta1", type=float, default=0.9, help="beta1 parameter for the Adam optimizer")
    parser.add_argument("--beta2", type=float, default=0.95, help="beta2 parameter for the Adam optimizer")
    parser.add_argument('--data-parallel', type=str, default='fsdp', help='data parallel strategy: fsdp, sdp, hsdp. ')
    parser.add_argument('--mixed-precision', type=str, default='bf16', help='mixed precision: fp32, tf32, bf16, fp16')
    parser.add_argument('--grad-precision', type=str, default='bf16', help='gradient precision: fp32, tf32, bf16, fp16')
    parser.add_argument('--activation-checkpointing', type=bool, default=True, help='use activation checkpointing')
    parser.add_argument("--no-wandb", type=str2bool, default='true', help='whether use wandb')

    # dataloader config
    parser.add_argument('--audio_tokenizer', type=str, default='semantic', help='the type of audio tokenizer')
    parser.add_argument('--text_tokenizer', type=str, default='llama3-8B', help='the type of audio tokenizer')
    args = parser.parse_args()
    
    return args