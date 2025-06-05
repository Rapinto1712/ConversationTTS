## Main idea

## Env for Emilia process
Please refer to Emilia

## Env for Tokenization

You must use python>=3.10

'''
conda create -n RSTnet python=3.10
conda activate RSTnet
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install tqdm
pip install librosa==0.9.1
pip install matplotlib
pip install omegaconf 
pip install einops
pip install vector_quantize_pytorch
pip install tensorboard
pip install deepspeed
pip install peft
'''

### 1. preprocess the dataset

注意，tts数据和播客数据，都尽可能分成32的整数倍来处理。这样容易分，因为我们实验会用32卡GPU跑。需要确保每卡一份

#### 1.1 Broadcast Data

>Input: A folder containing conversational waveforms.  
Output (ALL=0 ~ ngpu-1):  
broadcast_data.ALL.json: Metadata containing paths to `tokens.ALL+1.pt`. Provided as  `--train_data_jsons` / `--valid_data_jsons` parameters when training.  
tokens.ALL+1.pt: Dict of interleaved text-audio codes. utt_name: Torch.tensor with shape [33, T_audio] and type torch.int16.

First, prepare environment, download pretraiend model checkpoints and modify the config file following step 0 and 1 of [Emilia](https://github.com/open-mmlab/Amphion/tree/main/preprocessors/Emilia#setup-steps-). The env.sh and config.json can be found at `data_scripts/emilia`. Make sure both AudioPipeline and RSTnet environments are ready.

Then, modify paths and hyperparameters in `prepare_broadcast_data.sh`.

```bash
# line 5~7
db_root=<path-to-source-wavs>
processed_metadata_root=<path-to-dump-metadata>
processed_audio_root=<path-to-dump-processed-audio>

# line 13~15
source <conda-dir>/etc/profile.d/conda.sh
export PYTHONPATH=$PYTHONPATH:<path-to-RSTnet/MLLM_v2>

# line 87 & 111
--checkpoint_dir <path-to-pretrained-LLM>

# hyperparameters
# line 62
--max_duration: maximum length of a sample session in seconds. Should properly set so as to not exceed GPU memory limit. Default: 120
```

Finally, run `prepare_broadcast_data.sh`. The dataloader will yield a token sequence with the following [B, 33, t_text+t_audio+2] shape:
```
<|begin_of_text|>[<spk_id1>] <text> ··· [<spk_id1>] <text><|end_of_text|><|text_emply_token|>···
                                                                         <semantic_tokens>···
                                                                         <acoustic_tokens>···
```

In the last stage, we provide codes for dataloader sainity check. If all things works fine, it will output the text stream in the terminal and the audio stream as a .wav file under `egs/pretraining/data_scripts`. Just check if they are aligned.

#### 1.2 TTS Data

TTS metadata vastly differs from each other. We provide an example of processing [a huggingface cantonese dataset](https://huggingface.co/datasets/alvanlii/cantonese-youtube) in `prepare_hf_tts_data.sh`.

>Input: A folder containing .parquet files.  
Output (ALL=0 ~ ngpu-1):  
tts_cantonese_data.ALL.json: Metadata containing paths to `audio_codec.ALL+1.pt` and `text.ALL+1.pt`. Provided as  `--train_data_jsons` / `--valid_data_jsons` parameters when training.  
audio_codec.ALL+1.pt: Dict of mimicodec audio codes. utt_name: Torch.tensor with shape [32, T_audio] and type torch.int16.
text.ALL+1.pt: Dict of text token IDs. utt_name: Torch.tensor with shape [T_text, ] and type torch.int32.

As in 1.1, we need to modify paths and hyperparameters in `prepare_hf_tts_data.sh`.

```bash
# line 5~6
db_root=<path-to-source-wavs>
processed_root=<path-to-dump-outputs>

# line 12~16
source <conda-dir>/etc/profile.d/conda.sh
export PYTHONPATH=$PYTHONPATH:<path-to-RSTnet/MLLM_v2>

# line 66 & 88
--checkpoint_dir <path-to-pretrained-LLM>
```

### 2. Pre-training

### 3. Post-training

### 4. inference

