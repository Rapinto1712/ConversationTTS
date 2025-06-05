# the core scripts for SpeechLM
. ./path.sh # set the path
stage=8
stop_stage=8
ngpu=8 # how many GPUs, you want to use to train the model

train_set="train"
valid_set="val"
test_sets="test"

# Dataset paths
data_root=/data6/ydc/tts_exp/exp
experiment_name='tts_val_1' # set the experiments name

# training config
seed=999

batch_scale=2000
learning_rate=0.0001
tag="test"

# Prepare data following Espnet and split
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Prepare prepare wav.scp, text.scp" 
fi

# Split the data for $ngpu GPUs
# This is done before data preprocessing such that multiple GPUs can be used for data preprocessing
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "split the data for $ngpu GPUs"
    for part in $valid_set $train_set; do
        mkdir -p $data_root/${part}/${ngpu}splits
        # extra shuf to ensure balance across GPUs
        # So the generated data cannot be reproduced due to the shuffle randomness
        cat $data_root/${part}/wav.scp | shuf >  $data_root/${part}/wav.scp.shuf
        split_scp=
        for n in `seq 1 $ngpu`; do
            split_scp="$split_scp $data_root/${part}/${ngpu}splits/wav.${n}.scp"
        done
        utils/split_scp.pl $data_root/${part}/wav.scp.shuf $split_scp

        for n in `seq 1 $ngpu`; do
          python3 data_scripts/filter_scp.py \
            $data_root/${part}/${ngpu}splits/wav.${n}.scp \
            $data_root/${part}/text.scp \
            $data_root/${part}/${ngpu}splits/text.${n}.scp &
        done; wait
    done

fi

# Data Preprocessing
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "Prepare text sequence"
    for part in $train_set $valid_set; do
        utils/run.pl JOB=1:$ngpu  $data_root/${part}/${ngpu}splits/log/text_bpe_new.JOB.log \
        python  data_scripts/text_tokenization_scp.py \
            --rank JOB \
            --input-file $data_root/${part}/${ngpu}splits/text.JOB.scp \
            --checkpoint_dir /data6/ydc/ckpts/Llama-3.2-1B  \
            --output-file $data_root/${part}/${ngpu}splits/text_2.JOB.pt
    done
    
fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "Prepare audio sequence"
    for part in $valid_set $train_set; do
        utils/run.pl JOB=1:$ngpu  $data_root/${part}/${ngpu}splits/log/audio_tokenizer.JOB.log \
        python  data_scripts/offline_tokenization_scp.py \
            --rank JOB \
            --input-file  $data_root/${part}/${ngpu}splits/wav.JOB.scp \
            --output-file $data_root/${part}/${ngpu}splits/audio_tokens.JOB.pt  \
            --tokenizer 'mimi'
    done
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "Prepare text token reverse"
    ngpu=64 # set the number of GPU (多少.pt文件)
    for part in $train_set; do
        utils/run.pl JOB=1:$ngpu ./log/text_token_reverse.JOB.log \
        python  data_scripts/text_tokenization_reverse.py \
            --rank JOB \
            --input-file /turing_music_fs/music_data/ydc/exp_data/speech_lm/wenet/wenetspeech_process/train/64splits/text_2.JOB.pt \
            --output-file /turing_music_fs/music_data/ydc/exp_data/speech_lm/wenet/wenetspeech_process/train/64splits/text_reverse.JOB.pt \
            --checkpoint_dir /turing_music_fs/music_data/ydc/checkpoints/llama3_2  
    done
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "create data json for normal tts task"
    ngpu=64
    for n in `seq 0 $[$ngpu-1]`; do
        python3 data_scripts/create_data_json.py \
            --task speechllm_v1 \
            --out-json   exp_data/speech_lm/wenet/wenetspeech_process/train/64splits/data.${n}.json \
            --audio_seq  exp_data/speech_lm/wenet/wenetspeech_process/train/64splits/audio_tokens.$[$n+1].pt \
            --text_seq  exp_data/speech_lm/wenet/wenetspeech_process/train/64splits/text_reverse.$[$n+1].pt \
            &
    done
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    echo "create data json for podcast task"
    ngpu=32
    for part in $train_set; do
      for n in `seq 0 $[$ngpu-1]`; do
        python3 data_scripts/create_data_json.py \
         --task moshi \
         --out-json   exp_data/test_data/tmp2/podcast_metadata_packed/32splits/data.${n}.json \
         --hybrid_seq  exp_data/test_data/tmp2/podcast_metadata_packed/32splits/tokens.$[$n+1].pt \
         & 
      done; wait
    done
fi


### Stage 6: Training ###

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    if [ -z $tag ]; then
        echo "please provide a tag for this experiment" && exit 1;
    fi
    echo "stage 7: training..."
    #export CUDA_VISIBLE_DEVICES=0 #5,6,7,
    export HOST_GPU_NUM=8 # set the number of GPU to use
    export HOST_NUM=1 # the number of nodes
    export NODE_NUM=1
    export INDEX=0
    export CHIEF_IP="localhost"
    # export CUDA_LAUNCH_BLOCKING=1

    train_data_jsons="exp_data/test_data/tmp2/cn_podcast/32splits/data.ALL.json"
    
    mimi_codec_path='checkpoints/moshi/tokenizer-e351c8d8-checkpoint125.safetensors'
    llama3_2_path='/root/code2/CSM_v2/llama3_2' # can be found in the root of the repo
    exp_dir="exp_data/speech_lm/exp_v3" # 存储ckpt的路径

    NCCL_DEBUG=TRACE python3 -m torch.distributed.run --nproc_per_node=$HOST_GPU_NUM --nnodes=$HOST_NUM --master_addr=$CHIEF_IP --master_port=4397  \
            --node_rank=$INDEX ../../trainer/pre_training.py \
            --train_data_jsons $train_data_jsons \
            --exp_dir $exp_dir \
            --n_epoch 4  \
            --max_length 2048  \
            --min_length 10 \
            --batch_scale 7500 \
            --learning_rate 1e-5 \
            --checkpoint_path $llama3_2_path \
            --mimi_codec_path $mimi_codec_path \
            --audio_card 2051 \
            --empty_token 0 \
            --pad_token 2050 \
            --semantic_eos 0 \
            --text_empty_token 0 \
            --text_pad_token 128002 \
            --parallel_number 33 \
            --audio_tokenizer 'mimi' \
            --save_interval 10000  \
            --grad_accum 1 \
            --print_freq 100 \
            --grad_clip 1.0 
fi

