stage=1
stop_stage=6
ngpu=2

db_root=/mnt/users/hccl.local/jkzhao/data/cantonese/cantonese-youtube/data
processed_root=/mnt/users/hccl.local/jkzhao/projects/CSM/debug_data_tts

export CUDA_VISIBLE_DEVICES=6,7
available_gpus=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
echo "Available GPUs: $available_gpus"

source /mnt/users/hccl.local/jkzhao/softwares/miniconda3/etc/profile.d/conda.sh
# conda activate RSTnet
conda activate open-moshi

export PYTHONPATH=$PYTHONPATH:/mnt/users/hccl.local/jkzhao/projects/CSM

mkdir -p $processed_root
wav_scp=$processed_root/parquet.scp; [[ -f "$wav_scp" ]] && rm $wav_scp

# Prepare parquet.scp
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Prepare parquet.scp"
    find "${db_root}" -type f -name "*.parquet" | while read -r parquet_file; do
        id=$(basename $parquet_file .parquet)
        echo "$id $parquet_file" >> $processed_root/parquet.scp
    done
fi

# Split the $processed_root for $ngpu GPUs
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "split the $processed_root for $ngpu GPUs"
    mkdir -p $processed_root/${ngpu}splits
    # extra shuf to ensure balance across GPUs
    # So the generated $processed_root cannot be reproduced due to the shuffle randomness
    if [ -f $processed_root/parquet.scp.shuf ]; then
        rm -f $processed_root/parquet.scp.shuf
    fi
    
    cat $processed_root/parquet.scp | shuf >  $processed_root/parquet.scp.shuf
    split_scp=
    for n in `seq 1 $ngpu`; do
        split_scp="$split_scp $processed_root/${ngpu}splits/parquet.${n}.scp"
    done
    ../../tools/kaldi/utils/split_scp.pl $processed_root/parquet.scp.shuf $split_scp
fi

# NOTE: Something should be here for source separation if BGM exists

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "Audio Tokenization"
    ../../tools/kaldi/utils/run.pl JOB=1:$ngpu $processed_root/${ngpu}splits/log/mimi.JOB.log \
    python3 data_scripts/offline_codec_tokenization.py \
        --parquet-scp  $processed_root/${ngpu}splits/parquet.JOB.scp \
        --output-file  $processed_root/${ngpu}splits/audio_codec.JOB.pt \
        --output-text $processed_root/${ngpu}splits/text.JOB.scp \
        --tokenizer mimi --rank JOB || exit 1;
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "Prepare text sequence"
    ../../tools/kaldi/utils/run.pl JOB=1:$ngpu  $processed_root/${ngpu}splits/log/text_bpe.JOB.log \
    python  data_scripts/text_tokenization_scp.py \
        --rank JOB \
        --input-file  $processed_root/${ngpu}splits/text.JOB.scp \
        --checkpoint_dir /mnt/users/hccl.local/jkzhao/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B/snapshots/13afe5124825b4f3751f836b40dafda64c1ed062 \
        --output-file $processed_root/${ngpu}splits/text.JOB.pt || exit 1;
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "create $processed_root json"
    #mkdir -p $processed_root/${ngpu}splits
    for n in `seq 0 $[$ngpu-1]`; do
    python3 data_scripts/create_data_json.py \
        --task speechllm_v1 \
        --out-json $processed_root/${ngpu}splits/tts_cantonese_data.${n}.json \
        --audio_seq $processed_root/${ngpu}splits/audio_codec.$[$n+1].pt \
        --text_seq $processed_root/${ngpu}splits/text.$[$n+1].pt \
        & 
    done; wait
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "Dataloader test"
    python3 ../../utils/dataloader.py \
        --train_data_jsons $processed_root/${ngpu}splits/tts_cantonese_data.ALL.json \
        --valid_data_jsons $processed_root/${ngpu}splits/tts_cantonese_data.ALL.json \
        --checkpoint_path  /mnt/users/hccl.local/jkzhao/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B/snapshots/13afe5124825b4f3751f836b40dafda64c1ed062/lit_model.pth \
        --empty_token 0 \
        --pad_token 2050 \
        --semantic_eos 0 \
        --text_empty_token 0 \
        --text_pad_token 128002 \
        --parallel_number 33 \
        --audio_tokenizer 'mimi'
fi
