
# run the gpt3.sh graph generation task 
cd ..

if [ ! -d "GraphExtractor/out/GPT" ]; then
  mkdir GraphExtractor/out/GPT
fi

# GPT3
python3 phaze.py \
        --phaze_model megatrongpt3 \
        --phaze_exec_type extract_graph \
        --phaze_micro_batch_size 8 \
        --phaze_sequence_length 2048 \
        --phaze_max_tmp_width 8 \
        --phaze_hbm_size 64
