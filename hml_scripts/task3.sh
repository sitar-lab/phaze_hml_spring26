
# run the gpt3.sh graph populate estimates step with your configurations
cd ..

# Llama2 -7b
python3 phaze.py \
        --phaze_model llama2 \
        --phaze_exec_type run_solver \
        --phaze_micro_batch_size 1 2 4 8 \
        --phaze_sequence_length 4096 \
        --phaze_max_tmp_width 1 \
        --phaze_hbm_size 80
