PYTHONPATH=python3
SEED=144 # 145 146 147 148
VERBALIZER=kpt #soft auto
FILTER=tfidf_filter # none
KPTWLR=0.0 # 0.06
MAXTOKENSPLIT=-1 # 1
MODEL_NAME_OR_PATH=""
RESULTPATH="results_normal"
OPENPROMPTPATH=""

mkdir ckpts

CUDA_LAUNCH_BLOCKING=1 $PYTHONPATH citeprompt.py \
--model_name_or_path $MODEL_NAME_OR_PATH \
--result_file $RESULTPATH \
--openprompt_path $OPENPROMPTPATH \
--result_file results_fewshot_norefine.txt \
--seed $SEED \
--verbalizer $VERBALIZER \
--max_token_split $MAXTOKENSPLIT \
--kptw_lr $KPTWLR
