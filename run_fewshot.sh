PYTHONPATH=python3
SEED=100 # 145 146 147 148
SHOT=10 # 1 2 5 10 20
VERBALIZER=kpt #soft auto
FILTER=tfidf_filter # none
KPTWLR=0.0 # 0.06
MAXTOKENSPLIT=-1 # 1
MODEL_NAME_OR_PATH=""
RESULTPATH="results_fewshot"
OPENPROMPTPATH=""

$PYTHONPATH fewshot.py \
--model_name_or_path $MODEL_NAME_OR_PATH \
--result_file $RESULTPATH \
--openprompt_path $OPENPROMPTPATH \
--result_file results_fewshot_norefine.txt \
--seed $SEED \
--shot $SHOT \
--verbalizer $VERBALIZER \
--max_token_split $MAXTOKENSPLIT \
--kptw_lr $KPTWLR
