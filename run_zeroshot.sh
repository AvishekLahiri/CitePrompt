PYTHONPATH=python3
DATASET=acl_arc #acl_arc 
TEMPLATEID=0 # 1 2 3
SEED=100 # 145 146 147 148
SHOT=0 # 0 1 10 20
VERBALIZER=kpt #
CALIBRATION="--calibration" # ""
FILTER=tfidf_filter # none
MODEL_NAME_OR_PATH=""
RESULTPATH="results_zeroshot"
OPENPROMPTPATH=""


$PYTHONPATH zeroshot.py \
--model_name_or_path $MODEL_NAME_OR_PATH \
--result_file $RESULTPATH \
--openprompt_path $OPENPROMPTPATH \
--dataset $DATASET \
--seed $SEED \
--verbalizer $VERBALIZER $CALIBRATION \
--filter $FILTER

