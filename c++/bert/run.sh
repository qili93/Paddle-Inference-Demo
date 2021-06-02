#!/bin/bash
set +x
set -e

work_path=$(dirname $(readlink -f $0))

# 1. compile
bash ${work_path}/compile.sh

# 2. download bert infernece model of sst2
if [ ! -d bert_infer_model ]; then
    echo -e "\033[31m Cannot find BERT infer model, please download and rename it into bert_infer_model dir. \033[0m"
    exit 0
fi

# 3. run
./build/bert_predict --model_file bert_infer_model/model.pdmodel --params_file bert_infer_model/model.pdiparams --use_gpu true
