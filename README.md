### Team DOMLIN: Exploititing Evidence Enhancement for the FEVER Shared Task

pytorch implementation of our hand in for the FEVER 2.0 shared task (domlin_fever)

original hand in can be found here: https://github.com/dominiksinsaarland/domlin_fever


### Requirements
* Python 3.6
* AllenNLP
* pytorch

### installation

* Download and install Anaconda (https://www.anaconda.com/)
* Create a Python Environment and activate it:
```bash 
conda create -n domlin_fever_pytorch python=3.6
source activate domlin_fever_pytorch
conda install pytorch torchvision cudatoolkit==9.0 -c pytorch
pip install .
pip install -r ./examples/requirements.txt
```

### Domlin FEVER

Download original hand in (and follow installation instructions in https://github.com/dominiksinsaarland/domlin_fever)

```bash 
cd ..
git clone https://github.com/dominiksinsaarland/domlin_fever.git
cd domlin_fever_pytorch
```

run steps in original hand-in, but replace code for training the models (and evaluating models) with the pytorch equivalent, e.g. replace 

```bash 
# train the model (maybe set CUDA_VISIBLE_DEVICES and nohup, takes a while)
python src/domlin/run_fever.py --task_name=ir --do_train=true --do_eval=false --do_predict=true \
--path_to_train_file=fever_data/sentence_retrieval_1_training_set.tsv --vocab_file=cased_L-12_H-768_A-12/vocab.txt\
--bert_config_file=cased_L-12_H-768_A-12/bert_config.json --output_dir=fever_models/sentence_retrieval_part_1 --max_seq_length=128\
--do_lower_case=False --learning_rate=2e-5 --train_batch_size=32 --num_train_epochs=2 \
--init_checkpoint=cased_L-12_H-768_A-12/bert_model.ckpt --use_hingeloss=yes --negative_samples=4 \
--file_test_results=fever_data/dev_set_sentences_predicted_part_1.tsv --prediction_file=fever_data/sentence_retrieval_1_dev_set.tsv
```

etc. with the pytorch equivalent

### First sentence retrieval module
```bash 
mkdir ../pytorch_fever_models
TASK_NAME="first_sentence_retrieval"
filename_test_results="../domlin_fever/fever_data/dev_set_sentences_predicted_part_1_pytorch.tsv"
CUDA_VISIBLE_DEVICES=2 python run_fever.py \
    --model_type bert \
    --model_name_or_path bert-base-cased \
    --task_name=$TASK_NAME \
    --do_train \
    --do_eval \
    --data_dir /raid/dost01/domlin_fever/fever_data \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=32   \
    --per_gpu_train_batch_size=32   \
    --learning_rate 2e-5 \
    --num_train_epochs 2 \
    --output_dir ../pytorch_fever_models/$TASK_NAME \
    --vocab_file ../bert/bert/cased_L-12_H-768_A-12/vocab.txt \
    --prediction_file sentence_retrieval_1_dev_set.tsv \
    --filename_test_results=$filename_test_results

# pytorch transformers by default generates outfiles with at least two columns, we need only the first one
cut -f1 $filename_test_results
```


### Second sentence retrieval module
```bash 
TASK_NAME="second_sentence_retrieval"

CUDA_VISIBLE_DEVICES=2 python run_fever.py \
    --model_type bert \
    --model_name_or_path bert-base-cased \
    --task_name=$TASK_NAME \
    --do_train \
    --do_eval \
    --data_dir /raid/dost01/domlin_fever/fever_data \
    --max_seq_length 256 \
    --per_gpu_eval_batch_size=16   \
    --per_gpu_train_batch_size=16   \
    --learning_rate 2e-5 \
    --num_train_epochs 2 \
    --output_dir ../pytorch_fever_models/$TASK_NAME \
    --vocab_file ../bert/bert/cased_L-12_H-768_A-12/vocab.txt \
    --prediction_file sentence_retrieval_2_dev_set.tsv \
    --filename_test_results ../domlin_fever/fever_data/dev_set_sentences_predicted_part_2_pytorch.tsv
```

### RTE module

```bash 
TASK_NAME="rte"

CUDA_VISIBLE_DEVICES=2 python run_fever.py \
    --model_type bert \
    --model_name_or_path bert-base-cased \
    --task_name=$TASK_NAME \
    --do_train \
    --do_eval \
    --data_dir /raid/dost01/domlin_fever/fever_data \
    --max_seq_length 370 \
    --per_gpu_eval_batch_size=12   \
    --per_gpu_train_batch_size=12   \
    --learning_rate 3e-5 \
    --num_train_epochs 3 \
    --output_dir ../pytorch_fever_models/$TASK_NAME \
    --vocab_file ../bert/bert/cased_L-12_H-768_A-12/vocab.txt \
    --prediction_file RTE_dev_set.py \
    --filename_test_results ../domlin_fever/fever_data/RTE_dev_set_predicted.tsv
```


### Questions

If anything should not work or is unclear, please don't hesitate to contact the authors

* Dominik Stammbach (dominik.stammbach@bluewin.ch)

