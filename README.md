# AggregateLMs

## Directory Structure
- Current Directory:
- model_roberta
- model_gpt2
## How to Run
- Hyperparameter Search:
   For RoBERTa,
   '''
   python model_roberta/run_hyperparameter_search.py -d <data_dir> -o <checkpoint_dir>
   '''
   For GPT2,
   '''
   python model_gpt2/run_hyperparameter_search.py -d <data_dir> -o <checkpoint_dir>
   '''
- Model Training Given Hyperparameters:
  For RoBERTa,
   '''
   python /scratch/pfi203/AggregateLMs/model_roberta/train_models.py -d <data_dir>-o /<checkpoint_dir> -s <saved_model dir> <hyperparams ex: -e 3 -t 16 -lr 0.005 -w 0.001 -n 3>

   '''
   For GPT2,
   '''
   python model_gpt2/run_hyperparameter_search.py -d <data_dir> -o <checkpoint_dir>
   '''
- Evaluation:
