# AggregateLMs

## Directory Structure
```
AggregateLMs/
├──model_roberta
│    ├──data_utils.py
│    ├──dataset.py
│    ├──finetuning_utils.py
│    ├──run_hyperparameter_search.py
│    ├──train_models.py
├──model_gpt2
│    ├──data_utils.py
│    ├──dataset.py
│    ├──finetuning_utils.py
│    ├──run_hyperparameter_search.py
│    ├──train_models.py
├──models
│    ├──Taskname(ex:BoolQ)
│        ├──model1
│        ├──model2
├──data
│    ├──SuperGLUE
```
## How to Run
- <b>Download data: </b><br>
   ```
   ./download_data.sh
   ```
   Or just 
   ```
   curl -s -J -L  https://dl.fbaipublicfiles.com/glue/superglue/data/v2/combined.zip -o ./super.zip
   unzip super.zip -d <data_dir>
   ```
- <b>Hyperparameter Search:</b> <br>
   For RoBERTa,
   ```
   python model_roberta/run_hyperparameter_search.py -d <data_dir> -o <checkpoint_dir>
   ```
   For GPT2,
   ```
   python model_gpt2/run_hyperparameter_search.py -d <data_dir> -o <checkpoint_dir>
   ```
- <b>Model Training Given Hyperparameters:</b> <br>
  For RoBERTa,
   ```
   python /scratch/pfi203/AggregateLMs/model_roberta/train_models.py -d <data_dir>-o /<checkpoint_dir> -s <saved_model dir> <hyperparams ex: -e 3 -t 16 -lr 0.005 -w 0.001 -n 3>

   ```
- <b>Evaluation of ensembled models:</b> <br>
  For RoBERTa,
  ```
   python model_roberta/run_bagging.py
  ```
