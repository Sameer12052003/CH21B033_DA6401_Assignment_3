# CH21B033_DA6401_Assignment_3

## Name : Sameer Deshpande

## Roll No : CH21B033

This repository contains the implemenatation of assignment 3 of the course 'Introduction to Deep Learning' DA6401.

It trains a custom RNN, LSTM AND GRU models, validates and tests them on the dakshina dataset for transliteration task. We consider transliteration from Latin (English) to Devanagari Hindi script.

## Folder Strucuture
<pre> 
CH21B033_DA6401_Assignment_3/ 

├── dakshina_dataset_v1.0/                # Contains dakshina dataset.

├── data_preprocessing/
    ├── data_preprocess.py                # Contains necessary functions & creates pytorch dataset

├── predictions_attention/
    ├── predictions_attention.tsv         # Predictions on test set by attention-based model 

├── predictions_vanilla/
    ├── predictions_vanilla.tsv           # Predictions on test set by vanilla model 

├── seq2seqmodel/
    ├── model.py                          # Contains Seq2Seq model architecture code 

├── best_model_with_attention.py          
# Trains attention-based model, saves predictions in predictions_attention, plots attention heatmap and attention connectivity visualization
 
├── best_model_without_attention.py          
# Trains vanilla model, saves predictions in predictions_vanilla and plots prediction grid (3*3)
                         
├── README.md                                   # Project overview and instructions 

├── requirements.txt                            # Python dependencies 

├── test_accuracy_attention.txt                 # Contains test accuracy by attention-based model

├── test_accuracy.txt                           # Contains test accuracy by vanilla model

├── wandb_sweep.py                         
# Hyperparameter tuning for vanilla and attention-based models  </pre>

## Installation Instructions

1\) Clone the repository:
```bash
git clone https://github.com/Sameer12052003/CH21B033_DA6401_Assignment_3.git
cd CH21B033_DA6401_Assignment_3
```

2\) Please create a python virtual environment: 
```bash
python -m venv venv
```

3\) Activate the python environment:
```bash
source venv/Scripts/activate
```

4\) Install all the required dependencies:
```bash
pip install -r requirements.txt
```


### Usage Instructions

```bash
# To run the sweep code for hyperparameter tuning
export PYTHONPATH
python wandb_sweep.py

# To train and test the best vanilla model and plot visualizations
export PYTHONPATH
python best_model_without_attention.py

# To train and test the best attention-based model and plot visualizations
export PYTHONPATH
python best_model_with_attention.py

```

## Results

Vanilla model:

| Metric            | Accuracy (%)         |
|-------------------|----------------------|
| Val Accuracy      | 38.13                |
| Test Accuracy     | 37.34                |

Attention-based model:

| Metric            | Accuracy (%)         |
|-------------------|----------------------|
| Val Accuracy      | 40.08                |
| Test Accuracy     | 38.58                |
