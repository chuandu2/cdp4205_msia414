# Text-Analytics-Independent-Project
@Sophie Du


## Project Overview
- Topic: Sentence Similarity

- Goal: 
  
  Some Q&A platforms such as Quora has presented a need for an effective natural language processing method to compute the similarity between sentences or texts. Questions that are worded similarly but with the same meanings shall be collected together to save time for seekers in finding the best answers to their questions and also help writers to avoid providing answers for multiple versions of the same questions. This project will explore methods of computing similarity between two sentences to identify whether they are duplicates or not.

- Dataset: 

  The dataset used here is originally from Quora. It contains more than 400k rows and 6 variables, including ‘id’: the id of a question pair; ‘qid1’,‘qid2’: unique ids for each question; ‘question1’, ‘question2’: full text of each question in the pair; ‘is_duplicate’: the target variable, to determine if the questions in the pair are with the same meaning (Yes: 1, No: 0), human judgements are brought in for this variable. The dataset will be used to get a best performing binary predictive model for identifying sentences with similar meanings based on model evaluation results.


## Repo structure 
```
├── README.md                         <- You are here
│
├── api.py                            <- Main python code to run the app
│
├── best_model.sav                    <- The best performing predictive model selected from experiments
│
├── EDA_Feature_Experiments.ipynb     <- Jupyter notebook containing EDA, feature generation and model experiments
│
├── requirements.txt                  <- Python package dependencies 
```

## Running the application 
In this part, you need to enter two questions (strings) to `q1={ENTER YOUR 1ST SENTENCE}` and `q2={ENTER YOUR 2ND SENTENCE}` as user inputs (you can ignore the question mark when typing), an example will be given in part 2 below. Then the app will predict the probability that the two given sentences are duplicates using the XGBoost model saved in best_model.sav

### 1. Set up environment 
The `requirements.txt` file contains the packages required to run the prediction. An environment can be set up after you cd to the repo path. 

#### With `virtualenv`

```bash
pip install virtualenv
virtualenv sim_sent
source sim_sent/bin/activate
pip install -r requirements.txt
python -m spacy download en
```

### 2. Run the application
- Run api.py
 ```bash
python api.py
 ```

- Open a new terminal window and change directory to the repo path
 ```bash
curl -X GET http://127.0.0.1:5000/ -d q1='How can I reduce my belly fat through a diet?' -d q2='How can I reduce my lower belly fat in one month?'
 ```
 
The predicted probability of the 2 given sentences being with the same meaning fill be printed in terminal and saved in repo path as a json file.

For example, with the example input above, the returned output will be like

{
    "1st question": "How can I reduce my belly fat through a diet?",
    "2nd question": "How can I reduce my lower belly fat in one month?",
    "The probability of the two input sentences being with same meaning": "0.3899"
}
