from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import pandas as pd
import numpy as np
import sklearn
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
import json

app = Flask(__name__)
api = Api(app)

# parsing args
parser = reqparse.RequestParser()
parser.add_argument('q1')
parser.add_argument('q2')

class Predict_Prob_Duplicate(Resource):
    
    def get_weight(self, count, eps=10000, min_count=2):
        for count in range(2):
            return 0
        else:
            return 1 / (count + eps)

    def word_share(self, row):
        q1words = {}
        q2words = {}
        for word in str(row['question1']).lower().split():
            if word not in stop:
                q1words[word] = 1
        for word in str(row['question2']).lower().split():
            if word not in stop:
                q2words[word] = 1
        if len(q1words) == 0 or len(q2words) == 0:
            # The computer-generated chaff includes a few questions that are nothing but stopwords
            return 0
        shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
        shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
        R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
        return R

    def tfidf_word_match_share(self, row, weights):
        q1words = {}
        q2words = {}
        for word in str(row['question1']).lower().split():
            if word not in stop:
                q1words[word] = 1
        for word in str(row['question2']).lower().split():
            if word not in stop:
                q2words[word] = 1
        if len(q1words) == 0 or len(q2words) == 0:
            # The computer-generated chaff includes a few questions that are nothing but stopwords
            return 0
        # take weights in here
        shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in q2words.keys() if w in q1words]
        total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]

        R = np.sum(shared_weights) / np.sum(total_weights)
        return R

    def get(self):
        # use parser to find user's input
        args = parser.parse_args()
        q1_in = args['q1'] #str
        q2_in = args['q2']
        # collect data into df
        qs_in = pd.DataFrame({'test_id': [999999], 'question1': [q1_in], 'question2': [q2_in]})
        # calculate weights
        txt_qs = pd.Series(qs_in.question1.tolist()+qs_in.question2.tolist()).astype(str)
        words = (' '.join(txt_qs)).lower().split()
        counts = Counter(words)
        eps = 5000
        # weights = {word: self.get_weight(count) for word, count in counts.items()}
        weights = {word: 1/(count+eps) for word, count in counts.items()}
        # initialize dataset for prediction
        qs_in_test = pd.DataFrame()
        # add features
        qs_in_test['word_match'] = qs_in.apply(self.word_share, axis=1, raw=True)
        # qs_in_test['tfidf_word_match'] = qs_in.apply(self.tfidf_word_match_share, axis=1, raw=True)
        qs_in_test['tfidf_word_match'] = self.tfidf_word_match_share(row=qs_in[qs_in.test_id==999999], weights=weights)

        # load the model from disk
        bst = pickle.load(open('best_model.sav', 'rb'))
        # predict results
        d_test = xgb.DMatrix(qs_in_test)
        p_test = str(round(bst.predict(d_test)[0], 4))
        # output result
        print('The probability of the two input sentences being with same meaning is: ', p_test)
        # save prediction to kson
        with open('predict_result.json', 'w') as out:
            json.dump(p_test, out)
        output = {'1st question': str(q1_in), '2nd question': str(q2_in), 
        'The probability of the two input sentences being with same meaning': p_test,
        'weights': weights}
        return output

# set up API resource routing
api.add_resource(Predict_Prob_Duplicate, '/')

if __name__ == '__main__':
    # add stop words for later use
    stop = set(stopwords.words("english"))

    app.run(debug=True)