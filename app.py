import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import preprocessing as pp

app = Flask(__name__)
model = pickle.load(open('LR_BOW_Classifier.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    data = request.form.get("input_text")
    print("data is ", data)
    # data = "this is brilliant and I'm very excited to recommend this"
    

    print("Data received is: ", data)
    text_stripped_html = pp.strip_html_tags(data)
    text_nopunct = pp.remove_punctuations(text_stripped_html)
    text_noaccented_chars = pp.remove_accented_chars(text_nopunct)
    text_nospecial_chars = pp.remove_special_characters(text_noaccented_chars)
    text_expanded_contractions = pp.expand_contractions(text_nospecial_chars)
    text_lemmatized = pp.lemmatize_text(text_expanded_contractions)
    text_nostopwords = pp.remove_stopwords(text_lemmatized)

    # text = np.array([text_nostopwords])
    text_bow_transformed = pp.bow_transform(text_nostopwords)

    pred = model.predict(text_bow_transformed)
    sentiment = ""
    if(pred==0):
        sentiment = "Negative"
    else:
        sentiment = "Positive"
    
 
    prediction_text='Sentiment expressed: {}'.format(sentiment) 




    

    return render_template('index.html', prediction_text=prediction_text)

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)