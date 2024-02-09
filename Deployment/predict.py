import pickle
from flask import Flask
from flask import request
from flask import jsonify
output_file='model.bin'
print('opening File')

with open(output_file, 'rb') as f_in:
    dv,model= pickle.load(f_in) 

print('Model loaded')

app= Flask('churn')

@app.route('/predict',methods=['POST'])

def predict():
    customer=request.get_json()
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    churn=y_pred >= 0.5
    result={
        'churn_Prob':float(y_pred),
        'churn':bool(churn)

    }
    return jsonify(result)

if __name__ == "__main__":   
    app.run(debug=True,host='0.0.0.0',port=9696)


# print('Predicting Results')
# print('input:', customer)
# print(' ')
# print(' ')
# print('output:', y_pred)