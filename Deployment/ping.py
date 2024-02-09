
from flask import Flask

app= Flask('ping')

@app.route('/ping',methods=['Get'])
def ping():
    return "Welcome to Churn Detection"

if __name__ == "__main__":   
    app.run(debug=True,host='0.0.0.0',port=9696)