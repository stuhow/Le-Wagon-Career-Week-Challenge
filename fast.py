import pickle
from fastapi import FastAPI

app = FastAPI()

filename = ''

app.state.model = pickle.load(open(filename, 'rb'))


@app.get("/")
def root():
    # $CHA_BEGIN
    return dict(greeting="Hello")
    # $CHA_END

@app.get('/predict')
def predict():

    y_pred = app.state.model.predict()

    print(y_pred)

    return {'prediction': 'test'}
