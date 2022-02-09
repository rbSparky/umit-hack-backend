from flask import Flask, render_template, request, jsonify, session
import sklearn
from sklearn.decomposition import TruncatedSVD
import pandas as pd
import numpy as np
import pickle


app = Flask(__name__)


@app.route('/')
def hello():
    return 'hi'





if __name__ == '__main__': app.run(debug=True)
