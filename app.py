import pickle
from flask import Flask, request, jsonify, session
from flask_cors import CORS, cross_origin
import sklearn
from sklearn.decomposition import TruncatedSVD
import pandas as pd
import numpy as np


ranks = []
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

class Model:
  i = '0'
  lrank = 1
  hrank = 2
  r_names = None
  r_ID = None
  corr_ID = None
  recc = None
  cmat = []
  fir = []
  sec = []
  final = []

  def predict(self):
    self.sec = []
    self.fir = []
    flf = []
    SVD = TruncatedSVD(n_components = 10)
    decompm = SVD.fit_transform(self.cmat)
    df = pd.DataFrame(decompm)
    corrm = np.corrcoef(decompm)
    p_names = list(self.cmat.index)
    p_ID = p_names.index(str(self.i))
    c_ID = corrm[p_ID]
    Recommend = list(self.cmat.index[c_ID > 0.95])
    fl = []
    for i in range(len(c_ID)):
      if(c_ID[i] > 0.95):
        fl.append([c_ID[i], self.cmat.index[i]])
    fl.sort(reverse=True)
    flf = []
    for i in range(len(fl)):
      if (fl[i][0] > 0.95):
        flf.append(fl[i])

    clgdf, clgds = {}, {}
    for i in flf:
      for j in self.cmat.loc[i[1]].items():
        if ((j[1] == 5) and (j[0] not in self.fir) and (ranks[j[0]][1] >= self.lrank)):# and (self.hrank >= ranks[j[0]][0])):
          if(j[0] in clgdf):
            clgdf[j[0]] += 1
          else:
            clgdf[j[0]] = 1

          
        elif ((j[1] == 2) and (j[0] not in self.sec) and (ranks[j[0]][1] >= self.lrank)):# and (self.hrank >= ranks[j[0]][0])):
          if(j[0] in clgds):
            clgds[j[0]] += 1
          else:
            clgds[j[0]] = 1            

    tf, ts = [], []
    for k in clgdf:
      tf.append([clgdf[k], k])
    for k in clgds:
      ts.append([clgds[k], k])

    tf.sort(reverse=True)
    ts.sort(reverse=True)

    for i in tf:
      j = i[1]
      self.fir.append([(j.split())[0], j[len((j.split())[0]):], ranks[j][0], ranks[j][1]])

    
    for i in ts:
      j = i[1]
      self.sec.append([(j.split())[0], j[len((j.split())[0]):], ranks[j][0], ranks[j][1]])

    #print(self.fir, self.sec, sep = '\n\n\n')
    self.final = []
    for i in self.fir:
        self.final.append(i)
    for i in self.sec:
        self.final.append(i)
    rfinal = []
    [rfinal.append(x) for x in self.final if x not in rfinal]
    return jsonify(rfinal)

@app.route('/')
def hello():
  return 'hi main'

@app.route('/predict', methods=['POST','GET']) #or POST u see that
@cross_origin()
def predict():
  #take all these as input from args
  global ranks

  #session.clear()
  
  req_dat = request.get_json()
  lrank = req_dat['lrank']#5000
  hrank = req_dat['hrank']#7000
  stream1 = req_dat['stream1']#'Computer Science'
  stream2 = req_dat['stream2']#'Electronics'
  '''
  lrank = int(request.args.get("lrank"))
  hrank = int(request.args.get("hrank"))
  stream1 = request.args.get("p1")
  stream2 = request.args.get("p2")
  '''
  f = open('essentials.pckl', 'rb')
  f1 = pickle.load(f)
  f.close()
  #print(f1)

  f = open('Model2.pckl', 'rb')
  f2 = pickle.load(f)
  f.close()

  wsc = f1[0]
  ranks = f1[1]
  f2.cmat = f1[2]
  
  f2.lrank = lrank
  f2.hrank = hrank
  f2.stream1 = stream1
  f2.stream2 = stream2
  f2.final = []
  f2.i = wsc[(stream1, stream2)]

  return f2.predict()

if __name__ == '__main__':
  #app.secret_key = 'super secret key'
  #app.config['SESSION_TYPE'] = 'filesystem'
  #session.init_app(app)
  app.run(debug=True)
