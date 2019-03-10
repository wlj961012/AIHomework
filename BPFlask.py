from flask import Flask, render_template,jsonify,request
import networkx as nx
import matplotlib.pyplot as plt
import os
from PIL import Image
from io import BytesIO
import base64
import json
from GetNum import Net,Square,Two
import numpy as np

app = Flask(__name__)
app.debug = True
Squaretrain=None
Twotrain=None
Data=None
count=0

def getgraph(data):
    global count
    count+=1
    G = nx.DiGraph()
    draw_list = []
    list = data.copy()
    for i in range(len(list)):
        temp = []
        for j in range(list[i]):
            temp.append(str(i) + '-' + str(j))
        draw_list.append(temp)
    d = []
    for i in range(len(draw_list) - 1):
        for j in draw_list[i]:
            for k in draw_list[i + 1]:
                d.append((j, k))
    G.add_edges_from(d)
    options = {
        'node_color': 'grey',
        'node_size': 400,
        'width': 3,
        'edge_color': 'grey',
        'edge_size': 50,
        'with_labels': True,
        'font_color': 'white',
    }
    nx.draw(G, **options)
    path='path'+str(count)+'.png'
    print(path)
    plt.savefig(path)
    path=os.path.join(os.curdir,path)
    img=Image.open(path)
    buffered=BytesIO()
    img.save(buffered,format="PNG")
    img_str=base64.b64encode(buffered.getvalue())
    return img_str

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/getdata',methods=['POST'])
def getdata():
    global Data
    Data=request.get_json('a')
    for d in range(len(Data)):
        Data[d]=int(Data[d])
    img=getgraph(Data)
    print(Data)
    return img
########################################################平方###########################################

@app.route('/square')
def square():
    return render_template('square.html')

@app.route('/GetSquareData',methods=['POST'])
def getsquaredata():
    jsondata=request.get_data()
    params=json.loads(jsondata)
    x = np.linspace(params['Min'], params['Max'], params['Nums'])
    #print(params['Min'], params['Max'], params['Nums'])
    y = x ** 2
    x = np.expand_dims(x, axis=1)
    y = np.expand_dims(y, axis=1)
    x2 = np.sort(np.random.uniform(params['Min'], params['Max'], params['Nums']))
    val_x = np.expand_dims(x2, 1)
    val_y = val_x**2
    global Squaretrain
    Squaretrain=Square((x,y,val_x,val_y),netData=Data,lr=params['LearningRate'],epoches=params['Epoches'])
    traindata,valdata = Squaretrain.prepare_2ddata()
    train = dict(
        x=traindata[0],
        y=traindata[1],
        type='scatter',
        mode='markers',
    )
    val = dict(
        x=valdata[0],
        y=valdata[1],
        type='scatter',
        mode='markers'
    )
    return jsonify(Datatrain=train,Dataval=val)
@app.route('/getsquareTraincurve',methods=['GET'])
def getsquare_train_curve():
    if Squaretrain is not None:
        traincurve,valcurve=Squaretrain.get_pred_curve()
        datatrain = dict(
            x=traincurve[0],
            y=traincurve[1],
            type='plot',
            bnmode='lines'
        )
        dataval = dict(
            x=valcurve[0],
            y=valcurve[1],
            type='plot',
            mode='lines'
        )
    else:
        datatrain = dict(
            x=[],
            y=[],
            type='plot',
            mode='lines'
        )
        dataval = dict(
            x=[],
            y=[],
            type='plot',
            mode='lines'
        )
    return jsonify(Datatrain=datatrain,Dataval=dataval)


@app.route('/BeginSquareTrain',methods=['POST'])
def beginsquaretrain():
    Squaretrain.train()
    return jsonify(Data='200')

@app.route('/getsquareLoss')
def squareloss():
    if Squaretrain is not None:
        x, ty, vy = Squaretrain.getloss()
        traindata = dict(
            x=x,
            y=ty,
            type='plot',
            mode='lines',
        )
        valdata = dict(
            x=x,
            y=vy,
            type='plot',
            mode='lines',
        )
    else:
        traindata = dict(
            x=[],
            y=[],
            type='plot',
            mode='lines'
        )
        valdata = dict(
            x=[],
            y=[],
            type='plot',
            mode='lines',
        )
    return jsonify(trainloss=traindata,valloss=valdata)
###########################################sin(x)############################################################
@app.route('/sin')
def sin():
    return render_template('sin.html')

@app.route('/GetSinData',methods=['POST'])
def getsindata():
    jsondata=request.get_data()
    params=json.loads(jsondata)
    x = np.linspace(params['Min'], params['Max'], params['Nums'])
    y = np.sin(x*np.pi)
    x = np.expand_dims(x, axis=1)
    y = np.expand_dims(y, axis=1)
    x2 = np.sort(np.random.uniform(params['Min'], params['Max'], params['Nums']))
    val_x = np.expand_dims(x2, 1)
    val_y = np.sin(val_x*np.pi)
    global Squaretrain
    Squaretrain=Square((x,y,val_x,val_y),netData=Data,lr=params['LearningRate'],epoches=params['Epoches'])
    traindata, valdata = Squaretrain.prepare_2ddata()
    train = dict(
        x=traindata[0],
        y=traindata[1],
        type='scatter',
        mode='markers',
    )
    val = dict(
        x=valdata[0],
        y=valdata[1],
        type='scatter',
        mode='markers'
    )
    return jsonify(Datatrain=train, Dataval=val)

@app.route('/BeginSinTrain',methods=['POST'])
def beginsintrain():
    Squaretrain.train()
    return jsonify(Data='200')

@app.route('/getsinLoss')
def sinloss():
    if Squaretrain is not None:
        x, ty, vy = Squaretrain.getloss()
        traindata = dict(
            x=x,
            y=ty,
            type='plot',
            mode='lines',
        )
        valdata = dict(
            x=x,
            y=vy,
            type='plot',
            mode='lines',
        )
    else:
        traindata = dict(
            x=[],
            y=[],
            type='plot',
            mode='lines'
        )
        valdata = dict(
            x=[],
            y=[],
            type='plot',
            mode='lines',
        )
    return jsonify(trainloss=traindata,valloss=valdata)
@app.route('/getsinTraincurve',methods=['GET'])
def getsin_train_curve():
    if Squaretrain is not None:
        traincurve,valcurve=Squaretrain.get_pred_curve()
        datatrain = dict(
            x=traincurve[0],
            y=traincurve[1],
            type='plot',
            bnmode='lines'
        )
        dataval = dict(
            x=valcurve[0],
            y=valcurve[1],
            type='plot',
            mode='lines'
        )
    else:
        datatrain = dict(
            x=[],
            y=[],
            type='plot',
            mode='lines'
        )
        dataval = dict(
            x=[],
            y=[],
            type='plot',
            mode='lines'
        )
    return jsonify(Datatrain=datatrain,Dataval=dataval)
###############################################two#################################################
@app.route('/two')
def two():
    return render_template('two.html')

@app.route('/GetTwoData',methods=['POST'])
def gettwodata():
    jsondata=request.get_data()
    params=json.loads(jsondata)
    x = np.linspace(params['Min'], params['Max'], params['Nums'])
    y = np.linspace(params['Min2'],params['Max2'],params['Nums2'])
    X,Y=np.meshgrid(x,y)
    trainx=np.dstack((X,Y)).reshape(-1,2)
    trainy=(trainx[:,0]+trainx[:,1]).reshape(-1,1)

    val_x = np.sort(np.random.uniform(params['Min'], params['Max'],  params['Nums']))
    val_y = np.sort(np.random.uniform(params['Min'], params['Max'],  params['Nums']))
    valX, valY = np.meshgrid(val_x, val_y)
    valx = np.dstack((valX, valY)).reshape(-1, 2)
    valy = (valx[:, 0] + valx[:, 1]).reshape(-1, 1)

    global Twotrain
    Twotrain=Two(((x,y,trainx,trainy),(val_x,val_y,valx,valy)),netData=Data,lr=params['LearningRate'],epoches=params['Epoches'])
    x, y, z,valx,valy,valz = Twotrain.prepare_3ddata()
    data = dict(
        x=x,
        y=y,
        z=z,
        type='scatter3d',
        mode='markers',
        marker=dict(
            size=4
        )
    )
    valdata = dict(
        x=valx,
        y=valy,
        z=valz,
        type='scatter3d',
        mode='markers',
        marker=dict(
            size=4
        )
    )
    return jsonify(Data=data,Valdata=valdata)
@app.route('/get3dTrainsurface',methods=['GET'])
def get3d_train_surface():
    if Twotrain is not None:
        x,y,z,vx,vy,vz=Twotrain.get_pred_surface()
    else:
        x=y=z=vx=vy=vz=[]
    data=dict(
        type='surface',
        x=x,
        y=y,
        z=z
    )
    valdata = dict(
        type='surface',
        x=vx,
        y=vy,
        z=vz
    )
    return jsonify(Data=data,Valdata=valdata)

@app.route('/BeginTwoTrain',methods=['POST'])
def begintwotrain():
    Twotrain.train()
    return jsonify(Data='200')

@app.route('/gettwoLoss')
def twoloss():
    if Twotrain is not None:
        x,tl,vl=Twotrain.getloss()
        data = dict(
            x=x,
            y=tl,
            type='plot',
            mode='lines'
        )
        valdata = dict(
            x=x,
            y=vl,
            type='plot',
            mode='lines'
        )
    else:
        data = dict(
            x=[],
            y=[],
            type='plot',
            mode='lines'
        )
        valdata = dict(
            x=[],
            y=[],
            type='plot',
            mode='lines'
        )
    return jsonify(trainloss=data,valloss=valdata)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=4500, threaded=True)
