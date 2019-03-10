import numpy as np


class Z_socre(object):
    def z_score(self,x):
        mean = np.mean(x, axis=0)
        std = np.std(x, axis=0)
        return (x-mean)/std,mean,std
    def Dz_score(self,x,mean,std):
        return x*std+mean
    def z_score_mean_std(self,x,mean,std):
        return (x-mean)/std

class MinMax(object):
    def min_max_norm(self,x):
        minn=x.min(0)
        maxx=x.max(0)
        return (x-minn)/(maxx-minn),minn,maxx

    def Dminmax(self,x,min,max):
        return x*(max-min)+min

    def minmax_withminmax(self,x,min,max):
        return (x-min)/(max-min)


class LLayer:

    def __init__(self,num_inputs,num_outputs,bias=None,weights=None):
        if bias:
            self.bias=bias
        else:
            self.bias=np.zeros(num_outputs)
        if weights:
            self.weight=weights
        else:
            self.weight=np.random.randn(num_inputs,num_outputs)*0.05
        self.bias=np.random.randn(1,num_outputs)*0.05
        self.grad=np.zeros((num_inputs,num_outputs))
        self.wgrad=np.zeros((num_inputs,num_outputs))
        self.bgrad=np.zeros(num_outputs)
        self.output=None
        self.input=None

class Layer:
    #非线性层的时候不用定义wight,bias
    def __init__(self,out=None,grad=None):
        self.output=out
        self.grad=grad
        self.input=None

#用于记录计算图
class Net:

    def __init__(self):
        self.graph=[] #用来存放计算图
        self.InitInput=None
        self.LR=0.00001
        self.target=None
        self._loss=[]
        self.momentum=1

    def Relu(self):#1
        l=Layer()
        self.graph.append((1,l))

    def ReluForward(self,input,i):
        input=np.array(input)
        self.graph[i][1].output=np.maximum(0,input)
        self.graph[i][1].grad = np.zeros(input.shape[0])
        return self.graph[i][1].output

    def ReluBackward(self,input,i,grad):
        mask=(input==self.graph[i][1].output)
        newgrad=np.zeros(mask.shape)
        newgrad[mask]=1
        self.graph[i][1].grad=grad*newgrad
        return self.graph[i][1].grad

    def LeaklyRelu(self):#2
        l=Layer()
        self.graph.append((2,l))

    def LeaklyReluForward(self,input,i):
        input=np.array(input)
        if self.graph[i][1]:
            self.graph[i][1].output=np.maximum(0.1*input,input)
            self.graph[i][1].grad = np.zeros(input.shape[0])
        return self.graph[i][1].output

    def LeaklyReluBackward(self,input,i,grad):
        mask = (input == self.graph[i][1].output)
        newgrad = np.zeros(input.shape)
        newgrad[mask] = 1
        newgrad[~mask]=0.1
        if len(grad.shape)==1:
            grad=np.expand_dims(grad,axis=1)
        else:
            grad = grad.sum(axis=1)
            grad=np.expand_dims(grad,axis=1)
        self.graph[i][1].grad=(grad.flatten()*newgrad).T
        return self.graph[i][1].grad

    def Sigmoid(self):#4
        l=Layer()
        self.graph.append((4,l))

    def SigmoidForward(self,input,i):
        input=np.array(input)
        self.graph[i][1].output=1/(np.exp(-input)+1)
        self.graph[i][1].grad=np.zeros(input.shape[0])
        return self.graph[i][1].output

    def SigmoidBackward(self,input,i,grad):
        temp=self.graph[i][1].output*(1-self.graph[i][1].output)
        self.graph[i][1].grad=grad*temp
        return self.graph[i][1].grad

    def MSELoss(self):#3
        l=Layer()
        self.graph.append((3,l))

    def MSELossForward(self,input,i,label):
        self.graph[i][1].output = (input-label)**2/2
        return self.graph[i][1].output

    def MSELossBackward(self,input,i,grad):
        if self.graph[i][1].output is not None:
            if len(input.shape)<=0:
                input=np.expand_dims(input,axis=0)
            self.graph[i][1].grad=(input-self.target)
            return self.graph[i][1].grad

    def LinearLayer(self,input,output,bias=None,weights=None):#0
        l=LLayer(input,output,bias,weights)
        self.graph.append((0,l))

    def LinearLayerForward(self,input,i):
        self.graph[i][1].output=np.dot(input,self.graph[i][1].weight)+self.graph[i][1].bias
        return self.graph[i][1].output

    def LinearLayerBackward(self,input,i,gra,flag=None):
        grad=gra.copy()
        input=np.array(input)
        grad=np.array(grad)
        self.graph[i][1].weight=np.array(self.graph[i][1].weight)
        self.graph[i][1].wgrad=np.dot(input.T,grad)
        self.graph[i][1].bgrad=grad
        self.graph[i][1].grad=np.dot(grad,self.graph[i][1].weight.T)
        return self.graph[i][1].grad

    def forward(self,input,label=None):
        self.InitInput=np.array(input)
        self.target=np.array(label)
        output=input
        output=np.array(output)
        if label is None:
            for i in range(len(self.graph)-1):
                if self.graph[i][0] == 0:
                    output = self.LinearLayerForward(output, i)
                elif self.graph[i][0] == 1:
                    output = self.ReluForward(output, i)
                elif self.graph[i][0] == 2:
                    output = self.LeaklyReluForward(output, i)
                elif self.graph[i][0] == 4:
                    output = self.SigmoidForward(output, i)
            return self.graph[-2][1].output
        else:
            for i in range(len(self.graph)):
                if self.graph[i][0]==0:
                    output=self.LinearLayerForward(output,i)
                elif self.graph[i][0]==1:
                    output=self.ReluForward(output,i)
                elif self.graph[i][0]==2:
                    output=self.LeaklyReluForward(output,i)
                elif self.graph[i][0]==3:
                    output=self.MSELossForward(output,i,label)
                elif self.graph[i][0]==4:
                    output=self.SigmoidForward(output,i)
            return np.sum(self.graph[-1][1].output),self.graph[-2][1].output

    def backward(self):
        grad=np.ones(self.InitInput.shape)
        if len(self.graph)<2:
            return
        if self.graph[-1][0]==3:
            grad=self.MSELossBackward(self.graph[-2][1].output,-1,grad)
            flag=True
        elif self.graph[-1][0]==0:
            grad=self.LinearLayerBackward(self.graph[-2][1].output,-1,grad)
        for i in range(len(self.graph)-2,-1,-1):
            if i==0:
                if self.graph[i][0]==0:
                    grad=self.LinearLayerBackward(self.InitInput,i,grad)
            else:
                if self.graph[i][0]==0:
                    grad=self.LinearLayerBackward(self.graph[i-1][1].output,i,grad,flag)
                    flag=False
                elif self.graph[i][0]==4:
                    grad=self.SigmoidBackward(self.graph[i-1][1].output,i,grad)
                elif self.graph[i][0]==1:
                    grad=self.ReluBackward(self.graph[i-1][1].output,i,grad)
                elif self.graph[i][0]==2:
                    grad=self.LeaklyReluBackward(self.graph[i-1][1].output,i,grad)
        for i in range(len(self.graph)):
            if self.graph[i][0]==0:

                self.graph[i][1].weight=(self.graph[i][1].weight*self.momentum-self.LR*self.graph[i][1].wgrad)
                self.graph[i][1].bias+=(-self.LR*self.graph[i][1].bgrad)
                self.graph[i][1].wgrad=np.zeros(self.graph[i][1].wgrad.shape)
                self.graph[i][1].bgrad = np.zeros(self.graph[i][1].bgrad.shape)

from tqdm import tqdm
class Square(object):

    def __init__(self,data,netData,lr=0.1,epoches=3000):
        self.net=Net()
        for i in range(len(netData) - 1):
            self.net.LinearLayer(netData[i], netData[i + 1])
            if i != len(netData) - 2:
                self.net.Relu()
        self.net.MSELoss()
        self.net.LR=lr
        self.loss=[]
        self.val_loss=[]
        self.train_pred=[]
        self.val_pred=[]
        x_train,y_train,x_val,y_val=data
        normx, self.mean_train, self.std_train = Z_socre().z_score(x_train)
        normy, self.mean_label, self.std_label = Z_socre().z_score(y_train)
        x_val=Z_socre().z_score_mean_std(x_val,self.mean_train,self.std_train)
        y_val=Z_socre().z_score_mean_std(y_val,self.mean_label,self.std_label)
        self.data=(normx,normy,x_val,y_val)
        self.epoches=epoches
        self.lossepoch=epoches//100

    def train(self):
        x_train,y_train,x_val,y_val=self.data
        batches=x_train.shape[0]
        batches_val=x_val.shape[0]
        for epoch in tqdm(range(self.epoches)):
            runningloss = 0
            for batch in range(batches):
                trainx = x_train[batch:batch + 1, :]
                trainy = y_train[batch:batch + 1, :]
                loss, out = self.net.forward(trainx, trainy)
                runningloss += loss
                self.net.backward()
            val_loss = 0.0
            for batch in range(batches_val):
                input = x_val[batch:batch + 1]
                label = y_val[batch:batch + 1]
                loss, pred = self.net.forward(input, label)
                val_loss += loss
            if epoch%self.lossepoch==0:
                self.loss.append(runningloss/batches)
                self.val_loss.append(val_loss/batches_val)

            if epoch%10==0:
                tmp_pred = []
                for batch in range(batches):
                    input = x_train[batch:batch + 1]
                    pred = self.net.forward(input)
                    input=Z_socre().Dz_score(input,self.mean_train,self.std_train)
                    pred=Z_socre().Dz_score(pred,self.mean_label,self.std_label)
                    tmp_pred.append((input.flatten()[0], pred.flatten()[0]))
                self.train_pred = tmp_pred
                tmp_pred = []
                for batch in range(batches):
                    input = x_val[batch:batch + 1]
                    pred = self.net.forward(input)
                    input = Z_socre().Dz_score(input, self.mean_train, self.std_train)
                    pred = Z_socre().Dz_score(pred, self.mean_label, self.std_label)
                    tmp_pred.append((input.flatten()[0], pred.flatten()[0]))
                    print(input,pred)
                self.val_pred = tmp_pred
    def getloss(self):
        idx = np.ndarray.tolist(np.linspace(0, len(self.loss) - 1, len(self.loss)))
        return idx,self.loss,self.val_loss
    def get_pred_curve(self):
        input_train = [item[0] for item in self.train_pred]
        pred_train = [item[1] for item in self.train_pred]
        input_val = [item[0] for item in self.val_pred]
        pred_val = [item[1] for item in self.val_pred]
        return (input_train,pred_train),(input_val,pred_val)
    def prepare_2ddata(self):
        train_x, train_y,val_x,val_y=self.data
        train_x=Z_socre().Dz_score(train_x,self.mean_train,self.std_train)
        train_y=Z_socre().Dz_score(train_y,self.mean_label,self.std_label)
        trainx=np.ndarray.tolist(train_x.flatten())
        trainy=np.ndarray.tolist(train_y.flatten())
        val_x = Z_socre().Dz_score(val_x, self.mean_train, self.std_train)
        val_y = Z_socre().Dz_score(val_y, self.mean_label, self.std_label)
        valx = np.ndarray.tolist(val_x.flatten())
        valy = np.ndarray.tolist(val_y.flatten())
        return (trainx,trainy),(valx,valy)
class Two(object):

    def __init__(self,data,netData,lr=0.1,epoches=3000):
        self.net=Net()
        for i in range(len(netData) - 1):
            self.net.LinearLayer(netData[i], netData[i + 1])
            if i != len(netData) - 2:
                self.net.Relu()
        self.net.MSELoss()
        self.net.LR=lr
        self.loss=[]
        self.valloss=[]
        self.pred=[]
        self.valpred=[]
        traindata,valdata=data
        self.x_axis,self.y_axis,self.trainxy,self.trainz=traindata
        self.valx_axis, self.valy_axis, self.valxy, self.valz = valdata

        self.x=self.trainxy[:,0]
        self.y=self.trainxy[:,1]
        self.valx=self.valxy[:,0]
        self.valy=self.valxy[:,1]
        self.epoches=epoches
        self.lossepoch=epoches//100
    def train(self):
        x_train,y_train=self.trainxy,self.trainz
        x_val,y_val=self.valxy,self.valz
        batches=x_train.shape[0]
        batchesval=x_val.shape[0]
        for epoch in tqdm(range(self.epoches)):
            runningloss = 0
            for batch in range(batches):
                trainx = x_train[batch:batch + 1, :]
                trainy = y_train[batch:batch + 1, :]
                loss, out = self.net.forward(trainx, trainy)
                runningloss += loss
                self.net.backward()
            valloss = 0
            for batch in range(batchesval):
                trainx = x_val[batch:batch + 1, :]
                trainy = y_val[batch:batch + 1, :]
                loss, out = self.net.forward(trainx, trainy)
                valloss += loss
            if epoch%self.lossepoch==0:
                self.loss.append(runningloss/batches)
                self.valloss.append(valloss / batchesval)
                #print(runningloss/batches)
            if epoch % 10 == 0:
                tmppred = []
                for row in range(batches):
                    input = x_train[row:row + 1]
                    pred = self.net.forward(input)
                    tmppred.append(pred)
                tmppred = np.array(tmppred)
                tmppred = tmppred.reshape(len(self.x_axis),len(self.y_axis) )
                self.pred=np.ndarray.tolist(tmppred)
                tmppred = []
                for row in range(batchesval):
                    input = x_val[row:row + 1]
                    pred = self.net.forward(input)
                    tmppred.append(pred)
                tmppred = np.array(tmppred)
                tmppred = tmppred.reshape(len(self.valx_axis), len(self.valy_axis))
                self.valpred = np.ndarray.tolist(tmppred)
    def getloss(self):
        idx = np.ndarray.tolist(np.linspace(0, len(self.loss) - 1, len(self.loss)))
        return idx,self.loss,self.valloss
    def prepare_3ddata(self):
        return np.ndarray.tolist(self.x.flatten()),np.ndarray.tolist(self.y.flatten()),np.ndarray.tolist(self.trainz.flatten()),np.ndarray.tolist(self.valx.flatten()),np.ndarray.tolist(self.valy.flatten()),np.ndarray.tolist(self.valz.flatten())
    def get_pred_surface(self):
        return np.ndarray.tolist(self.x_axis),np.ndarray.tolist(self.y_axis),self.pred,np.ndarray.tolist(self.valx_axis),np.ndarray.tolist(self.valy_axis),self.valpred

if __name__=='__main__':
    x = np.linspace(-100, 100, 100)
    y = x ** 2
    x = np.expand_dims(x, axis=1)
    y = np.expand_dims(y, axis=1)
    Squaretrain = Square((x, y))
    print(Squaretrain.prepare_2ddata())
