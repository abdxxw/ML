#########################################################
############BY Hider TOULA & Abdellah BOUSBA ############
######################## IMPORTS ########################

import numpy as np
import pandas as pd
import collections as ct
import graphviz as gr
import matplotlib.pyplot as plt
from IPython.display import Image
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier as DTree
from sklearn.model_selection import KFold
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mltools import plot_data, plot_frontiere, make_grid, gen_arti
from sklearn import linear_model, neighbors, tree, svm
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.datasets import mnist
from sklearn.multiclass import OneVsRestClassifier 
from sklearn.multiclass import OneVsOneClassifier 
from sklearn.linear_model import Perceptron
import re


######################## TME 1 ########################



#EXO1

##############################


listD = [2, 3, 5, 10, 15, 20, 25, 30]

#Q1.

def entropie(vect):
    
    vect = np.asarray(vect)
    nb = ct.Counter(vect)
    nb = np.asarray(list(nb.values()))
    proba = nb / vect.size
    
    return -np.sum(proba * np.log(proba), axis=0)

#Q2.
    
def entropie_cond(list_vect):
    
    out = 0.0
    size = 0.0

    for v in list_vect:
        out += len(v) * entropie(v)
        size+=len(v)
        
    return out/size


#Q3


def getEntropies(datax,datay,fields):
    e_attr = []
    ec_attr = []

    for i in range(datax.shape[1]):
        
        e_attr.append(entropie(datay))
        ec_attr.append(entropie_cond([datay[(datax[:, i] == 1)],datay[(datax[:, i] != 1)]]))
            
        print("attribut :", fields[i])
        print("\tentropie =", e_attr[i])
        print("\tentropie conditonnelle =", ec_attr[i])
        print('\tdifference = ', ec_attr[i] - e_attr[i])
        
    e_attr = np.asarray(e_attr)
    ec_attr = np.asarray(ec_attr)
    diff = e_attr - ec_attr
    maxI = diff.argmax()
    print('\nLe meilleur attribut pour commencer : " %s " gain : %f' % (fields[maxI], diff[maxI])) 
    
    
    return np.asarray(e_attr), np.asarray(ec_attr)
        
        
        
#getEntropies(datax,datay,fields)


#Q4
    
def generateDT(datax,datay,fields):
    
    for depth in listD: 
        
        dt = DTree()
        dt.max_depth = depth 
        dt.min_samples_split = 2
        dt.fit(datax, datay)
        dt.predict(datax [:5 ,:])
        export_graphviz(dt, out_file ="data/tree"+str(depth)+".dot", feature_names = list(fields.values())[:32])
        
        g = gr.Source(open("data/tree"+str(depth)+".dot").read())
        png_bytes = g.pipe(format='png')
        with open("data/tree"+str(depth)+".png",'wb') as f:
            f.write(png_bytes)

#generateDT(datax,datay,fields)




#Q5

def getScorces(datax,datay,fields):
    out = []

    for depth in listD:    
        dt = DTree()
        dt.max_depth = depth 
        dt.min_samples_split = 2
        dt.fit(datax, datay)
        dt.predict(datax [:5 ,:])  
        out.append(dt.score(datax ,datay))
        
    plt.figure()
    plt.plot(listD, out, color='red')
    plt.xlabel("profondeur")
    plt.ylabel("score")
    plt.show()

#getScorces(datax,datay,fields)
    
    
#Q6
    
    
#Q7
    
def split_data(datax, datay, p):

    nb = datax.shape[0] 
    indices = np.arange(nb)
    np.random.shuffle(indices) 
    learn = indices[0:int(nb*p)] 
    test = indices[int(nb*p):] 
    return datax[learn], datax[test], datay[learn], datay[test]





def getError(datax, datay, p):


    datax_app, datax_test, datay_app, datay_test = split_data(datax, datay, p)
    errApp = []
    errTest = []
    
    for depth in listD:
        dt = DTree()
        dt.max_depth = depth 
        dt.min_samples_split = 2 
        dt.fit(datax_app, datay_app)
        errApp.append(np.linalg.norm(dt.predict(datax_app) - datay_app))
        errTest.append(np.linalg.norm(dt.predict(datax_test) - datay_test))
    
    plt.figure()
    plt.title("Error with learning rate="+str(p))
    plt.plot(listD, errApp, label="learning error", color='Blue')
    plt.plot(listD, errTest, label="test error", color='yellow')
    plt.xlabel("profondeur de l'arbre")
    plt.ylabel("erreur")
    plt.legend()
    plt.show()
    
    
    
#getError(datax, datay, 0.8)
    
    
    
#Q8
    
#Q9
    
def crossValidationError(datax, datay, nb):
    
    kf = KFold(n_splits=nb,shuffle=True)

    outApp = np.zeros(len(listD))
    outTest = np.zeros(len(listD))
    
    for train_index, test_index in kf.split(datax):
        
        datax_app, datay_app = datax[train_index], datay[train_index]
        datax_test, datay_test = datax[test_index], datay[test_index]
        
        errApp = []
        errTest = []
        
        for depth in listD:
            
            dt = DTree()
            dt.max_depth = depth
            dt.min_samples_split = 2 
            dt.fit(datax_app, datay_app)
            errApp.append(np.linalg.norm(dt.predict(datax_app) - datay_app))
            errTest.append(np.linalg.norm(dt.predict(datax_test) - datay_test))
            
        outApp += np.asarray(errApp)   
        outTest += np.asarray(errTest)   
    outApp /= nb
    outTest /= nb    
    plt.figure()
    plt.title("Error with Cross validation mean")
    plt.plot(listD, outApp, label="learning error", color='Blue')
    plt.plot(listD, outTest, label="test error", color='yellow')
    plt.xlabel("profondeur de l'arbre")
    plt.ylabel("erreur")
    plt.legend()
    plt.show()
    
    
    
    
    
    


######################## TME 2 ########################
    
    
#--------------------------------------------------------

POI_FILENAME = "data/poi-paris.pkl"
parismap = mpimg.imread("data/paris-48.806-2.23--48.916-2.48.jpg")

## coordonnees GPS de la carte
xmin, xmax = 2.23, 2.48  # coord_x min et max
ymin, ymax = 48.806, 48.916  # coord_y min et max
coords = [xmin, xmax, ymin, ymax]


# Valeurs potentielles de la constante de lissage.
smooth = [0.042,0.019,0.005,0.07]
# Quantités potentielles de cases dans les grilles.
step = [5,10,15,20]


#----------------------------------------------------------

class Density(object):

    def fit(self,data):
        pass

    def predict(self,data):
        pass

    def score(self,data):
        #A compléter : retourne la log-vraisemblance
        return np.log(self.predict(data)+10E-10).sum()


#------------------------------------------------------------

class Histogramme(Density):


    def __init__(self,steps=10):

        Density.__init__(self)
        self.steps = steps

        self.xmin = 2.23
        self.xmax = 2.48
        self.ymin = 48.806
        self.ymax = 48.916

        self.stepx = ((self.xmax-self.xmin)/self.steps)
        self.stepy = ((self.ymax-self.ymin)/self.steps)

    def fit(self,x):
        #A compléter : apprend l'histogramme de la densité sur x

        hist, edges = np.histogramdd(x, bins = (self.steps,self.steps))
        #hist /=hist.sum()
        self.hist = hist
        self.len = len(x)

    def predict(self,x):
        #A compléter : retourne la densité associée à chaque point de x
        preds = []

        for i in range(len(x)):
            preds.append(self.hist[int((x[i][0]-self.xmin)/self.stepx)][int((x[i][1]-self.ymin)/self.stepy)])



        return np.array(preds) /((self.stepx*self.stepy)*self.len)

#---------------------------------------------------------------------

def get_density2D(f,data,steps=100):
    """ Calcule la densité en chaque case d'une grille steps x steps dont les bornes sont calculées à partir du min/max de data. Renvoie la grille estimée et la discrétisation sur chaque axe.
    """
    xmin, xmax = data[:,0].min(), data[:,0].max()
    ymin, ymax = data[:,1].min(), data[:,1].max()
    xlin,ylin = np.linspace(xmin,xmax,steps),np.linspace(ymin,ymax,steps)
    xx, yy = np.meshgrid(xlin,ylin)
    grid = np.c_[xx.ravel(), yy.ravel()]
    res = f.predict(grid).reshape(steps, steps)
    return res, xlin, ylin

def show_density(f, data, steps=100, log=False):
    """ Dessine la densité f et ses courbes de niveau sur une grille 2D calculée à partir de data, avec un pas de discrétisation de steps. Le paramètre log permet d'afficher la log densité plutôt que la densité brute
    """
    res, xlin, ylin = get_density2D(f, data, steps)
    xx, yy = np.meshgrid(xlin, ylin)
    plt.figure()
    show_img()
    if log:
        res = np.log(res+1e-10)
    plt.scatter(data[:, 0], data[:, 1], alpha=0.8, s=3)
    show_img(res)
    plt.colorbar()
    plt.contour(xx, yy, res, 20)


def show_density_ax(ax, f, data, steps=100, log=False):
    """ Dessine la densité f et ses courbes de niveau sur une grille 2D calculée à partir de data, avec un pas de discrétisation de steps. Le paramètre log permet d'afficher la log densité plutôt que la densité brute
    """
    res, xlin, ylin = get_density2D(f, data, steps)
    xx, yy = np.meshgrid(xlin, ylin)
    show_img_ax(ax)
    if log:
        res = np.log(res+1e-10)
    ax.scatter(data[:, 0], data[:, 1], alpha=0.8, s=3)
    im = show_img_ax(ax,res)
    ax.contour(xx, yy, res, 20)


def show_img(img=parismap):
    """ Affiche une matrice ou une image selon les coordonnées de la carte de Paris.
    """
    origin = "lower" if len(img.shape) == 2 else "upper"
    alpha = 0.3 if len(img.shape) == 2 else 1.
    plt.imshow(img, extent=coords, aspect=1.5, origin=origin, alpha=alpha)
    ## extent pour controler l'echelle du plan

def show_img_ax(ax,img=parismap):
    """ Affiche une matrice ou une image selon les coordonnées de la carte de Paris.
    """
    origin = "lower" if len(img.shape) == 2 else "upper"
    alpha = 0.3 if len(img.shape) == 2 else 1.
    return ax.imshow(img, extent=coords, aspect=1.5, origin=origin, alpha=alpha)
    ## extent pour controler l'echelle du plan


def load_poi(typepoi,fn=POI_FILENAME):
    """ Dictionaire POI, clé : type de POI, valeur : dictionnaire des POIs de ce type : (id_POI, [coordonnées, note, nom, type, prix])

    Liste des POIs : furniture_store, laundry, bakery, cafe, home_goods_store,
    clothing_store, atm, lodging, night_club, convenience_store, restaurant, bar
    """
    poidata = pickle.load(open(fn, "rb"))
    data = np.array([[v[1][0][1],v[1][0][0]] for v in sorted(poidata[typepoi].items())])
    note = np.array([v[1][1] for v in sorted(poidata[typepoi].items())])
    return data,note


#------------------------------------------------------------------------

class KernelDensity(Density):

    def __init__(self,kernel=None,sigma=0.1):

        Density.__init__(self)
        self.kernel = kernel
        self.sigma = sigma

    def fit(self,x):
        self.x = x
        self.len = len(x)

    def kernelUniform(self,x):
        return np.array([1 if np.linalg.norm(xi) <= 0.5 else 0 for xi in x])

    def kernelGaussian(self,x):

        return np.array([( np.exp(-0.5*((np.linalg.norm(xi))**2)))/(2*np.pi)  for xi in x])

    def predict(self,data):

        preds = np.zeros(len(data))
        if self.kernel =='kernelUniform':

            for i in range(len(data)):
                preds[i] = (1 / (self.len * (self.sigma ** 2))) * (self.kernelUniform((data[i] - self.x) / self.sigma).sum())

        elif self.kernel == 'kernelGaussian':

            for i in range(len(data)):
                preds[i] = (1 / (self.len * self.sigma ** 2)) * (self.kernelGaussian((data[i] - self.x) / self.sigma).sum())

        return preds

#--------------------------------------------------------------------

class Nadaraya():


    def __init__(self,kernel,h):
        self.h = h
        self.thekernel = kernel
        self.coord = None
        self.notes = None

    def fit(self,data):
        self.coord = data[:,:2]
        self.notes = data[:,2]

    def kernelUniform(self,x,y):
        return np.where((np.abs(x-self.coord[:,0])<self.h/2)&(np.abs(y-self.coord[:,1])<self.h/2),1,0)

    def kernelGaussian(self,x,y):
        return np.exp(-0.5*((((self.coord[:,0]-x)/self.h)**2)+((self.coord[:,1]-y)/self.h)**2))/(np.sqrt(2*np.pi)*self.h)

    def predict(self,test):
        predictions = np.zeros(len(test))

        for i, dot in enumerate(test):

            if self.thekernel == 'uniforme':
                pond = self.kernelUniform(dot[0],dot[1])
            elif self.thekernel == 'gaussian':
                pond = self.kernelGaussian(dot[0],dot[1])
            predictions[i] = np.sum(self.notes*pond)/np.sum(pond)
        return predictions
    def score(self,data,test):
        return np.array((self.predict(data)-test)**2).mean()



def split(arr,cond):
    return [arr[cond],arr[~cond]]





######################## TME 3 ########################

NBITER = 100
epsValues = [ 0.1, 0.05, 0.01, 0.005, 0.001]

def mse(w,x,y):
    
    y = y.reshape(-1,1)
    w = w.reshape(-1,1)
    x = x.reshape(y.shape[0],w.shape[0])


    return ((x @ w - y) ** 2)



def mse_grad(w,x,y):
        
    y = y.reshape(-1,1)
    w = w.reshape(-1,1)
    x = x.reshape(y.shape[0],w.shape[0])

    return 2 * x * (x @ w - y)



def reglog(w,x,y):


    y = y.reshape(-1,1)
    w = w.reshape(-1,1)
    x = x.reshape(y.shape[0],w.shape[0])

    return np.log(1 + np.exp(-y * (x @ w))) 


def reglog_grad(w,x,y):
        
    y = y.reshape(-1,1)
    w = w.reshape(-1,1)
    x = x.reshape(y.shape[0],w.shape[0])
    
    return -x * y * (1/(1 + np.exp(y * (x @ w))))

    
    
def grad_check(f,f_grad,d=1,N=100,eps=1e-5):
    
    x = np.random.rand(N,1)
    y = np.random.randint(0,2,N)
    ws = np.random.rand(N,1)
    
    for w in ws:
        v1 = f(w,x,y)
        v2 = f(w+eps,x,y)
        grad = f_grad(w,x,y)
        r1,r2 = (v2-v1)/eps , grad
        if (np.max(np.abs(r1 - r2)) > eps):
            return False
    return True




def descente_gradient(datax,datay,f_loss,f_grad,eps,iter):
    

    w = np.random.rand(datax.shape[1])
    w = w.reshape(-1,1)
    w_history = np.zeros((iter,w.shape[0]))
    loss_history = []
        

    for i in range(iter):

        w = w - eps * np.mean(f_grad(w,datax,datay),0).reshape(-1, 1)
        w_history[i,:] = w.T
        loss_history.append(f_loss(w,datax,datay).mean())

    return w, w_history, np.asarray(loss_history)


def plot_wHist(datax, datay, f, w_history):
    grid, x, y = make_grid(xmin=-2, xmax=2, ymin=-2, ymax=2, step=10)
    plt.figure(figsize=(15,10))
    plt.contourf(x,y,np.array([f(w,datax,datay).mean() for w in grid]).reshape(x.shape),levels=50)
    plt.colorbar()
    
    wlist_show = [w_history[0]]
    for w in w_history[1:-1]:
        if np.linalg.norm(wlist_show[-1][:2] - w[:2]) > 0.1:
            wlist_show.append(w)
    wlist_show.append(w_history[-1])
    plt.plot(np.array(wlist_show)[:, 0], np.array(wlist_show)[:, 1], 'k-')
    plt.plot(np.array(wlist_show)[0, 0], np.array(wlist_show)[0, 1], 'bo', label='start')
    plt.plot(np.array(wlist_show)[1:-1, 0], np.array(wlist_show)[1:-1, 1], 'ko', label='between')
    plt.plot(np.array(wlist_show)[-1, 0], np.array(wlist_show)[-1, 1], 'ro', label='end')
    plt.legend()
    plt.title('Visualition de l\'historique des w')
    plt.show()
        
        
    
        
def get_frontiere(datax,datay,f=None,eps=0.05):
    
    

    if f =="mse"  :  
        
        w, w_history, loss_history = descente_gradient(datax,datay,mse,mse_grad,eps,NBITER)
        
        plt.figure(figsize=(15, 5))
        plt.subplot(1,2,1)
        plt.title('MSE : Frontière de décision avec w optimale')
        plot_frontiere(datax,lambda x : np.sign(x.dot(w)),step=100)
        plot_data(datax,datay.reshape(-1))

        
        plt.subplot(1,2,2)
        plt.title('MSE : Coût en fonction des itérations')
        plt.xlabel('Iterations')
        plt.ylabel('Coûts')
        plt.plot(np.arange(NBITER),loss_history)
        plt.show()
    
            
    elif f =="reglog"  :   
        
        w, w_history, loss_history = descente_gradient(datax,datay,reglog,reglog_grad,eps,NBITER)
        
        plt.figure(figsize=(15, 5))
        plt.subplot(1,2,1)
        plt.title('REGLOG : Frontière de décision avec w optimale')
        plot_frontiere(datax,lambda x : np.sign(x.dot(w)),step=100)
        plot_data(datax,datay.reshape(-1))

        
        plt.subplot(1,2,2)
        plt.title('REGLOG : Coût en fonction des itérations')
        plt.xlabel('Iterations')
        plt.ylabel('Coûts')
        plt.plot(np.arange(NBITER),loss_history)
        plt.show()
    
        
    else:
        plt.figure()
        plt.title('Frontière de décision avec w random')
        w  = np.random.randn(datax.shape[1],1)
        plot_frontiere(datax,lambda x : np.sign(x.dot(w)),step=100)
        plot_data(datax,datay.reshape(-1))
        plt.show()
        


    
        

        
def get_cost_by_eps(datax,datay):
    
    out1 = []    
    out2 = []


    for e in epsValues:
        w, w_history, loss_history = descente_gradient(datax,datay,mse,mse_grad,e,NBITER)
        out1.append(mse(w,datax,datay).mean())


        w, w_history, loss_history = descente_gradient(datax,datay,reglog,reglog_grad,e,NBITER)
        out2.append(reglog(w,datax,datay).mean())

    
        

    
    plt.figure(figsize=(15, 5))
    plt.xlabel('w')
    plt.ylabel('erreur')
    
    
    plt.subplot(1,2,1)   
    plt.title('MSE en fonction de w')
    plt.xticks(np.arange(len(epsValues)),labels=list(map(str,epsValues)))
    plt.plot(np.arange(len(epsValues)),out1) 
    
    
    plt.subplot(1,2,2) 
    plt.title('REGLOG en fonction de w')
    plt.xticks(np.arange(len(epsValues)),labels=list(map(str,epsValues)))
    plt.plot(np.arange(len(epsValues)),out2)
    
    plt.show()
    
    
    
######################## TME 4 ########################



def perceptron_loss(w,x,y):
    x, y = x.reshape(len(y), -1), y.reshape(-1, 1)
    w = w.reshape(-1, 1)
    return np.maximum(0, -y * x.dot(w))

def perceptron_gradient(w,x,y):


    x, y = x.reshape(len(y), -1), y.reshape(-1, 1)
    w = w.reshape(-1, 1)
    return -(np.where(perceptron_loss(w,x,y)>0,1,0))*y*x


class Lineaire(object):
    
    def __init__(self,loss=perceptron_loss,loss_g=perceptron_gradient,max_iter=1000,eps=0.01):
        self.max_iter, self.eps = max_iter,eps
        self.w = None
        self.loss,self.loss_g = loss,loss_g

    def fit(self,trainx,trainy,testx=None,testy=None):
        self.w = np.ones((trainx.shape[1],1))
        self.allw=[]
        self.allwT=[]
        for i in range(self.max_iter):
            self.w = self.w - self.eps * np.mean(self.loss_g(self.w,trainx,trainy),0).reshape(-1, 1)
            self.allw.append(self.loss(self.w,trainx,trainy).mean())
            if(not testx is None):
                self.allwT.append(self.loss(self.w,testx,testy).mean())
            if(self.loss(self.w,trainx,trainy).mean()<self.eps):
                break
        return self.w


    def predict(self,datax):
        return np.sign(np.dot(datax,self.w)).reshape(-1)

    def score(self,datax,datay):
        return np.mean(self.predict(datax) == datay)
    
def load_usps(fn):
    with open(fn,"r") as f:
        f.readline()
        data = [[float(x) for x in l.split()] for l in f if len(l.split())>2]
    tmp=np.array(data)
    return tmp[:,1:],tmp[:,0].astype(int)

def get_usps(l,datax,datay):
    if type(l)!=list:
        resx = datax[datay==l,:]
        resy = datay[datay==l]
        return resx,resy
    tmp =   list(zip(*[get_usps(i,datax,datay) for i in l]))
    tmpx,tmpy = np.vstack(tmp[0]),np.hstack(tmp[1])
    return tmpx,tmpy

def show_usps(data):
    plt.imshow(data.reshape((16,16)),interpolation="nearest",cmap="gray")    
    
######################## TME 5 ########################



def perceptron_loss_reg(w,x,y,alpha=1):
    return perceptron_loss(w,x,y) + alpha*np.sum(w**2) 

def perceptron_gradient_reg(w,x,y,alpha=1):
    w = w.reshape(-1)
    return perceptron_gradient(w,x,y) + 2*alpha*w 
    

def get_mnist(maxtrain=2000,maxtest=800):
    
    (trainx, trainy), (testx, testy) = mnist.load_data()
    trainx = trainx.reshape(trainx.shape[0], -1)
    testx = testx.reshape(testx.shape[0], -1)
    trainx = trainx / np.max(trainx)
    testx = testx / np.max(testx)
    
    
    trainx = trainx[:maxtrain]
    trainy = trainy[:maxtrain]
    testx = testx[:maxtest]
    testy = testy[:maxtest]
    
    return trainx, trainy, testx, testy

    
def plot_perceptron_MNIST(maxtrain=2000,maxtest=800):
    
    trainx , trainy, testx, testy = get_mnist(maxtrain, maxtest)    
    
    # 9 vs all 
    
    trainy = np.where(trainy == 9, 1, -1)
    testy = np.where(testy == 9, 1, -1)
    
    perceptron = Lineaire(perceptron_loss, perceptron_gradient)
    perceptron.fit(trainx,trainy,testx,testy)
    
        
    perceptronReg = Lineaire(perceptron_loss_reg, perceptron_gradient_reg)
    perceptronReg.fit(trainx,trainy,testx,testy)
    
    svc = svm.SVC(max_iter=1000, kernel='linear')
    svc.fit(trainx,trainy)
    print("Score 9 vs all perceptron simple : train {0}, test {1}".format(
            perceptron.score(trainx, trainy), perceptron.score(testx, testy)))
    
    print("Score 9 vs all perceptron regularisé : train {0}, test {1}".format(
            perceptronReg.score(trainx, trainy), perceptronReg.score(testx, testy)))
    
    print("Score SVM 9 vs all : train {0}, test {1}".format(svc.score(trainx,trainy),svc.score(testx,testy)))
    
    
    plt.figure(figsize =(15,5))
    plt.subplot(1,2,1)
    plt.title('Perceptron simple 9 vs all')
    
    plt.plot(np.arange(len(perceptron.allw)), np.array(perceptron.allw), 'blue', label='Learn')
    plt.plot(np.arange(len(perceptron.allwT)), np.array(perceptron.allwT), 'yellow', label='Test')
    
    plt.xlabel('History')
    plt.ylabel('Loss')
    plt.legend()
    
    

    plt.subplot(1,2,2)
    plt.title('Perceptron Regularisé 9 vs all')
    plt.plot(np.arange(len(perceptronReg.allw)), np.array(perceptronReg.allw), 'blue', label='Learn')
    plt.plot(np.arange(len(perceptronReg.allwT)), np.array(perceptronReg.allwT), 'yellow', label='Test')
    
    plt.xlabel('History')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.plot()
    
    
    
    
def plot_frontiere_proba(data,f,step=20):
    grid,x,y=make_grid(data=data,step=step)
    plt.contourf(x,y,f(grid).reshape(x.shape), 255)
    plt.colorbar()


    
def plot_sklearn_models(x_train, y_train, x_test, y_test):

    perceptron_sklearn = linear_model.Perceptron(max_iter=1000, tol=None)
    perceptron_sklearn.fit(x_train, y_train.ravel())
    
    knn_sklearn = neighbors.KNeighborsClassifier(n_neighbors=3)
    knn_sklearn.fit(x_train, y_train.ravel())
    
    tree_sklearn = tree.DecisionTreeClassifier()
    tree_sklearn.fit(x_train, y_train.ravel())
    
    print("Score Perceptron : train {0}, test {1}".format(
        perceptron_sklearn.score(x_train, y_train), perceptron_sklearn.score(x_test, y_test)
    ))    
    print("Score KNN : train {0}, test {1}".format(
        knn_sklearn.score(x_train, y_train), knn_sklearn.score(x_test, y_test)
    ))    
    print("Score Tree : train {0}, test {1}".format(
        tree_sklearn.score(x_train, y_train), tree_sklearn.score(x_test, y_test)
    ))
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1,3,1)
    plt.title('Perceptron : Frontière de décision')
    plot_frontiere(x_train, perceptron_sklearn.predict, 200)
    plot_data(x_train, y_train.ravel())
    
    plt.subplot(1,3,2)
    plt.title('KNN : Frontière de décision')
    plot_frontiere(x_train, knn_sklearn.predict, 200)
    plot_data(x_train, y_train.ravel())
    
    plt.subplot(1,3,3)
    plt.title('Tree : Frontière de décision')
    plot_frontiere(x_train, tree_sklearn.predict, 200)
    plot_data(x_train, y_train.ravel())
    
    plt.show()
    
    
    
def plot_svm_alea(params,trainx,trainy,testx,testy):


    svc = svm.SVC(**params)
    svc.fit(trainx, trainy.ravel())
    
    print("Score SVM : train {0}, test {1}".format( svc.score(trainx, trainy), svc.score(testx, testy)))
    
    plt.figure(figsize=(15, 6))
    plt.subplot(1,2,1)
    plt.title('Frontières')
    plot_frontiere_proba(trainx, lambda x: svc.predict_proba(x)[:, 0], 200)
    plot_data(trainx, trainy.reshape(-1))
    
    # plt.figure()
    plt.subplot(1,2,2)
    plt.grid(True, alpha=0.5)
    plot_data(trainx[svc.support_], trainy[svc.support_].reshape(-1))
    plt.title('Vecteurs supports')
    
    plt.show()


def grid_search(x_train, x_test, y_train, y_test,kernel='rbf'):

    param = {'kernel': [kernel],
        'C': [0.5, 1.0, 2.0, 5.0, 10.0],
        'max_iter': [1000,2000,4000],
        'shrinking':[True,False],
        'probability': [True, False]
       }
    
    if kernel in ['rbf', 'poly', 'sigmoid']:
        param['gamma'] = ['auto', 'scale', 0.001, 0.01, 0.1, 1.0]
        if kernel in ['poly', 'sigmoid']:
            param['coef0'] = [-1.0, 0.0, 1.0]
            if kernel == 'poly':
                param['degree'] = [1,3,5,7]
                

    clf = GridSearchCV(svm.SVC(), param, cv=5)
    clf.fit(x_train, y_train.ravel())

    return clf.best_params_



def pred_mnist_one_vs_all(maxtrain=2000,maxtest=800,kernel = 'rbf',max_iter = 1000,nb_pred = 6):

    trainx , trainy, testx, testy = get_mnist(maxtrain, maxtest)
    

    
    svcOneVsAll = OneVsRestClassifier(svm.SVC(max_iter=max_iter, kernel=kernel))
    svcOneVsAll.fit(trainx, trainy)
    print("Score (one-vs-rest) : train {0}, test {1}".format(svcOneVsAll.score(trainx, trainy),svcOneVsAll.score(testx, testy)))
    
    
    random_ind = np.random.choice(np.arange(testx.shape[0]), nb_pred, replace=False)
    plt.figure(figsize=(15,5*np.ceil(nb_pred / 3)))
    j = 1
    for i in random_ind:
        plt.subplot(np.ceil(nb_pred / 3),3,j)
        plt.title("One-vs-rest. pred : {0} true : {1}".format(svcOneVsAll.predict(testx[i].reshape(1, -1)), testy[i]))
        plt.imshow(testx[i].reshape(28, 28), interpolation='nearest', cmap='gray')
        j+=1



def pred_mnist_one_vs_one(maxtrain=2000,maxtest=800,kernel = 'rbf',max_iter = 1000,nb_pred = 6):

    trainx , trainy, testx, testy = get_mnist(maxtrain, maxtest)
    

    
    svcOneVsOne = OneVsOneClassifier(svm.SVC(max_iter=max_iter, kernel=kernel))
    svcOneVsOne.fit(trainx, trainy)
    print("Score (one-vs-one) : train {0}, test {1}".format(svcOneVsOne.score(trainx, trainy),svcOneVsOne.score(testx, testy)))
    
    
    random_ind = np.random.choice(np.arange(testx.shape[0]), nb_pred, replace=False)
    plt.figure(figsize=(15,5*np.ceil(nb_pred / 3)))
    j = 1
    for i in random_ind:
        plt.subplot(np.ceil(nb_pred / 3),3,j)
        plt.title("One-vs-rest. pred : {0} true : {1}".format(svcOneVsOne.predict(testx[i].reshape(1, -1)), testy[i]))
        plt.imshow(testx[i].reshape(28, 28), interpolation='nearest', cmap='gray')
        j+=1


def sousSequence(x, u):
    out = []
    
    list_x = list(x)
    list_u = list(u)
    lx = len(list_x)
    lu = len(list_u)
    

    for i in range(lx-lu):
        seq = list_x[i:i+lu]

        if seq == list_u:
            out.append(list(range(i,i+lu)))  
    return out


def get_words(data,n):
    
    todelete = " ,.!?/&-:;@'..."

    out = set()
    for text in data:
        tList = list(''.join(ch for ch in re.split("["+"\\".join(todelete)+"]", text) if ch))
        for i in range(len(tList)-n):
            out.add(''.join(tList[i:i+n]))
    return list(out)


def stringKernel(s,t,n,lamb = 0.5):

    words = get_words([s],n)
    out = 0
    
    for u in words:
        alls = sousSequence(s, u)
        allt = sousSequence(t, u)

        for tt in allt:
            for ss in alls:
                out += + lamb ** (tt[len(tt)-1]-tt[0]+1)+(ss[len(ss)-1]-ss[0]+1)
    return out 


def getSimMat(data,n,lamb=0.5):
    
    N = len(data)

    out = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            out[i,j] = stringKernel(data[i],data[j],n,lamb)
    return out

def plot_svm_stringKernal():
    # from https://www.kaggle.com/coolcoder22/quotes-dataset
    data = pd.read_csv("data/QUOTE.csv")
    
    data_2classes = data[data["Author"].isin(["David Brinkley","Marc Jacobs"])]
    data_2classes["Author"].replace({"David Brinkley":1,"Marc Jacobs":-1}, inplace=True)
    
    datax = data_2classes["quote"].to_numpy()
    datay = data_2classes["Author"].to_numpy()
    
    
    mat = getSimMat(datax,4,0.5)
    
    plt.title("matrice de similarité entre deux Auteur David Brinkley et Marc Jacobs")
    plt.imshow(mat-np.diag(np.diag(mat)))
    plt.colorbar()
    
    ind = list(range(datax.shape[0]))
    np.random.shuffle(ind)
    
    trainInd = ind[len(ind)//4:]
    testInd = ind[:len(ind)//4]
    
    
    trainx = mat[trainInd]
    trainy = datay[trainInd]
    testx = mat[testInd]
    testy = datay[testInd]
    
    s = svm.SVC(C=100,max_iter=5000)
    s.fit(trainx, trainy)
    
    print("Score SVM StringKernel : train {0}, test {1}".format(s.score(trainx,trainy),s.score(testx,testy)))
    

    

