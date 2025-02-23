import numpy as np
import matplotlib.pyplot as plt

class LogisticNeuron():

    def __init__(self, n_inputs):
        self.w = -1 * 2*np.random.rand(n_inputs)
        self.b = -1 * 2*np.random.rand()

    def predict_proba(self, X):
        Z = np.dot(self.w, X) + self.b
        Yest = 1 / (1 + np.exp(-Z))
        return Yest
    
    def predict(self, X, umbral=0.5):
        Z = np.dot(self.w, X) + self.b
        Yest = 1 / (1 + np.exp(-Z))
        return 1 * (Yest >= umbral)
    
    def fit(self, X, Y, epochs=500, lr=0.01):
        p = X.shape[1]
        for _ in range(epochs):
            Yest = self.predict_proba(X)
            self.w += (lr/p) * np.dot((Y-Yest), X.T).ravel()
            self.b += (lr/p) * np.sum(Y-Yest)
            

#%% Ejemplo -----------------------------------------------
X = np.array([[0,0,1,1],
              [0,1,0,1]])
Y = np.array([[0,0,0,1]])


neuron = LogisticNeuron(2)
neuron.fit(X, Y, lr=1)
print(neuron.predict_proba(X))
print(neuron.predict(X))

#%% Dibujo ------------------------
def draw_2d_percep(model):
    w1, w2, b = model.w[0], model.w[1], model.b
    plt.plot([-2, 2], [(1/w2)*(-w1*(-2)-b), (1/w2)*(-w1*2-b)], '--k')

_, p = X.shape
for i in range(p):
    if Y[0,i] == 0:
        plt.plot(X[0,i], X[1,i], 'or')
    else:
        plt.plot(X[0,i], X[1,i], 'ob')

plt.title('Neurona logistica')
plt.grid('on')
plt.xlim([-2,2])
plt.ylim([-2,2])
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
draw_2d_percep(neuron)
plt.show()