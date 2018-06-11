import numpy as np
import matplotlib.pyplot as plt

class Model():
    def __init__(self):
        self.x_train = [1, 2, 3, 4, 5]
        self.y_train = [2, 4, 6, 8, 10]
    def forward(self, x, w):
        return x*w
    def loss(self, y, y_pred):
        return (y_pred - y) * (y_pred -y)

model = Model()

plt.subplot(2, 1, 1)  
plt.title('Output Lines')
plt.plot(model.x_train, model.y_train, 'r-' , label='True Line')
plt.ylabel('y')
plt.xlabel('x')

losses = [] 
for w in np.arange(0, 5, 1):
    y_predictions = [] 
    loss = 0
    print(" *********************** Weight= ", w, " ***************************")
    for x, y in zip(model.x_train , model.y_train):
        y_pred = model.forward(x=x, w=w)
        y_predictions.extend([y_pred])
        print("x_value = ", x)
        print("y_value = ", y)
        print("y_predic = ", y_pred)
        loss += model.loss(y=y, y_pred = y_pred)
        print("loss = ", loss)  
    plt.plot(model.x_train, y_predictions)
    losses.extend([loss])

plt.subplot(2, 1, 2)
plt.title('Weight to loss ratio')
plt.plot(np.arange(0, 5, 1), losses)   
plt.show()
