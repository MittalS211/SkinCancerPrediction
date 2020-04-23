import numpy as np 

def sigmoid(z):
    return 1/(1+np.exp(-z))

def init_weights(dims):
    w = np.zeros((dims,1))
    b = 0
    
    return w,b

def forward(w,b,x,y):
   
    
    m = x.shape[1]
    print(m)
    assert(m>0)
    A = sigmoid(np.dot(w.T,x) + b)
    cost = -1/m*np.sum(np.dot(y,np.log(A).T) + np.dot((1-y),np.log(1-A).T)) 
    
    dw = 1/m*np.dot(x,(A-y).T)
    db = 1/m*np.sum(A-y)
    
    cost = np.squeeze(cost)
    
    grads = {
        "dw" : dw,
        "db" : db
    }
    
    return grads,cost

def get_weights(w,b,x,y,itr,alpha):
    
    costs = []
    for i in range(itr):
        
        grads,cost = forward(w,b,x,y)
        
        dw = grads["dw"]
        db = grads["db"]
        
        w = w - alpha*dw
        b = b - alpha*db
        
        if(i%100 == 0):
            costs.append(cost)
            
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads,costs

def predict(w,b,x):
    a = sigmoid(np.dot(w.T,x)+b)
    return (a>0.50).astype('float')

def Model(train_x,train_y,test_x,test_y,itr,alpha):
    
    w,b = init_weights(train_x.shape[0])
    params,grads,costs = get_weights(w,b,train_x,train_y,itr,alpha)
    
    w = params["w"]
    b = params["b"]
    
    train_predict = predict(w,b,train_x)
    test_predict = predict(w,b,test_x)
    
    print("train accuracy: {} %".format(100 - np.mean(np.abs(train_predict - train_y)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(test_predict - test_y)) * 100))
    
    d = {"costs": costs,
         "test_predict": test_predict, 
         "train_predict" : train_predict, 
         "w" : w, 
         "b" : b,
         "alpha" : alpha,
         "itr": itr}
    
    return d