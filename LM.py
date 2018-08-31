from scipy.special import expit

import time
import numpy as np
import os

eta =0.01 # for singular Hessian
np.random.seed(1)

def step(fun , x , deriv =False):
	if (fun == 'nonlin'):
		if (deriv == True):
			return x * (1 - x)
		return 1 / (1 + np.exp(-x))


	if (fun == 'tanh'):
		if (deriv ==True):
			return 1.0 - np.tanh(x)**2
		return np.tanh(x)


# NUMBER OF  MINIMUMM LAYERS 2 AS ONE FOR INPUT AND ONE FOR OUTPUT
# pass a list  number of neurons in layers
def create_Net(input_data,output_data,nL):
    num_hiddenLayers =len(nL)
    net = [{'name': 'input data','weights':input_data}]
    in_w = {"name":"input Layer"}
    in_w['weights'] = create_Weights(len(input_data[0]),nL[0])
    net.append(in_w)
    # Create hidden Layers
    for i in range(num_hiddenLayers-1):
        hidden = {"name":"hidden "+str(i)}
        hidden['weights'] = create_Weights(nL[i],nL[i+1])
        net.append(hidden)
    out = {"name": "output Layer"}
    out["weights"] = create_Weights(nL[-1], len(output_data[0]))
    net.append(out)
    net.append({"name":"output data", "weights":output_data})
    return net

# for initializing weights between 2 layers
def create_Weights(layer0,layer1):
    return 2*np.random.random((layer0,layer1)) - 1

def forward(weights,activation):
    layers = [weights[0]["weights"]]
    
    for index, w in enumerate(weights[:-1]):
        if index == 0: continue
        layers.append(step('nonlin',np.dot(layers[index - 1],w["weights"])))
    return layers

def back(layers, weights,Learning_rate,activation):
    # loss function is 1/2 (y-t)**2 Squred ERROR
    errors = [weights[-1]["weights"] - layers[-1]]
    weight_index = -1
    layer_index = -1
    error_index = 0
    delta_index = 0 # 
    delta= []

    while len(layers) - abs(layer_index) > 0:
        delta.append(errors[error_index] *step(activation,layers[layer_index], deriv=True))
        weight_index -= 1
        layer_index -= 1

        errors.append(delta[delta_index].dot(weights[weight_index]["weights"].T))
        error_index += 1
        delta_index += 1


    weight_index = -2
    layer_index = -2
    delta_index = 0

    while len(layers) - abs(layer_index) >= 0:
        
        weights[weight_index]["weights"] += Learning_rate * layers[layer_index].T.dot(delta[delta_index])
        weight_index -= 1
        layer_index -= 1
        delta_index += 1

    return weights, errors[0]

def lmBack(layers,weights):
    #loss function Squred Error
    errors = [weights[-1]["weights"]- layers[-1]]
    #indeies
    weight_index = -1
    layer_index = -1
    error_index = 0
    delta_index = 0
    jacobian_index=0
    Jacobian =[]
    delta= [] 
    Hessian = []
    

    while len(layers) - abs(layer_index) > 0:
        delta.append(errors[error_index] *step('nonlin',layers[layer_index], deriv=True))
        Jacobian.append(step('nonlin',layers[jacobian_index], deriv=True))
        Hessian.append(Jacobian[delta_index].T.dot(Jacobian[delta_index]))
        Hessian[delta_index]=Hessian[delta_index] + (eta *np.eye(len(Hessian[delta_index]))) 

        weight_index -= 1
        layer_index -= 1
        errors.append(delta[delta_index].dot(weights[weight_index]["weights"].T))
        error_index += 1
        delta_index += 1
        jacobian_index +=1

    jacobian_index=-1
    weight_index = -2
    layer_index = -2
    delta_index = 0

    
    for x in Hessian:
        U =(np.linalg.inv(Hessian[jacobian_index]))  
        
        wk=U.dot(layers[layer_index].T.dot(delta[delta_index]))
        weights[weight_index]["weights"] +=wk	
        weight_index -= 1
        layer_index -= 1
        delta_index += 1
        jacobian_index -=1

    return weights, errors[0]

def STD(num_hidden_neurons, input_data, output_data, epochs,Learning_rate=0.5,activation='nonlin'):
    np.random.seed(1)
    print("Training Algo :Standerd Backprop")
    errors =[]
    net = create_Net(input_data, output_data, num_hidden_neurons)

    for j in range(epochs):
        layers = forward(net,activation)
        net, error = back(layers, net,Learning_rate,activation)

        if j % 2 == 0:
            errors.append(np.mean(np.abs(error)))
    	
    return errors

def LM(num_hidden_neurons, input_data, output_data,epochs=10000,activation='nonlin'):
    np.random.seed(1)
    print("Training Algo :Levenberg Marqurt")
    errors =[]
    net = create_Net(input_data, output_data, num_hidden_neurons)
    for x in range(epochs):
    	layers = forward(net,'nonlin')
    	net, error = lmBack(layers,net)

    	if x % 2 == 0:
    		errors.append(np.mean(np.abs(error)))
    return errors

def Train ( train,num_hidden_neurons, input_data,output_data,epochs=2000 ,activation='nonlin'):
	if(train == 'STD'):
		return STD(num_hidden_neurons, input_data,output_data,epochs,activation)

	if(train == 'LM'):
		return LM(num_hidden_neurons, input_data,output_data,epochs,activation)


if  __name__  ==  '__main__':
    start_time= time.time()
    
    X= np.random.random((2,500))
    X= eta * X 
    y= np.random.random((1,500)) 
    y= eta * y 


    num_hidden_neurons =[3,2]
    epochs = 5000
    train='STD'
    #errors=train(train,num_hidden_neurons, X, y,epochs,activation='nonlin')
    
    errors =LM(num_hidden_neurons, X, y,epochs,activation='nonlin')
    print('\n\nStats:')
    print('#hidden layers is :'+str(num_hidden_neurons))
    print('MAXIMUM error is: '+str(max(errors)))
    print('MINIMUM error is: '+str(min(errors)))
    print('AVARAGE error is: '+str(sum(errors)/float(len(errors))))


    print("\n\n------excution time : %s seconds ------" % (time.time() - start_time))
    
