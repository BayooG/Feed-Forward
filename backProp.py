import numpy as np # for matrices multiplication 
import time 

#activation function
def step(fun , x , deriv =False):
	if (fun == 'nonlin'):
		if (deriv == True):
			return x * (1 - x)
		return 1 / (1 + np.exp(-x))

'''
	for initializing weights between 2 layers
	layer0 : number of neurons in the previous layer
	layer1 : number of neurons in the next layer 
'''
def create_Weights(layer0,layer1):
    return 2*np.random.random((layer0,layer1)) - 1
'''
	input  data : training examples 
	output data : target output
	nL : list of hidden layeres number of nurons

	net is list of dictionaries  each one represent a layer exept for the first and the last one 
	first dictionary  represents training example  
	last dictionary represents tragented output 
'''
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

'''
	Layers list represents the activation function of neurons 

'''
def forward(weights,activation):
    layers = [weights[0]["weights"]]
    
    for index, w in enumerate(weights[:-1]):
        if index == 0: continue
        layers.append(step('nonlin',np.dot(layers[index - 1],w["weights"])))
    return layers


# this is where magic happens 
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

#putting it togther
def STD(num_hidden_neurons, input_data, output_data, epochs,Learning_rate=0.5,activation='nonlin'):
    np.random.seed(1)
    errors =[]
    net = create_Net(input_data, output_data, num_hidden_neurons)

    for j in range(epochs):
        layers = forward(net,activation)
        net, error = back(layers, net,Learning_rate,activation)

        if j % 2 == 0:
            errors.append(np.mean(np.abs(error)))
    	
    return errors
if __name__ == '__main__':
    start_time= time.time()
    eta = 0.001 
    X= np.random.random((2,500))
    X= eta * X 
    y= np.random.random((1,500)) 
    y= eta * y 

    num_hidden_neurons =[3,2,8]
    epochs = 500
    
    errors=STD(num_hidden_neurons, X, y,epochs,activation='nonlin')
    
    print('\n\nStats:')
    print('#hidden layers is :'+str(num_hidden_neurons))
    print('MAXIMUM error is: '+str(max(errors)))
    print('MINIMUM error is: '+str(min(errors)))
    print('AVARAGE error is: '+str(sum(errors)/float(len(errors))))


    print("\n\n------excution time : %s seconds ------" % (time.time() - start_time))
