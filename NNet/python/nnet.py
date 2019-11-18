from math import inf

import numpy as np
import copy
class NNet():
    """
    Class that represents a fully connected ReLU network from a .nnet file
    
    Args:
        filename (str): A .nnet file to load
    
    Attributes:
        numLayers (int): Number of weight matrices or bias vectors in neural network
        layerSizes (list of ints): Size of input layer, hidden layers, and output layer
        inputSize (int): Size of input
        outputSize (int): Size of output
        mins (list of floats): Minimum values of inputs
        maxes (list of floats): Maximum values of inputs
        means (list of floats): Means of inputs and mean of outputs
        ranges (list of floats): Ranges of inputs and range of outputs
        weights (list of numpy arrays): Weight matrices in network
        biases (list of numpy arrays): Bias vectors in network
    """
    def __init__ (self, filename):
        with open(filename) as f:
            line = f.readline()
            cnt = 1
            while line[0:2] == "//":
                line=f.readline() 
                cnt+= 1
            #numLayers does't include the input layer!
            numLayers, inputSize, outputSize, _ = [int(x) for x in line.strip().split(",")[:-1]]
            line=f.readline()

            #input layer size, layer1size, layer2size...
            layerSizes = [int(x) for x in line.strip().split(",")[:-1]]

            line=f.readline()
            symmetric = int(line.strip().split(",")[0])

            line = f.readline()
            inputMinimums = [float(x) for x in line.strip().split(",")[:-1]]

            line = f.readline()
            inputMaximums = [float(x) for x in line.strip().split(",")[:-1]]

            line = f.readline()
            inputMeans = [float(x) for x in line.strip().split(",")[:-1]]

            line = f.readline()
            inputRanges = [float(x) for x in line.strip().split(",")[:-1]]

            weights=[]
            biases = []
            for layernum in range(numLayers):

                previousLayerSize = layerSizes[layernum]
                currentLayerSize = layerSizes[layernum+1]
                weights.append([])
                biases.append([])
                weights[layernum] = np.zeros((currentLayerSize,previousLayerSize))
                for i in range(currentLayerSize):
                    line=f.readline()
                    aux = [float(x) for x in line.strip().split(",")[:-1]]
                    for j in range(previousLayerSize):
                        weights[layernum][i,j] = aux[j]
                #biases
                biases[layernum] = np.zeros(currentLayerSize)
                for i in range(currentLayerSize):
                    line=f.readline()
                    x = float(line.strip().split(",")[0])
                    biases[layernum][i] = x

            self.numLayers = numLayers
            self.layerSizes = layerSizes
            self.inputSize = inputSize
            self.outputSize = outputSize
            self.mins = inputMinimums
            self.maxes = inputMaximums
            self.means = inputMeans
            self.ranges = inputRanges
            self.weights = weights
            self.biases = biases
            
    def evaluate_network(self, inputs):
        '''
        Evaluate network using given inputs
        
        Args:
            inputs (numpy array of floats): Network inputs to be evaluated
            
        Returns:
            (numpy array of floats): Network output
        '''
        numLayers = self.numLayers
        inputSize = self.inputSize
        outputSize = self.outputSize
        biases = self.biases
        weights = self.weights

        # Prepare the inputs to the neural network
        inputsNorm = np.zeros(inputSize)

        for i in range(inputSize):
            if inputs[i]<self.mins[i]:
                inputsNorm[i] = (self.mins[i]-self.means[i])/self.ranges[i]
            elif inputs[i]>self.maxes[i]:
                inputsNorm[i] = (self.maxes[i]-self.means[i])/self.ranges[i] 
            else:
                inputsNorm[i] = (inputs[i]-self.means[i])/self.ranges[i] 

        # Evaluate the neural network
        for layer in range(numLayers-1):
            inputsNorm = np.maximum(np.dot(weights[layer],inputsNorm)+biases[layer],0)
        outputs = np.dot(weights[-1],inputsNorm)+biases[-1]

        # Undo output normalization
        for i in range(outputSize):
            outputs[i] = outputs[i]*self.ranges[-1]+self.means[-1]
        return outputs

    def evaluate_network_multiple(self,inputs):
        '''
        Evaluate network using multiple sets of inputs
        
        Args:
            inputs (numpy array of floats): Array of network inputs to be evaluated.
            
        Returns:
            (numpy array of floats): Network outputs for each set of inputs
        '''
        
        numLayers = self.numLayers
        inputSize = self.inputSize
        outputSize = self.outputSize
        biases = self.biases
        weights = self.weights
        inputs = np.array(inputs).T

        # Prepare the inputs to the neural network
        numInputs = inputs.shape[1]
        inputsNorm = np.zeros((inputSize,numInputs))
        for i in range(inputSize):
            for j in range(numInputs):
                if inputs[i,j]<self.mins[i]:
                    inputsNorm[i,j] = (self.mins[i]-self.means[i])/self.ranges[i]
                elif inputs[i,j] > self.maxes[i]:
                    inputsNorm[i,j] = (self.maxes[i]-self.means[i])/self.ranges[i] 
                else:
                    inputsNorm[i,j] = (inputs[i,j]-self.means[i])/self.ranges[i]

        # Evaluate the neural network
        for layer in range(numLayers-1):
            inputsNorm = np.maximum(np.dot(weights[layer],inputsNorm)+biases[layer].reshape((len(biases[layer]),1)),0)
        outputs = np.dot(weights[-1],inputsNorm)+biases[-1].reshape((len(biases[-1]),1))

        # Undo output normalization
        for i in range(outputSize):
            for j in range(numInputs):
                outputs[i,j] = outputs[i,j]*self.ranges[-1]+self.means[-1]
        return outputs.T

    def num_inputs(self):
        ''' Get network input size'''
        return self.inputSize

    def num_outputs(self):
        ''' Get network output size'''
        return self.outputSize


    def split_network (self, layer):
        '''
        levels:
        1. check layers
        2. stop at the given layer and
        3. put the given output of the

        4. first network: - WriteNNt with everything splits at [:layers]
        5. save output of the layer'th level as the input
        6. second network - WriteNNt with everything splits at [layers:]
        output three of them.
        TODO: check what is the means
        :param layer:
        :return:
        '''
        if layer >= self.numLayers:
            raise ('Not enough layers to split')
        first_net = copy.deepcopy(self)
        second_net = copy.deepcopy(self)

        first_net.numLayers = layer
        first_net.layerSizes = self.layerSizes[:layer+1]
        first_net.inputSize = self.inputSize
        first_net.outputSize = self.layerSizes[layer]
        first_net.mins = self.mins
        first_net.maxes = self.maxes
        # first_net.means = first_net.means TODO! understand what is the mean
        first_net.ranges = self.ranges # TODO! understand what is the ranges
        first_net.weights = self.weights[:layer+1]
        first_net.biases = self.biases[:layer+1]


        second_net.numLayers = self.numLayers - layer
        second_net.layerSizes = self.layerSizes
        second_net.inputSize = self.layerSizes[layer]
        second_net.outputSize = self.outputSize
        second_net.mins =[-inf for i in range(second_net.inputSize)] # todo
        second_net.maxes = [inf for i in range(second_net.inputSize)]
        second_net.means = [1 for i in range(second_net.inputSize)] # todo
        second_net.ranges = [1 for i in range(second_net.inputSize)] # todo
        second_net.weights = self.weights[layer:]
        second_net.biases = self.biases[layer:]
        return [first_net, second_net]

inputs = [0.63,0,0,0.49,-0.49]
network = NNet('/cs/labs/guykatz/noyahoch/NNet/nnet/noyas_network.nnet')
orig_output = network.evaluate_network(inputs)
splitted_networks = network.split_network(3)
phi = splitted_networks[0].evaluate_network(inputs)
print(phi)
splitted_output = splitted_networks[1].evaluate_network(phi)
print(orig_output)
print(splitted_output)