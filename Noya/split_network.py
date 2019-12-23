
import numpy as np
import copy
from maraboupy import Marabou

def printNet(network):
    print("Layers sizes: "+ str(network.layerSizes), end= "; ")
    print("Number of layers: "+ str(network.numLayers), end= "; ")
    print("Input size: " + str(network.inputSize), end= "; ")
    print("Output size: " +str(network.outputSize))

def split_network_marabou(network, layer):
    if layer <= 1 or layer >= network.numLayers - 1:
        raise Exception("Splitting layer is too small/big")

    # deep copy of original network
    first_net, second_net = copy.deepcopy(network), copy.deepcopy(network)

    # change first network
    first_net.numLayers = layer
    first_net.layerSizes = first_net.layerSizes[:layer+1]
    first_net.outputSize = first_net.layerSizes[-1]
    first_net.weights = first_net.weights[:layer]
    first_net.biases = first_net.biases[:layer+1]

    # change second network
    second_net.numLayers = network.numLayers - layer
    second_net.layerSizes = second_net.layerSizes[layer:]
    second_net.inputSize = second_net.layerSizes[0]
    second_net.weights = network.weights[layer:]
    second_net.biases = network.biases[layer:]

    # avoid zero division
    first_net.maxLayersize = max(first_net.layerSizes[1:])
    second_net.maxLayersize = max(second_net.layerSizes[1:])
    second_net.inputMinimums = [1]* second_net.layerSizes[0]
    second_net.inputMaximums = [2] *second_net.layerSizes[0]
    second_net.inputMeans = [1] * second_net.layerSizes[0]
    second_net.inputRanges = [1]* second_net.layerSizes[0]

    # initialize networks equations
    network.post_init_settings()
    first_net.post_init_settings()
    second_net.post_init_settings()

    return [first_net, second_net]

NETWORK_NAME = "/cs/labs/guykatz/noyahoch/Repo/Marabou/resources/nnet/acasxu/ACASXU_experimental_v2a_2_7.nnet"
# PROPERTY_NAME="/cs/labs/guykatz/noyahoch/Repo/Marabou/resources/resources/properties/acas_property_3.txt"
inputValues = np.array([[0.63,-1,0,150,250]])
network = Marabou.MarabouNetworkNNet(NETWORK_NAME)
printNet(network)
n1, n2 = split_network_marabou(network, 2)
printNet(n1)
printNet(n2)


res_all = network.evaluate(inputValues, useMarabou=True, options=None)
res1 = n1.evaluate(inputValues, useMarabou=True, options=None)
res1 = np.multiply(res1, res1>0)
res2 = n2.evaluate(res1, useMarabou=True, options=None)

print(res_all)
# print(res1)
print (res2)


