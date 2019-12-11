from math import inf
import numpy as np
import copy
from maraboupy import Marabou, MarabouUtils

def printNet(network):
    print("Layers sizes: "+ str(network.layerSizes), end= "; ")
    print("Number of layers: "+ str(network.numLayers), end= "; ")
    print("Input size: " + str(network.inputSize), end= "; ")
    print("Output size: " +str(network.outputSize))

def split_network_marabou(network, layer):
    first_net, second_net = copy.deepcopy(network), copy.deepcopy(network)
    first_net.numLayers = layer -1
    first_net.layerSizes = network.layerSizes[:layer+1]
    first_net.outputSize = network.layerSizes[layer]
    first_net.weights = network.weights[:layer+1]
    first_net.biases = network.biases[:layer+1]
    first_net.variableRanges()
    second_net.numLayers = network.numLayers - layer +1
    second_net.inputSize = network.layerSizes[layer]
    second_net.layerSizes = network.layerSizes[layer:]
    second_net.weights = network.weights[layer:]
    second_net.biases = network.biases[layer:]
    second_net.variableRanges()
    second_net.numberOfVariables()
    # second_net.mins = [0] * second_net.inputSize
    # second_net.maxes = [0] * second_net.inputSize
    # second_net.means = [0] * (second_net.inputSize +1 )
    # second_net.ranges = [0] * (second_net.inputSize + 1)

    return [first_net, second_net]


# inputs = [0.63,0,0,0.49,-0.49]
NETWORK_NAME = "/cs/labs/guykatz/noyahoch/Repo/Marabou/resources/nnet/acasxu/ACASXU_experimental_v2a_2_7.nnet"
PROPERTY_NAME="/cs/labs/guykatz/noyahoch/Repo/Marabou/resources/resources/properties/acas_property_3.txt"
FIRST_FILE= "/cs/labs/guykatz/noyahoch/Repo/Marabou/Noya/firstnet.nnet"
SEC_FILE = "/cs/labs/guykatz/noyahoch/Repo/Marabou/Noya/secnet.nnet"
TRY_NET_NAME = "ACASXU_experimental_v2a_2_7_try1.nnet"
# split network #

network = Marabou.read_nnet(NETWORK_NAME)
# network1 = Marabou.read_nnet(TRY_NET_NAME)

n1, n2 = split_network_marabou(network, 3)
# printNet(network)
# printNet(n1)
# printNet(n2)

inputValues = np.array([[0, 0, 344, 1, 0]])
res_all = network.evaluate(inputValues, useMarabou=True, options=None)
res1 = n1.evaluate(inputValues, useMarabou=True, options=None)
res2 = n2.evaluate(res1, useMarabou=True, options=None)
print(res_all)
print(res1)
print (res2)