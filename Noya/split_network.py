import numpy as np
import copy
from maraboupy import Marabou
from maraboupy.AcasNet import AcasNet
from NNet.utils.writeNNet import writeNNet
import os
NNET_PATH = '/cs/usr/noyahoch/Marabou/Marabou/Noya/nnets'
##### helper functions #####
def printNet(network):
    print("Layers sizes: " + str(network.layerSizes), end="; ")
    print("Number of layers: " + str(network.numLayers), end="; ")
    print("Input size: " + str(network.inputSize), end="; ")
    print("Output size: " + str(network.outputSize))

def write_to_nnet(network, net_name):
    weights = np.array(network.weights)
    biases= np.array(network.biases)
    inputMins = np.array(network.inputMinimums)
    inputMaxes = np.array(network.inputMaximums)
    means = np.array(network.inputMeans)
    ranges = np.array(network.inputRanges)
    fileName = os.path.join(NNET_PATH, net_name + '.nnet')
    writeNNet(weights,biases,inputMins,inputMaxes,means,ranges,fileName)
    return fileName

def init_marabou_obj (network, net_name, property_path):
    network_path = write_to_nnet(network, net_name)
    net = AcasNet(network_path, property_path)
    return net

##### network core functions #####
def split_network_marabou(network, layer):
    if layer <= 1 or layer >= network.numLayers - 1:
        raise Exception("Splitting layer is too small/big")

    # deep copy of original network
    first_net, second_net = copy.deepcopy(network), copy.deepcopy(network)

    # change first network
    first_net.numLayers = layer
    first_net.layerSizes = first_net.layerSizes[:layer + 1]
    first_net.outputSize = first_net.layerSizes[-1]
    first_net.weights = first_net.weights[:layer]
    first_net.biases = first_net.biases[:layer + 1]

    # change second network
    second_net.numLayers = network.numLayers - layer
    second_net.layerSizes = second_net.layerSizes[layer:]
    second_net.inputSize = second_net.layerSizes[0]
    second_net.weights = network.weights[layer:]
    second_net.biases = network.biases[layer:]

    # avoid zero division
    first_net.maxLayersize = max(first_net.layerSizes[1:])
    second_net.maxLayersize = max(second_net.layerSizes[1:])
    second_net.inputMinimums = [1] * second_net.layerSizes[0]
    second_net.inputMaximums = [2] * second_net.layerSizes[0]
    second_net.inputMeans = [1] * (second_net.layerSizes[0] + 1)
    second_net.inputRanges = [1] * (second_net.layerSizes[0] + 1)

    # initialize networks equations
    network.post_init_settings()
    first_net.post_init_settings()
    second_net.post_init_settings()

    return [first_net, second_net]

##### properties functions #####
def extract_activation_pattern(ev_res):
    return np.where(ev_res>0, np.ones_like(ev_res), np.zeros_like(ev_res))

def neg_property_from_activation_pattern(ap):
    # make each >=0 number to -epsilon, makee < number to>=0
    neg_ap = np.where(ap>=0, np.full_like(ap, 0), np.ones_like(ap))
    return neg_ap
def create_property_file(first_property, ap, ap_is_x, prop_name):
    lines = []
    ap=ap[0]
    if ap_is_x:
        for ind, val in enumerate(ap):
            if val > 0:
                sign = '>='
                s_val = 0
            else:
                sign = '<='
                s_val = -0.1
            lines.append('x{} {} {}\n'.format(ind, sign, s_val))
        with open(first_property) as first:
            for line in first:
                if not line.startswith('x'):
                    lines.append(line)
    else:
        with open(first_property) as first:
            for line in first:
                if line.startswith('x'):
                    lines.append(line)
        for ind, val in enumerate(ap):
            if val > 0:
                sign = '>='
                s_val = 0
            else:
                sign = '<='
                s_val = -0.1
            lines.append('+y{} +y{} {} 0\n'.format(ind, ind, sign, s_val))
    fileName = os.path.join('properties', prop_name + '.txt')
    with open(fileName, 'w+') as f:
        f.writelines(lines)
    return fileName

##### flow functions #####

def check_split_unsat(n1, n2, property_path, input):
    res = n1.evaluate(input)
    ap = extract_activation_pattern(res)
    neg_ap = neg_property_from_activation_pattern(ap)
    p1_path = create_property_file(first_property=property_path, ap=neg_ap, ap_is_x=False, prop_name='n1_prop')
    p2_path = create_property_file(first_property=property_path, ap=ap, ap_is_x=True, prop_name='n2_prop')
    n1_obj = init_marabou_obj (n1, "n1", p1_path)
    n2_obj = init_marabou_obj (n2, "n2", p2_path)
    solved_n1 =  n1_obj.solve()[0]
    solved_n2 = n2_obj.solve()[0]
    return solved_n1, solved_n2

def split_check_unsat(network_PATH, property_path, split_level, input,print = True):
    '''
    splits network, check if they fulfill our requirements (n1 unsat negated activation pattern, n2 unsat property) :
    '''
    #create network
    network = Marabou.MarabouNetworkNNet(network_PATH)
    # split network
    n1, n2 = split_network_marabou(network, split_level)
    sat1, sat2 = check_split_unsat(n1, n2, property_path, input)
    return sat1 == sat2

if __name__ == '__main__':
    NETWORK_PATH = "/cs/labs/guykatz/noyahoch/Repo/Marabou/resources/nnet/acasxu/ACASXU_experimental_v2a_2_7.nnet"
    PROPERTY_PATH ="/cs/usr/noyahoch/Marabou/Marabou/resources/properties/acas_property_4.txt"
    inputValues = np.array([[0.63, -1, 0, 150, 250]])
    r1 = split_check_unsat(NETWORK_PATH, PROPERTY_PATH, 2, inputValues)
    print(r1)
    # create network
    # network = Marabou.MarabouNetworkNNet(NETWORK_NAME)
    # printNet(network)
    # # split network in given layer
    # n1, n2 = split_network_marabou(network, 2)
    # printNet(n1)
    # printNet(n2)

    # get sat/unstat of whole network
    # res_all = network.evaluate(inputValues, useMarabou=True, options=None)
    # # get sat/unstat of first part of network
    # res1 = n1.evaluate(inputValues, useMarabou=True, options=None)
    # #
    # res1 = np.multiply(res1, res1 > 0)
    # res2 = n2.evaluate(res1, useMarabou=True, options=None)
    #
    #
    # print(res_all)
    # # print(res1)
    # print(res2)
