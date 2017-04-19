from DiscoverDetection import DiscoverDetection
from CommunityDetection import network
from Evaluation import purity, entropy
from Parameter import *
import math
import numpy
import random
import jsonrpclib

import time

if dataset == 'DBLP':
    from DBLP import *
elif dataset == 'coauthorship':
    from coauthorship import *
elif dataset == 'synthetic':
    from synthetic import *

print "hello!"
# this is to ensure the server runs before the main program starts
# time.sleep(30)
print "start to connect!"

whole_network = jsonrpclib.Server(endpoint)

def generate_cost(seed=0):
    shuffled_nodes = whole_network.nodes()
    for node in target_nodes:
        shuffled_nodes.remove(node)
    if seed != 0:
        random.seed(seed)
    random.shuffle(shuffled_nodes)
    index = 0
    for node in shuffled_nodes:
        index += 1
        cost[node] = 1 / math.pow(index, theta)
    norm = len(shuffled_nodes) / sum(cost.values())
    for node in shuffled_nodes:
        cost[node] *= norm
    for node in target_nodes:
        cost[node] = 1

result_file = open('_'.join(['result', dataset, str(sample_mode), str(theta), str(consider_cost), str(c), \
        'MRFtuning']), 'wb')
# MRF parameter tuning (beta, eta)
for (beta, eta) in [(beta, eta) for beta in map(lambda x: x/10.0, range(1, 100, 3)) \
 for eta in range(1, 30, 2)]:

    purity_dict = dict()
    entropy_dict = dict()
    size_dict = dict()

    for budget in budget_list:
        purity_dict[budget] = list()
        entropy_dict[budget] = list()
        size_dict[budget] = list()

    for k in range(test_num):
        # result_file.write('\n')
        current_budget = budget_list[0]
        cost = dict()
        generate_cost(k)
        current_network = network()
        current_network.add_nodes(target_nodes)
        algorithm = DiscoverDetection(whole_network, current_network, target_nodes, cost, 99999999999, cluster_num, sample_mode, consider_cost, c)
        while True:
            try:
                node_to_sample = algorithm.sample_new_node()
                # print str(algorithm.get_current_cost())
                if algorithm.time_to_update_community() is True:
                    communities = algorithm.update_community_MRF(beta, eta)
                if algorithm.get_current_cost() >= current_budget:
                    communities = algorithm.update_community_MRF(beta, eta, False)
                    communities = algorithm.update_community_MRF(beta, eta, True)
                    tmp_communities = dict(communities)
                    for element in current_network.nodes():
                        if element not in target_nodes:
                            del tmp_communities[element]
                    # result_file.write('Budget: ' + str(current_budget) + '\n')
                    # result_file.write('Sampled Nodes: ' + str(len(current_network.graph().nodes())) + '\n')
                    purity_dict[current_budget].append(purity(tmp_communities, label))
                    entropy_dict[current_budget].append(entropy(tmp_communities, label))
                    size_dict[current_budget].append(len(current_network.graph().nodes()))
                    print 'Purity: ' + str(purity_dict[current_budget][-1])
                    print 'Entropy: ' + str(entropy_dict[current_budget][-1])
                    # result_file.write('Purity: ' + str(purity_dict[current_budget][-1]))
                    # result_file.write('Entropy: ' + str(entropy_dict[current_budget][-1]))
                    # result_file.write('\n')
                    # result_file.flush()
                    print 'current budget is ' + str(current_budget)
                    if budget_list.index(current_budget) + 1 >= len(budget_list):
                        break
                    current_budget = budget_list[budget_list.index(current_budget) + 1]
            except Exception, e:
                print e
                import traceback
                traceback.print_exc()
                break

        del algorithm
        del communities

    print 'beta is ' + str(beta)
    print 'eta is ' + str(eta)
    result_file.write('Purity(beta_' + str(beta) + '_eta_' +str(eta) + '):\t')
    # result_file.write('[')
    for budget in budget_list:
        result_file.write(str(numpy.mean(purity_dict[budget])) + '\t')
    result_file.write('\n')

    #        ' variance: ' + str(numpy.var(purity_list)) + '\n')
    result_file.write('Entropy: \t')
    # result_file.write('[')
    for budget in budget_list:
        result_file.write(str(numpy.mean(entropy_dict[budget])) + '\t')
    result_file.write('\n')

    # result_file.write('Size Avearge: \n')
    # result_file.write('[')
    # for budget in budget_list:
    #     result_file.write(str(numpy.mean(size_dict[budget])) + ' ')
    # result_file.write(']\n')

result_file.close()
