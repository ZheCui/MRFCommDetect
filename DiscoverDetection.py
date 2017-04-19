from pygraph.algorithms.accessibility import connected_components
import random
import math
from CommunityDetection import network

import deepwalkgraph
import node2vec
from gensim.models import Word2Vec
from skipgram import Skipgram
import numpy as np
import scipy.sparse as sp
import copy
import transform
import theano

class DiscoverDetection(object):
    def __init__(self, whole_network, initial_network, target_nodes, cost, budget, cluster_number, sample_mode, consider_cost=False, c=0.1):
        self.whole_network = whole_network
        self.current_network = initial_network
        #input for deepwalk training, contains both current sampled network (self.current_network)
        # and candidate nodes (self.candidates_to_sample)
        self.current_and_candidate_network = copy.deepcopy(initial_network)
        self.model = None
        self.target_nodes = target_nodes
        self.cost = cost
        self.budget = budget
        self.current_cost = 0
        self.sampled_nodes = set()
        self.candidates_to_sample = set()
        self.target_nodes_all_sampled = False
        self.cluster_number = cluster_number
        self.consider_cost = consider_cost
        self.c = c
        self.sample_mode = sample_mode
        self.stage = 1
        self.community_detected = False
        self.network_disconnected_number = len(target_nodes)
        self.neighbors = dict()
        self.stage_1_sample_nodes_num = len(target_nodes)

        self.rank_score = dict()
        self.largest_cost = -1
        self.whole_network_size = self.whole_network.size()

    def detect_disconnected_components(self):
        components = connected_components(self.current_network.graph())
        return components

    def sample_new_node(self):
        if self.sample_mode == 1:
            self.consider_target_nodes = False
            return self.sample_new_node_entropy(5)
        elif self.sample_mode == 2:
            return self.sample_new_node_rw(5)
        elif self.sample_mode == 3:
            self.consider_target_nodes = False
            return self.sample_new_node_greedy(5)
        elif self.sample_mode == 4:
            self.consider_target_nodes = True
            return self.sample_new_node_greedy(5)
        elif self.sample_mode == 5:
            return self.sample_new_node_entropy(5)
        elif self.sample_mode == 6:
            return self.sample_new_node_ncut(5)
        elif self.sample_mode == 7:
            return self.sample_new_node_modularity(5)
        elif self.sample_mode == 8:
            return self.sample_new_node_deepwalk_transform(5)
        elif self.sample_mode == 9:
            return self.sample_new_node_deepwalk(5)
        elif self.sample_mode == 10:
            return self.sample_new_node_node2vec(5)
        elif self.sample_mode == 11:
            return self.sample_new_node_node2vec_transform(5)
        else:
            raise Exception('Sample mode not supported.')

    # added by Zhe, for node2vec + transformation matrix
    def sample_new_node_node2vec_transform(self, sample_num=1):
        if self.stage == 1:
            nodes = self.target_nodes
            random.shuffle(nodes)
            for node in nodes:
                if node not in self.sampled_nodes and \
                        self.current_cost + self.cost[node] <= self.budget:
                    node_to_sample = node
                    self.add_node(node_to_sample)
                    self.stage_1_sample_nodes_num -= 1
                    if self.stage_1_sample_nodes_num == 0:
                        self.stage = 2
                        print self.network_info()
                        if self.current_cost <= self.budget:
                            if self.network_disconnected_number <= self.cluster_number:
                                self.stage = 3
                                self.community_detection()
                            else:
                                nodes_ = self.current_network.nodes()
                                for node_ in nodes_:
                                    if self.current_cost + self.cost[node_] > self.budget or \
                                            node_ in self.sampled_nodes:
                                        continue
                                    neighbors = self.neighbors[node_]
                                    if (len(neighbors) == 0):
                                        continue
                                    count = 0
                                    for neighbor in neighbors:
                                        if neighbor in self.sampled_nodes:
                                            count += 1
                                        self.rank_score[node_] = float(count)
                                if self.consider_cost == True:
                                    sorted_rank_score = sorted(set(self.rank_score.itervalues()))
                                    for node in self.rank_score.iterkeys():
                                        self.rank_score[node] = (sorted_rank_score.index(self.rank_score[node]) + 1) / math.pow(self.cost[node], self.c)

                    break
            if 'node_to_sample' not in locals().keys():
                raise Exception('Budget is exhausted.')
            #print self.network_info()
        elif self.stage == 2:
            if self.rank_score:
                node_to_sample = max(self.rank_score.iterkeys(), key=lambda k: self.rank_score[k])

            if 'node_to_sample' not in locals().keys():
                raise Exception('Budget is exhausted.')
            self.add_node(node_to_sample)
            print self.network_info()
            if self.network_disconnected_number <= self.cluster_number:
                self.stage = 3
                self.community_detection()
        elif self.stage == 3:
            print self.current_and_candidate_network.size()
            print self.current_network.size()
            print len(self.candidates_to_sample)
            pre_G = node2vec.load_edges(self.current_and_candidate_network, undirected=True)
            G = node2vec.Graph(pre_G, False, 1, 1)
            G.preprocess_transition_probs()
            walks = G.simulate_walks(10, 40)
            print("Number of nodes: {}".format(len(G.G.nodes())))

            print("Node2Vec Walking...")
            walks = [map(str, walk) for walk in walks]
            print("Training...")
            model = Word2Vec(walks, size=64, window=5, min_count=0, sg=0, workers=4)
            score = dict()
            prev_nodes_list = dict()
            current_nodes_list = list(model.vocab.keys())
            # file = open('test', 'wb')
            # for k in self.cost:
            #     file.write(str(k) + ' ' + str(self.cost[k]) + '\n')
            # file.close()
            # for the first iteration, just computes the current feature representations distance
            if self.model is None:
                for node in current_nodes_list:
                    # print type(node)
                    # print 'node'
                    # print self.cost[3923]
                    # print self.cost[int(node)]
                    node = int(node)
                    if self.current_cost + self.cost[node] > self.budget or\
                            node in self.sampled_nodes:
                        continue
                    score[node] = -np.linalg.norm(model[str(node)])
            else:
                prev_model = self.model
                prev_nodes_list = list(prev_model.vocab.keys())
                curr_feature_list = []
                prev_feature_list = []
                feature_index = dict()
                ind = 0
                # store the common vertex feature representation from current
                # and previous iterations into a matrix
                for node in model.vocab.keys() and prev_model.vocab.keys():
                    curr_feature_list.append(model[node])
                    prev_feature_list.append(prev_model[node])
                    feature_index[node] = ind
                    ind += 1

                curr_feature_matrix = np.array(curr_feature_list)
                prev_feature_matrix = np.array(prev_feature_list)
                print curr_feature_matrix.shape
                (num_of_vertex, num_of_feature_dim) = curr_feature_matrix.shape

                curr_feature_matrix = np.asarray(curr_feature_matrix, dtype=theano.config.floatX)
                prev_feature_matrix = np.asarray(prev_feature_matrix, dtype=theano.config.floatX)

                curr_theano_matrix = sp.csr_matrix(curr_feature_matrix, shape=(num_of_vertex, num_of_feature_dim))
                prev_theano_matrix = sp.csr_matrix(prev_feature_matrix, shape=(num_of_vertex, num_of_feature_dim))

                curr_transform_matrix = transform.iterations(prev_theano_matrix, curr_theano_matrix, curr_feature_matrix, num_of_vertex)
                print curr_transform_matrix
                curr_transformed_matrix = np.dot(curr_transform_matrix, curr_feature_matrix)

                for node in current_nodes_list:
                    if self.current_cost + self.cost[int(node)] > self.budget or\
                            node in self.sampled_nodes:
                        continue
                    if node in prev_nodes_list:
                        #L2 norm of the model
                        temp_score = np.linalg.norm(prev_model[node] - curr_transformed_matrix[feature_index[node]])
                    # else:
                    #     temp_score = np.linalg.norm(model[node])
                        score[node] = -temp_score
            self.model = model
            
            if self.consider_cost == True:
                if self.current_cost + sample_num > self.budget:
                    raise Exception('Budget is exhausted.')
                sorted_score = sorted(set(score.itervalues()))
                for node in score.iterkeys():
                    score[node] = (sorted_score.index(score[node]) + 1) * math.pow(self.cost[node], self.c)

            if not score:
                raise Exception('Budget is exhausted.')

            while sample_num > 0:
                sample_num -= 1
                while True:
                    node_to_sample = min(score.iterkeys(), key=lambda k: score[k])
                    if self.current_cost + self.cost[int(node_to_sample)] > self.budget:
                        del score[node_to_sample]
                        del node_to_sample
                        if not score:
                            raise Exception('Budget is exhausted.')
                    else:
                        break
                if 'node_to_sample' in locals().keys():
                    print 'added nodes'
                    print node_to_sample
                    self.add_node(node_to_sample)
                    del score[node_to_sample]
                    if not score:
                        break

        if 'node_to_sample' not in locals().keys():
            return 0
    #added by Zhe, for node2vec only
    def sample_new_node_node2vec(self, sample_num=1):
        if self.stage == 1:
            nodes = self.target_nodes
            random.shuffle(nodes)
            for node in nodes:
                if node not in self.sampled_nodes and \
                        self.current_cost + self.cost[node] <= self.budget:
                    node_to_sample = node
                    self.add_node(node_to_sample)
                    self.stage_1_sample_nodes_num -= 1
                    if self.stage_1_sample_nodes_num == 0:
                        self.stage = 2
                        print self.network_info()
                        if self.current_cost <= self.budget:
                            if self.network_disconnected_number <= self.cluster_number:
                                self.stage = 3
                                self.community_detection()
                            else:
                                nodes_ = self.current_network.nodes()
                                for node_ in nodes_:
                                    if self.current_cost + self.cost[node_] > self.budget or \
                                            node_ in self.sampled_nodes:
                                        continue
                                    neighbors = self.neighbors[node_]
                                    if (len(neighbors) == 0):
                                        continue
                                    count = 0
                                    for neighbor in neighbors:
                                        if neighbor in self.sampled_nodes:
                                            count += 1
                                        self.rank_score[node_] = float(count)
                                if self.consider_cost == True:
                                    sorted_rank_score = sorted(set(self.rank_score.itervalues()))
                                    for node in self.rank_score.iterkeys():
                                        self.rank_score[node] = (sorted_rank_score.index(self.rank_score[node]) + 1) / math.pow(self.cost[node], self.c)

                    break
            if 'node_to_sample' not in locals().keys():
                raise Exception('Budget is exhausted.')
            #print self.network_info()
        elif self.stage == 2:
            if self.rank_score:
                node_to_sample = max(self.rank_score.iterkeys(), key=lambda k: self.rank_score[k])

            if 'node_to_sample' not in locals().keys():
                raise Exception('Budget is exhausted.')
            self.add_node(node_to_sample)
            print self.network_info()
            if self.network_disconnected_number <= self.cluster_number:
                self.stage = 3
                self.community_detection()
        elif self.stage == 3:
            print self.current_and_candidate_network.size()
            print self.current_network.size()
            print len(self.candidates_to_sample)
            pre_G = node2vec.load_edges(self.current_and_candidate_network, undirected=True)
            G = node2vec.Graph(pre_G, False, 1, 1)
            G.preprocess_transition_probs()
            walks = G.simulate_walks(10, 40)
            print("Number of nodes: {}".format(len(G.G.nodes())))

            print("Node2Vec Walking...")
            walks = [map(str, walk) for walk in walks]
            print("Training...")
            model = Word2Vec(walks, size=64, window=5, min_count=0, sg=0, workers=4)
            score = dict()
            prev_nodes_list = dict()
            current_nodes_list = list(model.vocab.keys())
            # file = open('test', 'wb')
            # for k in self.cost:
            #     file.write(str(k) + ' ' + str(self.cost[k]) + '\n')
            # file.close()
            # for the first iteration, just computes the current feature representations distance
            if self.model is None:
                for node in current_nodes_list:
                    # print type(node)
                    # print 'node'
                    # print self.cost[3923]
                    # print self.cost[int(node)]
                    node = int(node)
                    if self.current_cost + self.cost[node] > self.budget or\
                            node in self.sampled_nodes:
                        continue
                    score[node] = -np.linalg.norm(model[str(node)])
            else:
                prev_model = self.model
                prev_nodes_list = list(prev_model.vocab.keys())

                for node in current_nodes_list:
                    if self.current_cost + self.cost[int(node)] > self.budget or\
                            node in self.sampled_nodes:
                        continue
                    if node in prev_nodes_list:
                        #L2 norm of the model
                        temp_score = np.linalg.norm(prev_model[node] - model[node])
                    # else:
                    #     temp_score = np.linalg.norm(model[node])
                        score[node] = -temp_score
            self.model = model
            
            if self.consider_cost == True:
                if self.current_cost + sample_num > self.budget:
                    raise Exception('Budget is exhausted.')
                sorted_score = sorted(set(score.itervalues()))
                for node in score.iterkeys():
                    score[node] = (sorted_score.index(score[node]) + 1) * math.pow(self.cost[node], self.c)

            if not score:
                raise Exception('Budget is exhausted.')

            while sample_num > 0:
                sample_num -= 1
                while True:
                    node_to_sample = min(score.iterkeys(), key=lambda k: score[k])
                    if self.current_cost + self.cost[int(node_to_sample)] > self.budget:
                        del score[node_to_sample]
                        del node_to_sample
                        if not score:
                            raise Exception('Budget is exhausted.')
                    else:
                        break
                if 'node_to_sample' in locals().keys():
                    print 'added nodes'
                    print node_to_sample
                    self.add_node(node_to_sample)
                    del score[node_to_sample]
                    if not score:
                        break

        if 'node_to_sample' not in locals().keys():
            return 0

    #added by Zhe, for deepwalk only
    def sample_new_node_deepwalk(self, sample_num=1):
        if self.stage == 1:
            nodes = self.target_nodes
            random.shuffle(nodes)
            for node in nodes:
                if node not in self.sampled_nodes and \
                        self.current_cost + self.cost[node] <= self.budget:
                    node_to_sample = node
                    self.add_node(node_to_sample)
                    self.stage_1_sample_nodes_num -= 1
                    if self.stage_1_sample_nodes_num == 0:
                        self.stage = 2
                        print self.network_info()
                        if self.current_cost <= self.budget:
                            if self.network_disconnected_number <= self.cluster_number:
                                self.stage = 3
                                self.community_detection()
                            else:
                                nodes_ = self.current_network.nodes()
                                for node_ in nodes_:
                                    if self.current_cost + self.cost[node_] > self.budget or \
                                            node_ in self.sampled_nodes:
                                        continue
                                    neighbors = self.neighbors[node_]
                                    if (len(neighbors) == 0):
                                        continue
                                    count = 0
                                    for neighbor in neighbors:
                                        if neighbor in self.sampled_nodes:
                                            count += 1
                                        self.rank_score[node_] = float(count)
                                if self.consider_cost == True:
                                    sorted_rank_score = sorted(set(self.rank_score.itervalues()))
                                    for node in self.rank_score.iterkeys():
                                        self.rank_score[node] = (sorted_rank_score.index(self.rank_score[node]) + 1) / math.pow(self.cost[node], self.c)

                    break
            if 'node_to_sample' not in locals().keys():
                raise Exception('Budget is exhausted.')
            #print self.network_info()
        elif self.stage == 2:
            if self.rank_score:
                node_to_sample = max(self.rank_score.iterkeys(), key=lambda k: self.rank_score[k])

            if 'node_to_sample' not in locals().keys():
                raise Exception('Budget is exhausted.')
            self.add_node(node_to_sample)
            print self.network_info()
            if self.network_disconnected_number <= self.cluster_number:
                self.stage = 3
                self.community_detection()
        elif self.stage == 3:
            print self.current_and_candidate_network.size()
            print self.current_network.size()
            print len(self.candidates_to_sample)
            G = deepwalkgraph.load_edges(self.current_and_candidate_network, undirected=True)
            print("Number of nodes: {}".format(len(G.nodes())))

            num_walks = len(G.nodes()) * 10

            print("Number of walks: {}".format(num_walks))

            data_size = num_walks * 40

            print("Data size (walks*length): {}".format(data_size))

            print("Walking...")
            walks = deepwalkgraph.build_deepwalk_corpus(G, num_paths=10,
                                        path_length=40, alpha=0, rand=random.Random(0))
            print("Training...")
            model = Word2Vec(walks, size=64, window=5, min_count=0, workers=4)
            # print 'word is '
            # print list(model.vocab.keys())
            # print 1029 in model.vocab
            score = dict()
            prev_nodes_list = dict()
            current_nodes_list = list(model.vocab.keys())
            # for the first iteration, just computes the current feature representations distance
            if self.model is None:
                for node in current_nodes_list:
                    if self.current_cost + self.cost[node] > self.budget or\
                            node in self.sampled_nodes:
                        continue
                    score[node] = -np.linalg.norm(model[node])
            else:
                prev_model = self.model
                prev_nodes_list = list(prev_model.vocab.keys())

                for node in current_nodes_list:
                    if self.current_cost + self.cost[node] > self.budget or\
                            node in self.sampled_nodes:
                        continue
                    if node in prev_nodes_list:
                        #L2 norm of the model
                        temp_score = np.linalg.norm(prev_model[node] - model[node])
                    # else:
                    #     temp_score = np.linalg.norm(model[node])
                        score[node] = -temp_score
            self.model = model
            
            if self.consider_cost == True:
                if self.current_cost + sample_num > self.budget:
                    raise Exception('Budget is exhausted.')
                sorted_score = sorted(set(score.itervalues()))
                for node in score.iterkeys():
                    score[node] = (sorted_score.index(score[node]) + 1) * math.pow(self.cost[node], self.c)

            if not score:
                raise Exception('Budget is exhausted.')

            while sample_num > 0:
                sample_num -= 1
                while True:
                    node_to_sample = min(score.iterkeys(), key=lambda k: score[k])
                    if self.current_cost + self.cost[node_to_sample] > self.budget:
                        del score[node_to_sample]
                        del node_to_sample
                        if not score:
                            raise Exception('Budget is exhausted.')
                    else:
                        break
                if 'node_to_sample' in locals().keys():
                    self.add_node(node_to_sample)
                    del score[node_to_sample]
                    if not score:
                        break

        if 'node_to_sample' not in locals().keys():
            return 0

    #added by Zhe, for deepwalk with transformation construction
    def sample_new_node_deepwalk_transform(self, sample_num=1):
        if self.stage == 1:
            nodes = self.target_nodes
            random.shuffle(nodes)
            for node in nodes:
                if node not in self.sampled_nodes and \
                        self.current_cost + self.cost[node] <= self.budget:
                    node_to_sample = node
                    self.add_node(node_to_sample)
                    self.stage_1_sample_nodes_num -= 1
                    if self.stage_1_sample_nodes_num == 0:
                        self.stage = 2
                        print self.network_info()
                        if self.current_cost <= self.budget:
                            if self.network_disconnected_number <= self.cluster_number:
                                self.stage = 3
                                self.community_detection()
                            else:
                                nodes_ = self.current_network.nodes()
                                for node_ in nodes_:
                                    if self.current_cost + self.cost[node_] > self.budget or \
                                            node_ in self.sampled_nodes:
                                        continue
                                    neighbors = self.neighbors[node_]
                                    if (len(neighbors) == 0):
                                        continue
                                    count = 0
                                    for neighbor in neighbors:
                                        if neighbor in self.sampled_nodes:
                                            count += 1
                                        self.rank_score[node_] = float(count)
                                if self.consider_cost == True:
                                    sorted_rank_score = sorted(set(self.rank_score.itervalues()))
                                    for node in self.rank_score.iterkeys():
                                        self.rank_score[node] = (sorted_rank_score.index(self.rank_score[node]) + 1) / math.pow(self.cost[node], self.c)

                    break
            if 'node_to_sample' not in locals().keys():
                raise Exception('Budget is exhausted.')
            #print self.network_info()
        elif self.stage == 2:
            if self.rank_score:
                node_to_sample = max(self.rank_score.iterkeys(), key=lambda k: self.rank_score[k])

            if 'node_to_sample' not in locals().keys():
                raise Exception('Budget is exhausted.')
            self.add_node(node_to_sample)
            print self.network_info()
            if self.network_disconnected_number <= self.cluster_number:
                self.stage = 3
                self.community_detection()
        elif self.stage == 3:
            print self.current_and_candidate_network.size()
            print self.current_network.size()
            print len(self.candidates_to_sample)
            G = deepwalkgraph.load_edges(self.current_and_candidate_network, undirected=True)
            print("Number of nodes: {}".format(len(G.nodes())))

            num_walks = len(G.nodes()) * 10

            print("Number of walks: {}".format(num_walks))

            data_size = num_walks * 40

            print("Data size (walks*length): {}".format(data_size))

            print("Walking...")
            walks = deepwalkgraph.build_deepwalk_corpus(G, num_paths=10,
                                        path_length=40, alpha=0, rand=random.Random(0))
            print("Training...")
            model = Word2Vec(walks, size=64, window=5, min_count=0, workers=4)
            # print 'word is '
            # print list(model.vocab.keys())
            # print 1029 in model.vocab
            score = dict()
            prev_nodes_list = dict()
            current_nodes_list = list(model.vocab.keys())
            # for the first iteration, just computes the current feature representations distance
            if self.model is None:
                for node in current_nodes_list:
                    if self.current_cost + self.cost[node] > self.budget or\
                            node in self.sampled_nodes:
                        continue
                    score[node] = -np.linalg.norm(model[node])
            else:
                prev_model = self.model
                prev_nodes_list = list(prev_model.vocab.keys())
                curr_feature_list = []
                prev_feature_list = []
                feature_index = dict()
                ind = 0
                # store the common vertex feature representation from current
                # and previous iterations into a matrix
                for node in model.vocab.keys() and prev_model.vocab.keys():
                    curr_feature_list.append(model[node])
                    prev_feature_list.append(prev_model[node])
                    feature_index[node] = ind
                    ind += 1

                curr_feature_matrix = np.array(curr_feature_list)
                prev_feature_matrix = np.array(prev_feature_list)
                print curr_feature_matrix.shape
                (num_of_vertex, num_of_feature_dim) = curr_feature_matrix.shape

                curr_feature_matrix = np.asarray(curr_feature_matrix, dtype=theano.config.floatX)
                prev_feature_matrix = np.asarray(prev_feature_matrix, dtype=theano.config.floatX)

                curr_theano_matrix = sp.csr_matrix(curr_feature_matrix, shape=(num_of_vertex, num_of_feature_dim))
                prev_theano_matrix = sp.csr_matrix(prev_feature_matrix, shape=(num_of_vertex, num_of_feature_dim))

                curr_transform_matrix = transform.iterations(prev_theano_matrix, curr_theano_matrix, curr_feature_matrix, num_of_vertex)
                print curr_transform_matrix
                curr_transformed_matrix = np.dot(curr_transform_matrix, curr_feature_matrix)

                for node in current_nodes_list:
                    if self.current_cost + self.cost[node] > self.budget or\
                            node in self.sampled_nodes:
                        continue
                    if node in prev_nodes_list:
                        #L2 norm of the model
                        temp_score = np.linalg.norm(prev_model[node] - curr_transformed_matrix[feature_index[node]])
                    # else:
                    #     temp_score = np.linalg.norm(model[node])
                        score[node] = -temp_score
            self.model = model
            
            if self.consider_cost == True:
                if self.current_cost + sample_num > self.budget:
                    raise Exception('Budget is exhausted.')
                sorted_score = sorted(set(score.itervalues()))
                for node in score.iterkeys():
                    score[node] = (sorted_score.index(score[node]) + 1) * math.pow(self.cost[node], self.c)

            if not score:
                raise Exception('Budget is exhausted.')

            while sample_num > 0:
                sample_num -= 1
                while True:
                    node_to_sample = min(score.iterkeys(), key=lambda k: score[k])
                    if self.current_cost + self.cost[node_to_sample] > self.budget:
                        del score[node_to_sample]
                        del node_to_sample
                        if not score:
                            raise Exception('Budget is exhausted.')
                    else:
                        break
                if 'node_to_sample' in locals().keys():
                    self.add_node(node_to_sample)
                    del score[node_to_sample]
                    if not score:
                        break


            # self.add_node()
            # model.save_word2vec_format('dataset/outputtest')

        if 'node_to_sample' not in locals().keys():
            return 0

    def sample_new_node_modularity(self, sample_num=1):
        if self.stage == 1:
            nodes = self.target_nodes
            random.shuffle(nodes)
            for node in nodes:
                if node not in self.sampled_nodes and \
                        self.current_cost + self.cost[node] <= self.budget:
                    node_to_sample = node
                    self.add_node(node_to_sample)
                    self.stage_1_sample_nodes_num -= 1
                    if self.stage_1_sample_nodes_num == 0:
                        self.stage = 2
                        print self.network_info()
                        if self.current_cost <= self.budget:
                            if self.network_disconnected_number <= self.cluster_number:
                                self.stage = 3
                                self.community_detection()
                            else:
                                nodes_ = self.current_network.nodes()
                                for node_ in nodes_:
                                    if self.current_cost + self.cost[node_] > self.budget or \
                                            node_ in self.sampled_nodes:
                                        continue
                                    neighbors = self.neighbors[node_]
                                    if (len(neighbors) == 0):
                                        continue
                                    count = 0
                                    for neighbor in neighbors:
                                        if neighbor in self.sampled_nodes:
                                            count += 1
                                        self.rank_score[node_] = float(count)
                                if self.consider_cost == True:
                                    sorted_rank_score = sorted(set(self.rank_score.itervalues()))
                                    for node in self.rank_score.iterkeys():
                                        self.rank_score[node] = (sorted_rank_score.index(self.rank_score[node]) + 1) / math.pow(self.cost[node], self.c)

                    break
            if 'node_to_sample' not in locals().keys():
                raise Exception('Budget is exhausted.')
            #print self.network_info()
        elif self.stage == 2:
            if self.rank_score:
                node_to_sample = max(self.rank_score.iterkeys(), key=lambda k: self.rank_score[k])
            #sorted_degree = sorted(self.rank_score.items(), key=lambda x: x[1])
            #for node in sorted_degree:
            #   node_to_sample = node[0]
            #   break
            if 'node_to_sample' not in locals().keys():
                raise Exception('Budget is exhausted.')
            self.add_node(node_to_sample)
            print self.network_info()
            if self.network_disconnected_number <= self.cluster_number:
                self.stage = 3
                self.community_detection()
        elif self.stage == 3:
            score = dict()

            label = self.current_network.get_label()
            eii = [0] * self.cluster_number
            ai = [0] * self.cluster_number
            m = 0
            for node, node_label in label.iteritems():
                neighbors = self.current_network.neighbors(node)
                for neighbor in neighbors:
                    if label[neighbor] == node_label:
                        eii[node_label] += 1
                    ai[node_label] += 1
                    m += 1

            for node in self.candidates_to_sample:
                if self.current_cost + self.cost[node] > self.budget or\
                        node in self.sampled_nodes:
                    continue
                #if self.sample_mode == 1:
                tmp_score = [0] * self.cluster_number
                for k in xrange(self.cluster_number):
                    neighbors = self.neighbors[node]
                    tmp_eii = list(eii)
                    tmp_ai = list(ai)
                    tmp_m = m
                    for neighbor in neighbors:
                        if label[neighbor] == k:
                            tmp_eii[k] += 2
                        tmp_ai[k] += 1
                        tmp_ai[label[neighbor]] += 1
                        tmp_m += 2
                    tmp_score[k] = sum([float(i) / tmp_m - math.pow(float(j) / tmp_m, 2) for i, j in zip(tmp_eii, tmp_ai)])
                score[node] = -max(tmp_score)

            if self.consider_cost == True:
                if self.current_cost + sample_num > self.budget:
                    raise Exception('Budget is exhausted.')
                sorted_score = sorted(set(score.itervalues()))
                for node in score.iterkeys():
                    score[node] = (sorted_score.index(score[node]) + 1) * math.pow(self.cost[node], self.c)

            if not score:
                raise Exception('Budget is exhausted.')

            while sample_num > 0:
                sample_num -= 1
                while True:
                    node_to_sample = min(score.iterkeys(), key=lambda k: score[k])
                    if self.current_cost + self.cost[node_to_sample] > self.budget:
                        del score[node_to_sample]
                        del node_to_sample
                        if not score:
                            raise Exception('Budget is exhausted.')
                    else:
                        break
                if 'node_to_sample' in locals().keys():
                    self.add_node(node_to_sample)
                    del score[node_to_sample]
                    if not score:
                        break
            #print self.network_info()
            #if self.target_nodes_all_sampled == False:
            #   self.target_nodes_all_sampled = True
            #   for node in self.target_nodes:
            #       if node not in self.sampled_nodes:
            #           self.target_nodes_all_sampled = False
                #if self.target_nodes_all_sampled == True:
                #   self.current_network.erase_labels()
        if 'node_to_sample' not in locals().keys():
            return 0

    def sample_new_node_ncut(self, sample_num=1):
        if self.stage == 1:
            nodes = self.target_nodes
            random.shuffle(nodes)
            for node in nodes:
                if node not in self.sampled_nodes and \
                        self.current_cost + self.cost[node] <= self.budget:
                    node_to_sample = node
                    self.add_node(node_to_sample)
                    self.stage_1_sample_nodes_num -= 1
                    if self.stage_1_sample_nodes_num == 0:
                        self.stage = 2
                        print 'stage 2'
                        print self.network_info()
                        if self.current_cost <= self.budget:
                            if self.network_disconnected_number <= self.cluster_number:
                                print 'stage 3'
                                self.stage = 3
                                self.community_detection()
                            else:
                                nodes_ = self.current_network.nodes()
                                for node_ in nodes_:
                                    if self.current_cost + self.cost[node_] > self.budget or \
                                            node_ in self.sampled_nodes:
                                        continue
                                    neighbors = self.neighbors[node_]
                                    if (len(neighbors) == 0):
                                        continue
                                    count = 0
                                    print 'neighbor'
                                    for neighbor in neighbors:
                                        if neighbor in self.sampled_nodes:
                                            count += 1
                                        print 'rank_score'
                                        self.rank_score[node_] = float(count)
                                if self.consider_cost == True:
                                    sorted_rank_score = sorted(set(self.rank_score.itervalues()))
                                    for node in self.rank_score.iterkeys():
                                        self.rank_score[node] = (sorted_rank_score.index(self.rank_score[node]) + 1) / math.pow(self.cost[node], self.c)
                    break
            if 'node_to_sample' not in locals().keys():
                raise Exception('Budget is exhausted.')
            #print self.network_info()
        elif self.stage == 2:
            if self.rank_score:
                node_to_sample = max(self.rank_score.iterkeys(), key=lambda k: self.rank_score[k])
            #sorted_degree = sorted(self.rank_score.items(), key=lambda x: x[1])
            #for node in sorted_degree:
            #   node_to_sample = node[0]
            #   break
            print 'node to sample is ' + str(self.rank_score)
            if 'node_to_sample' not in locals().keys():
                raise Exception('Budget is exhausted.')
            self.add_node(node_to_sample)
            print self.network_info()
            if self.network_disconnected_number <= self.cluster_number:
                self.stage = 3
                self.community_detection()
        elif self.stage == 3:
            score = dict()

            label = self.current_network.get_label()
            cut = [0] * self.cluster_number
            assoc = [0] * self.cluster_number
            for node, node_label in label.iteritems():
                neighbors = self.current_network.neighbors(node)
                for neighbor in neighbors:
                    if label[neighbor] != node_label:
                        cut[node_label] += 1
                    assoc[node_label] += 1
            for node in self.candidates_to_sample:
                if self.current_cost + self.cost[node] > self.budget or\
                        node in self.sampled_nodes:
                    continue
                #if self.sample_mode == 1:
                tmp_score = [0] * self.cluster_number
                for k in xrange(self.cluster_number):
                    neighbors = self.neighbors[node]
                    tmp_assoc = list(assoc)
                    tmp_cut = list(cut)
                    for neighbor in neighbors:
                        if label[neighbor] != k:
                            tmp_cut[k] += 1
                            tmp_cut[label[neighbor]] += 1
                        tmp_assoc[k] += 1
                        tmp_assoc[label[neighbor]] += 1
                    tmp_score[k] = sum([float(i) / j for i, j in zip(tmp_cut, tmp_assoc)])
                    score[node] = min(tmp_score)

            if self.consider_cost == True:
                if self.current_cost + sample_num > self.budget:
                    raise Exception('Budget is exhausted.')
                sorted_score = sorted(set(score.itervalues()))
                for node in score.iterkeys():
                    score[node] = (sorted_score.index(score[node]) + 1) * math.pow(self.cost[node], self.c)

            if not score:
                raise Exception('Budget is exhausted.')

            while sample_num > 0:
                sample_num -= 1
                while True:
                    node_to_sample = min(score.iterkeys(), key=lambda k: score[k])
                    '''
                    d=list()
                    for node in score:
                        if score[node] == score[node_to_sample]:
                            d.append(node)
                    print d
                    #print node_to_sample
                    neighbors = self.neighbors[node_to_sample]
                    tmp_score = [0] * self.cluster_number
                    for neighbor in neighbors:
                        if neighbor in label:
                            tmp_score[label[neighbor]] += 1
                    print tmp_score
                    #for neighbor in neighbors:
                    #    print str(len(self.current_network.neighbors(neighbor))) + "  " + str(neighbor) + " " + str(len(self.neighbors[neighbor]))
                    '''
                    if self.current_cost + self.cost[node_to_sample] > self.budget:
                        del score[node_to_sample]
                        del node_to_sample
                        if not score:
                            raise Exception('Budget is exhausted.')
                    else:
                        break
                if 'node_to_sample' in locals().keys():
                    self.add_node(node_to_sample)
                    del score[node_to_sample]
                    if not score:
                        break
            #print self.network_info()
            #if self.target_nodes_all_sampled == False:
            #   self.target_nodes_all_sampled = True
            #   for node in self.target_nodes:
            #       if node not in self.sampled_nodes:
            #           self.target_nodes_all_sampled = False
                #if self.target_nodes_all_sampled == True:
                #   self.current_network.erase_labels()
        if 'node_to_sample' not in locals().keys():
            return 0

    def sample_new_node_rw(self, sample_num=1):
        if self.stage == 1:
            nodes = self.target_nodes
            random.shuffle(nodes)
            for node in nodes:
                if node not in self.sampled_nodes and \
                        self.current_cost + self.cost[node] <= self.budget:
                    node_to_sample = node
                    self.add_node(node_to_sample)
                    break
            if 'node_to_sample' not in locals().keys():
                raise Exception('Budget is exhausted.')
            if len(self.sampled_nodes) == len(self.target_nodes):
                self.stage = 2
            #print self.network_info()
        elif self.stage == 2:
            while sample_num > 0:
                if 'node_to_sample' in locals().keys():
                    del node_to_sample
                sample_num -= 1
                real_candidates = set()
                nodes = self.candidates_to_sample
                for node in nodes:
                    if node not in self.sampled_nodes and \
                            self.current_cost + self.cost[node] <= self.budget:
                        real_candidates.add(node)
                nodes = list(real_candidates)
                cost_sum = 0
                for node in nodes:
                    if self.consider_cost == True:
                        cost_sum += 1 / math.pow(self.cost[node], self.c)
                    else:
                        cost_sum += 1
                rand_value = random.random()
                random.shuffle(nodes)
                fraction = 0
                for node in nodes:
                    if self.consider_cost == True:
                        fraction += 1 / math.pow(self.cost[node], self.c) / cost_sum
                    else:
                        fraction += float(1) / cost_sum
                    if fraction >= rand_value:
                        node_to_sample = node
                        self.add_node(node_to_sample)
                        break
                if 'node_to_sample' not in locals().keys():
                    raise Exception('Budget is exhausted.')
            #print self.network_info()
        if 'node_to_sample' not in locals().keys():
            return 0
        return node_to_sample

    def sample_new_node_greedy(self, sample_num=1):
        if self.stage == 1:
            nodes = self.target_nodes
            random.shuffle(nodes)
            for node in nodes:
                if node not in self.sampled_nodes and \
                        self.current_cost + self.cost[node] <= self.budget:
                    node_to_sample = node
                    self.add_node(node_to_sample)
                    break
            if 'node_to_sample' not in locals().keys():
                raise Exception('Budget is exhausted.')
            if len(self.sampled_nodes) == len(self.target_nodes):
                self.stage = 2
                for node in self.candidates_to_sample:
                    if self.current_cost + self.cost[node] > self.budget or \
                            node in self.sampled_nodes:
                        continue
                    self.compute_rank_score(node)
            #print self.network_info()
        elif self.stage == 2:
            while sample_num > 0:
                if 'node_to_sample' in locals().keys():
                    del node_to_sample
                sample_num -= 1
                if self.rank_score:
                    node_to_sample = max(self.rank_score.iterkeys(), key=lambda k: self.rank_score[k])
                    self.add_node(node_to_sample)
                #sorted_degree = sorted(self.rank_score.items(), key=lambda x: x[1])
                #for node in sorted_degree:
                #   node_to_sample = node[0]
                #   break
                if 'node_to_sample' not in locals().keys():
                    raise Exception('Budget is exhausted.')
            #print self.network_info()
        return node_to_sample

    def sample_new_node_entropy(self, sample_num=1):
        if self.stage == 1:
            nodes = self.target_nodes
            random.shuffle(nodes)
            for node in nodes:
                if node not in self.sampled_nodes and \
                        self.current_cost + self.cost[node] <= self.budget:
                    node_to_sample = node
                    self.add_node(node_to_sample)
                    self.stage_1_sample_nodes_num -= 1
                    if self.stage_1_sample_nodes_num == 0:
                        self.stage = 2
                        print self.network_info()
                        if self.current_cost <= self.budget:
                            if self.network_disconnected_number <= self.cluster_number:
                                self.stage = 3
                                self.community_detection()
                            else:
                                nodes_ = self.current_network.nodes()
                                for node_ in nodes_:
                                    if self.current_cost + self.cost[node_] > self.budget or \
                                            node_ in self.sampled_nodes:
                                        continue
                                    neighbors = self.neighbors[node_]
                                    if (len(neighbors) == 0):
                                        continue
                                    count = 0
                                    for neighbor in neighbors:
                                        if neighbor in self.sampled_nodes:
                                            count += 1
                                        self.rank_score[node_] = float(count)
                                if self.consider_cost == True:
                                    sorted_rank_score = sorted(set(self.rank_score.itervalues()))
                                    for node in self.rank_score.iterkeys():
                                        self.rank_score[node] = (sorted_rank_score.index(self.rank_score[node]) + 1) / math.pow(self.cost[node], self.c)
                    break
            if 'node_to_sample' not in locals().keys():
                raise Exception('Budget is exhausted.')
            #print self.network_info()
        elif self.stage == 2:
            if self.rank_score:
                node_to_sample = max(self.rank_score.iterkeys(), key=lambda k: self.rank_score[k])
            #sorted_degree = sorted(self.rank_score.items(), key=lambda x: x[1])
            #for node in sorted_degree:
            #   node_to_sample = node[0]
            #   break
            if 'node_to_sample' not in locals().keys():
                raise Exception('Budget is exhausted.')
            self.add_node(node_to_sample)
            print self.network_info()
            if self.network_disconnected_number <= self.cluster_number:
                self.stage = 3
                self.community_detection()
        elif self.stage == 3:
            score = dict()
            label = self.current_network.get_label()
            for node in self.candidates_to_sample:
                if node not in self.sampled_nodes:
                    neighbors = self.neighbors[node]
                    cluster_dist = [0] * self.cluster_number
                    total = 0
                    for neighbor in neighbors:
                        cluster_dist[label[neighbor]] += 1
                        total += 1
                    entropy = 0
                    for cluster in range(self.cluster_number):
                        if total == 0:
                            continue
                        prob = cluster_dist[cluster] / float(total)
                        if prob != 0:
                            entropy += - prob * math.log(prob)
                    self.compute_rank_score(node)
                    if self.rank_score[node] == 0:
                        continue
                    score[node] = entropy / self.rank_score[node]
                    # print score[node]
            if self.consider_cost == True:
                if self.current_cost + sample_num > self.budget:
                    raise Exception('Budget is exhausted.')
                sorted_score = sorted(set(score.itervalues()))
                for node in score.iterkeys():
                    score[node] = (sorted_score.index(score[node]) + 1) * math.pow(self.cost[node], self.c)

            if not score:
                raise Exception('Budget is exhausted.')

            while sample_num > 0:
                sample_num -= 1
                while True:
                    node_to_sample = min(score.iterkeys(), key=lambda k: score[k])
                    '''
                    d=list()
                    for node in score:
                        if score[node] == score[node_to_sample]:
                            d.append(node)
                    print d
                    #print node_to_sample
                    neighbors = self.neighbors[node_to_sample]
                    tmp_score = [0] * self.cluster_number
                    for neighbor in neighbors:
                        if neighbor in label:
                            tmp_score[label[neighbor]] += 1
                    print tmp_score
                    #for neighbor in neighbors:
                    #    print str(len(self.current_network.neighbors(neighbor))) + "  " + str(neighbor) + " " + str(len(self.neighbors[neighbor]))
                    '''
                    if self.current_cost + self.cost[node_to_sample] > self.budget:
                        del score[node_to_sample]
                        del node_to_sample
                        if not score:
                            raise Exception('Budget is exhausted.')
                    else:
                        break
                if 'node_to_sample' in locals().keys():
                    self.add_node(node_to_sample)
                    del score[node_to_sample]
                    if not score:
                        break
            #print self.network_info()
            #if self.target_nodes_all_sampled == False:
            #   self.target_nodes_all_sampled = True
            #   for node in self.target_nodes:
            #       if node not in self.sampled_nodes:
            #           self.target_nodes_all_sampled = False
                #if self.target_nodes_all_sampled == True:
                #   self.current_network.erase_labels()
        if 'node_to_sample' not in locals().keys():
            return 0

    def add_node(self, node):
        # node = int(node)
        self.current_cost += self.cost[int(node)]
        self.sampled_nodes.add(node)
        neighbors = self.whole_network.neighbors(int(node))
        for neighbor in neighbors:
            if neighbor in self.neighbors:
                self.neighbors[neighbor].append(node)
            else:
                self.neighbors[neighbor] = [node]
        self.neighbors[node] = neighbors
        if not self.current_network.has_node(node):
            self.current_network.add_node(node)
        if node in self.candidates_to_sample:
            self.candidates_to_sample.remove(node)
        for neighbor in neighbors:
            if neighbor in self.current_network.nodes():
                if not self.current_network.has_edge((node, neighbor)):
                    self.current_network.add_edge((node, neighbor))
            else:
                self.candidates_to_sample.add(neighbor)

        #add nodes to current_and_candidate_network
        if not self.current_and_candidate_network.has_node(node):
            print "added" + str(node)
            self.current_and_candidate_network.add_node(node)
        for neighbor in neighbors:
            if neighbor not in self.current_and_candidate_network.nodes():
                self.current_and_candidate_network.add_node(neighbor)
            if not self.current_and_candidate_network.has_edge((node, neighbor)):
                self.current_and_candidate_network.add_edge((node, neighbor))

        if self.sample_mode != 2 and self.stage == 2:
            for neighbor in neighbors:
                if self.current_cost + self.cost[neighbor] <= self.budget and \
                        neighbor in self.candidates_to_sample:
                    if self.cost[neighbor] > self.largest_cost:
                        self.largest_cost = self.cost[neighbor]
                    self.compute_rank_score(neighbor)
            if node in self.rank_score:
                del self.rank_score[node]
            if self.budget - self.current_cost < self.largest_cost:
                node_list = list(self.rank_score.iterkeys())
                for node in node_list:
                    if self.current_cost + self.cost[node] > self.budget:
                        del self.rank_score[node]
            

    def compute_rank_score(self, node):
        neighbors = self.neighbors[node]
        if self.consider_target_nodes == True:
            count = 0
            for neighbor in neighbors:
                if neighbor in self.target_nodes:
                    count += 1
        else:
            count = 0
            for neighbor in neighbors:
                if neighbor in self.sampled_nodes:
                    count += 1

        if self.consider_cost == True:
            sorted_rank_score = sorted(self.rank_score.itervalues())
            for node in self.rank_score.iterkeys():
                self.rank_score[node] = (sorted_rank_score.index(self.rank_score[node]) + 1) / math.pow(self.cost[node], self.c)
        else:
            self.rank_score[node] = count

    def time_to_update_community(self):
        if self.sample_mode == 1 or self.sample_mode == 5 or self.sample_mode == 6 or self.sample_mode == 7:
            if self.stage == 3:
                return True
        elif self.sample_mode == 2:
            if self.stage == 2:
                return True
        elif self.sample_mode == 3:
            if self.stage == 2:
                return True
        elif self.sample_mode == 4:
            if self.stage == 2:
                return True
        return False

    def update_community(self, update_global_flag=False):
        if self.community_detected == True:
            return self.current_network.update(update_global_flag)
        else:
            self.community_detected = True
            return self.community_detection()

    # added by Zhe, for MRF params tuning
    def update_community_MRF(self, beta, eta, update_global_flag=False):
        if self.community_detected == True:
            return self.current_network.update(update_global_flag)
        else:
            self.community_detected = True
            return self.community_detection_MRF(beta, eta)

    def community_detection(self):
        self.community_detected = True
        return self.current_network.community_detection(self.cluster_number)

    def community_detection_MRF(self, beta, eta):
        self.community_detected = True
        # Zhe: for tuning MRF models
        return self.current_network.community_detection_MRF(self.cluster_number, beta, eta)

    def spectral_clustering(self):
        return self.current_network.spectral_clustering(self.cluster_number)

    def erase_labels(self):
        self.current_network.erase_labels()

    def entropy_reduction(self, node):
        neighbors = self.current_network.neighbors(node)
        count = float(self.whole_network_size) / len(self.current_network.nodes()) * len(neighbors)
        count += len(self.current_network.nodes())
        #for neighbor in neighbors:
        #   if neighbor in self.target_nodes:
        #       count += 1
        return - 1 / count * self.current_network.entropy(node)

    def __str__(self):
        return str(self.current_network)

    def get_current_network(self):
        return self.current_network

    def network_info(self):
        components = self.detect_disconnected_components()
        self.network_disconnected_number = len(set((value) for value in components.itervalues()))
        return "Components: " + str(self.network_disconnected_number)

    def get_current_cost(self):
        return self.current_cost

    def get_nodes_size(self):
        return self.current_network.size()

    def initialize_label(self, label):
        self.current_network.initialize_label(dict(label))


if __name__ == "__main__":
    whole_network = network()
    whole_network.add_nodes([1, 2, 3, 4, 5])
    whole_network.add_edge((1, 2))
    whole_network.add_edge((1, 3))
    whole_network.add_edge((4, 5))
    whole_network.add_edge((2, 3))

    initial_network = network()
    initial_network.add_nodes([1, 2, 3])

    target_nodes = [1, 2, 3]

    cost = dict()
    for node in initial_network.nodes():
        cost[node] = 1

    budget = 3

    algorithm = DiscoverDetection(whole_network, initial_network, target_nodes, cost, budget, 2, 6)
    while True:
        try:
            node_to_sample = algorithm.sample_new_node()
            print 'sample node: ' + str(node_to_sample)
            communities = algorithm.community_detection()
        except Exception as e:
            print e.args
            print 'label:' + str(communities)
            print 'current network: ' + str(algorithm)
            break
