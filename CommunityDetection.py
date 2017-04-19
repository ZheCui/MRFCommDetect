from pygraph.classes.graph import graph
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import eigs
from scipy.cluster.vq import *
import numpy
import random
import math
import copy

class network(object):
    def __init__(self):
        self.network = graph()
        self.qij = dict()
        self.thetaiz = dict()
        self.taoiz = dict()
        self.loglikelihood = list()
        self.nodes_size = 0
        self.edges_size = 0
        self.new_edges_size = 0
        self.pruned_edges = set()
        self.pure_nodes = set()
        self.new_nodes = set()
        self.new_edges = set()

    def graph(self):
        return self.network

    def community_detection(self, cluster_num, converge_ratio=1e-7):
        self.cluster_num = cluster_num
        print "Block model run."
        self.pruned_edges = set(self.network.edges())
        self.pure_nodes = set()
        self.qij = dict()
        if 'label' not in self.__dict__:
            self.label = dict()
            self.initialize_qij(self.pruned_edges, True, 0.1)
        else:
            self.initialize_qij(self.pruned_edges, False, 0.1)
        iter_num = 50
        while True:
            iter_num -= 1
            if iter_num == 0:
                break
            if iter_num > 10:
                self.update_thetaiz(False)
            else:
                self.update_thetaiz(False)
            self.update_qij(self.pruned_edges)
            print "All edges size: " + str(len(self.network.edges())) + " Pruned edges size: " + str(len(self.pruned_edges))

        self.update_label(self.network.nodes())

        self.new_edges_size = 0
        self.new_nodes.clear()
        self.new_edges.clear()
        return self.label

    # added by Zhe for MRF tuning
    def community_detection_MRF(self, cluster_num, beta, eta, converge_ratio=1e-7):
        self.cluster_num = cluster_num
        print "Block model run."
        self.pruned_edges = set(self.network.edges())
        self.pure_nodes = set()
        self.qij = dict()
        if 'label' not in self.__dict__:
            self.label = dict()
            self.initialize_qij(self.pruned_edges, True, 0.1)
        else:
            self.initialize_qij(self.pruned_edges, False, 0.1)
        iter_num = 50
        while True:
            iter_num -= 1
            if iter_num == 0:
                break
            if iter_num > 10:
                self.update_thetaiz(False)
            else:
                self.update_thetaiz(False)
            self.update_qij(self.pruned_edges)
            print "All edges size: " + str(len(self.network.edges())) + " Pruned edges size: " + str(len(self.pruned_edges))

        self.update_label(self.network.nodes())
        # added by Zhe for MRF
        # self.update_label_MRF(self.network.nodes(), beta, eta)
        self.new_edges_size = 0
        self.new_nodes.clear()
        self.new_edges.clear()
        return self.label

    def update_global(self, clear_qij_flag=False):
        if clear_qij_flag:
            backup_qij = dict(self.qij)
            self.qij.clear()
            iter_num = 50
        else:
            iter_num = 15
        self.pruned_edges = set(self.network.edges())
        self.pure_nodes = set()
        self.initialize_qij(self.pruned_edges, False)
        threshold = iter_num * 0.7
        while True:
            iter_num -= 1
            if iter_num == 0:
                break
            if iter_num > threshold:
                self.update_thetaiz(False)
            else:
                self.update_thetaiz(True)
            self.update_qij(self.pruned_edges)
        self.new_edges_size = 0
        print str(len(self.network.edges())) + " " + str(len(self.pruned_edges))
        if clear_qij_flag:
            self.qij = backup_qij

    def update_label(self, nodes):
        self.label_count = [0] * self.cluster_num
        for node in nodes:
            if node in self.thetaiz:
                self.label[node] = self.thetaiz[node].index(max(self.thetaiz[node]))
            else:
                self.label[node] = random.choice(xrange(self.cluster_num))
            self.label_count[self.label[node]] += 1

    # added by Zhe
    def update_label_MRF(self, nodes, beta, eta):
        observe_label = copy.deepcopy(self.label)
        # number of iterations
        iterations = 50
        # parameters of energy functions
        # beta = 1.0
        # eta = 15
        while iterations >= 0:
            iterations -= 1
            changes = 0
            for node in observe_label:
                min_cost = float('inf')
                curr_label = observe_label[node]
                new_label = curr_label
                neighbors = self.network.neighbors(node)
                # for each label compute energy function and find the minimum one
                # temp_label is array index + 1
                for temp_label in range(1, self.cluster_num + 1):
                    curr_cost = 0
                    print 'min cost is ' + str(min_cost)
                    # curr_cost += -beta * numpy.sum([observe_label[neighbor] for neighbor in neighbors])
                    curr_cost += -beta * numpy.sum([min(1, abs(temp_label - \
                        observe_label[neighbor])) for neighbor in neighbors])
                    print numpy.sum([min(1, abs(temp_label - \
                        observe_label[neighbor])) for neighbor in neighbors])
                    curr_cost += -eta * min(1, abs(temp_label - curr_label))
                    print min(1, abs(temp_label - curr_label))
                    if curr_cost < min_cost:
                        new_label = temp_label
                        min_cost = curr_cost

                if new_label != curr_label:
                    observe_label[node] = new_label
                    print "label changed!"
                    print node, curr_label, new_label
                    changes += 1
            print 'changes number are ' + str(changes)
            if changes < 20:
                break

        print 'cluster num is ' + str(self.cluster_num)
        print observe_label
        self.label = observe_label

    def update(self, final_update_flag=False):
        '''
        self.initialize_qij(self.new_edges, False)
        self.update_global(final_update_flag)
        self.update_label(self.network.nodes())
        self.new_nodes.clear()
        self.new_edges.clear()
        '''
        self.initialize_qij(self.new_edges, False)
        edges_for_update_thetaiz = set()
        nodes_to_update = set()
        for seed in self.new_nodes:
            search_result = self.breadth_first_search(seed, 1)
            nodes = search_result[0]
            nodes_to_update = nodes_to_update.union(nodes)
            edges = search_result[1]
            blocked_edges = set()
            tmp_edges_for_update_thetaiz = set(edges)
            last_added_nodes = search_result[2]
            for edge in edges:
                if edge[0] not in nodes or \
                        (edge[0] in last_added_nodes and edge[1] in last_added_nodes):
                    blocked_edges.add(edge)
            for edge in blocked_edges:
                #if edge in tmp_edges_for_update_qij:
                #   tmp_edges_for_update_qij.remove(edge)
                #if (edge[1], edge[0]) in tmp_edges_for_update_qij:
                #   tmp_edges_for_update_qij.remove((edge[1], edge[0]))
                tmp_edges_for_update_thetaiz.remove(edge)
            #edges_for_update_qij = edges_for_update_qij.union(tmp_edges_for_update_qij)
            edges_for_update_thetaiz = edges_for_update_thetaiz.union(tmp_edges_for_update_thetaiz)
        self.update_thetaiz_local(edges_for_update_thetaiz)
        #while True:
        #   iter_num -= 1
        #   if iter_num == 0:
        #       break
            #self.update_qij(edges_for_update_qij)
        #   self.update_thetaiz_local(edges_for_update_thetaiz)

        #print str(len(self.network.edges())) + " " + str(len(self.pruned_edges))
        #self.update_global(False)

        if self.new_edges_size > 0.05 * self.edges_size or final_update_flag == True:
            if final_update_flag == True:
                self.update_global(True)
            else:
                self.update_global(False)
            nodes = self.network.nodes()
        self.update_label(self.network.nodes())  # nodes_to_update)

        self.new_nodes.clear()
        self.new_edges.clear()

        return self.label

    #@profile
    def initialize_qij(self, edges, spectral_clustering=True, bias=0.1):
        if spectral_clustering:
            print "run spectral clustering to initialize block model"
            self.label = self.spectral_clustering(self.cluster_num)
        for edge in edges:
            if edge in self.qij:
                continue
            if (edge[1], edge[0]) in self.qij:
                self.qij[edge] = self.qij[(edge[1], edge[0])]
            else:
                self.qij[edge] = [0] * self.cluster_num
                #tmp = [random.random() for i in xrange(self.cluster_num)]
                if edge[0] in self.label:
                    self.qij[edge][self.label[edge[0]]] += bias
                    #self.qij[edge] = [i / (sum(tmp) / (0.5 - bias)) for i in tmp]
                    self.qij[edge] = [i + j for i, j in zip(self.qij[edge], [(0.5 - bias) / self.cluster_num] * self.cluster_num)]
                else:
                    #self.qij[edge] = [i / (sum(tmp) / 0.5) for i in tmp]
                    self.qij[edge] = [i + j for i, j in zip(self.qij[edge], [0.5 / self.cluster_num] * self.cluster_num)]
                #tmp = [random.random() for i in xrange(self.cluster_num)]
                if edge[1] in self.label:
                    self.qij[edge][self.label[edge[1]]] += bias
                    #self.qij[edge] = [i / (sum(tmp) / (0.5 - bias)) for i in tmp]
                    self.qij[edge] = [i + j for i, j in zip(self.qij[edge], [(0.5 - bias) / self.cluster_num] * self.cluster_num)]
                else:
                    #self.qij[edge] = [i / (sum(tmp) / 0.5) for i in tmp]
                    self.qij[edge] = [i + j for i, j in zip(self.qij[edge], [0.5 / self.cluster_num] * self.cluster_num)]
                #random_prob = [random.randint(0,1000) for r in xrange(self.cluster_num)]
                #self.qij[edge] = [float(i) / sum(random_prob) for i in random_prob]

    def update_thetaiz_local(self, edges):
        nodes = set()
        for edge in edges:
            nodes.add(edge[0])
        normalize_part = self.normalize_part
        for node in self.new_nodes:
            self.taoiz[node] = [0] * self.cluster_num
        for node in nodes:
            if node in self.taoiz:
                normalize_part = [i - j for i, j in zip(normalize_part, self.taoiz[node])]
            self.taoiz[node] = [0] * self.cluster_num

        for edge in edges:
            self.taoiz[edge[0]] = [i + j for i, j in zip(self.taoiz[edge[0]], self.qij[edge])]
            normalize_part = [i + j for i, j in zip(normalize_part, self.qij[edge])]

        #print normalize_part
        self.pruned_nodes = self.pruned_nodes.union(nodes)
        for node in self.pruned_nodes:
            if node in self.taoiz:
                self.thetaiz[node] = [i / (math.sqrt(j) + 0.000001) for i, j in zip(self.taoiz[node], normalize_part)]

    #@profile
    def update_thetaiz(self, prune=False):
        edges = self.pruned_edges
        self.pruned_nodes = set()
        for edge in edges:
            self.pruned_nodes.add(edge[0])
        for node in self.pruned_nodes:
            self.taoiz[node] = [0] * self.cluster_num
        self.normalize_part = [0] * self.cluster_num

        for edge in edges:
            self.taoiz[edge[0]] = [i + j for i, j in zip(self.taoiz[edge[0]], self.qij[edge])]

        if prune == True:
            self.prune_edges(self.pruned_nodes)

        for edge in edges:
            self.normalize_part = [i + j for i, j in zip(self.normalize_part, self.qij[edge])]

        #print normalize_part
        for node in self.pruned_nodes:
            self.thetaiz[node] = [i / (math.sqrt(j) + 0.0001) for i, j in zip(self.taoiz[node], self.normalize_part)]

    #@profile
    def update_qij(self, edges):
        for edge in edges:
            q = self.qij[edge]
            for cluster_ind in xrange(self.cluster_num):
                q[cluster_ind] = self.thetaiz[edge[0]][cluster_ind] * self.thetaiz[edge[1]][cluster_ind]

            normalize_part = sum(q)
            if normalize_part != 0:
                for cluster_ind in xrange(self.cluster_num):
                    q[cluster_ind] /= normalize_part

    def prune_edges(self, involved_nodes):
        candidate_nodes = set()
        for node in involved_nodes:
            count = 0
            for cluster in xrange(self.cluster_num):
                if self.taoiz[node][cluster] < 0.001:
                    self.taoiz[node][cluster] = 0
                    count += 1
            if count == self.cluster_num - 1:
                candidate_nodes.add(node)
        for node in candidate_nodes:
            for neighbor in self.network.neighbors(node):
                if neighbor in self.pure_nodes or neighbor in candidate_nodes:
                    if (node, neighbor) in self.pruned_edges:
                        self.pruned_edges.remove((node, neighbor))
                        self.pruned_edges.remove((neighbor, node))

        self.pure_nodes = self.pure_nodes.union(candidate_nodes)

    def loglikelihood_value(self):
        edges = self.network.edges()
        loglikelihood = 0
        for edge in edges:
            node1 = edge[0]
            node2 = edge[1]
            tmp = 0
            for cluster_ind in xrange(self.cluster_num):
                tmp += self.thetaiz[node1][cluster_ind] * self.thetaiz[node2][cluster_ind]
            if tmp > 0:
                loglikelihood += math.log(tmp)  # self.network.edge_weight((node1, node2))
        return loglikelihood

    def spectral_clustering(self, cluster_num):
        W = self.graph_to_matrix()
        tmp = numpy.divide(1, numpy.sqrt(W.sum(1)))
        #tmp = W.sum(1)
        for n in xrange(len(tmp)):
            if tmp[n] == numpy.inf:
                tmp[n] = 0
        D = lil_matrix((self.nodes_size, self.nodes_size))
        D.setdiag(tmp)
        L = D * W * D
        lam, u = eigs(L, k=cluster_num, which='LR')
        '''
        for i in xrange(u.shape[0]):
            tmp = numpy.linalg.norm(u[i, :])
            u[i, :] = [j / tmp for j in u[i, :]]
        '''
        #convert u type to float32
        centroid, variance = kmeans(u.astype(numpy.float32), cluster_num, iter=100)
        label, distance = vq(u, centroid)
        self.label = dict()
        for node in self.network.nodes():
            self.label[node] = label[self.nodes_to_matrix_index[node]]
        return self.label

    def graph_to_matrix(self):
        self.nodes_to_matrix_index = dict()
        nodes = self.network.nodes()
        count = 0
        for node in nodes:
            self.nodes_to_matrix_index[node] = count
            count += 1
        W = lil_matrix((self.nodes_size, self.nodes_size))
        for edge in self.network.edges():
            W[self.nodes_to_matrix_index[edge[0]], self.nodes_to_matrix_index[edge[1]]] = 1
            W[self.nodes_to_matrix_index[edge[1]], self.nodes_to_matrix_index[edge[0]]] = 1

        return W

    def breadth_first_search(self, seed, hop=2):
        nodes = set()
        edges = set()
        last_added_nodes = set()
        pool = set([seed])
        for k in xrange(hop):
            tmp_pool = set()
            for candidate in pool:
                if k == hop - 1 and candidate not in nodes:
                    last_added_nodes.add(candidate)
                nodes.add(candidate)
                neighbors = self.network.neighbors(candidate)
                for neighbor in neighbors:
                    tmp_pool.add(neighbor)
                    edges.add((candidate, neighbor))
                    edges.add((neighbor, candidate))
            pool = set(tmp_pool)
        return [nodes, edges, last_added_nodes]

    def add_nodes(self, nodes):
        self.network.add_nodes(nodes)
        self.nodes_size += len(nodes)

    def add_node(self, node):
        self.network.add_node(node)
        self.new_nodes.add(node)
        self.nodes_size += 1

    def add_edge(self, edge, wt=1):
        self.network.add_edge(edge, wt)
        self.edges_size += 2
        self.new_edges_size += 2
        self.new_edges.add(edge)
        self.new_edges.add((edge[1], edge[0]))

    def initialize_label(self, label):
        self.label = label

    def get_label_count(self):
        return self.label_count

    def get_label(self):
        return self.label

    def get_qij(self):
        return self.qij

    def get_thetaiz(self):
        return self.thetaiz

    def neighbors(self, node):
        return self.network.neighbors(node)

    def has_node(self, node):
        return self.network.has_node(node)

    def edges(self):
        return self.network.edges()

    def has_edge(self, edge):
        return self.network.has_edge(edge)

    def nodes(self):
        return self.network.nodes()

    def __str__(self):
        return str(self.network)

    def size(self):
        return self.nodes_size

if __name__ == "__main__":
    network = network()
    network.add_nodes([1, 2, 3, 4, 5])
    network.add_edge((1, 2), 3)
    network.add_edge((1, 3))
    network.add_edge((4, 5))
    label = network.community_detection()
    print label
