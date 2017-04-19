import jsonrpclib

target_nodes = list()
label = dict()

# synthetic dataset
target_file = open("dataset/binary_networks/target_community_N36000_k8_maxk12_mu0.3.dat", 'rb')
for line in target_file:
    elements = line.split('\t')
    label[int(elements[0])] = int(elements[1])
    target_nodes.append(int(elements[0]))

cluster_num = 10
print 'number of target nodes is ' + str(len(target_nodes))
print 'number of cluster for synthetic dataset is ' + str(cluster_num)
endpoint = "http://127.0.0.1:9000/"

if __name__ == '__main__':
    whole_network = jsonrpclib.Server(endpoint)
    nodes = whole_network.nodes()
    print str(len(nodes))
