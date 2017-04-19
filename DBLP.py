import jsonrpclib

target_nodes = list()
label = dict()
# DBLP Dataset
target_file = open("dataset/DBLP/label", 'rb')
for line in target_file:
    elements = line.split('\t')
    label[int(elements[0])] = int(elements[1])
    target_nodes.append(int(elements[0]))

cluster_num = 4
endpoint = "http://127.0.0.1:8000/"



if __name__ == '__main__':
    whole_network = jsonrpclib.Server(endpoint)
    nodes = whole_network.nodes()
    print str(len(nodes))
