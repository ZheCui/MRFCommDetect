from CommunityDetection import network
from jsonrpclib.SimpleJSONRPCServer import SimpleJSONRPCServer

def nodes():
	return whole_network.nodes()

def size():
	return whole_network.size()

def neighbors(node):
	return whole_network.neighbors(node)

whole_network = network()
# synthetic dataset
whole_network.add_nodes(range(1, 36001))
coauthor_file = open("dataset/binary_networks/network_N36000_k8_maxk12_mu0.3.dat", 'rb')
for line in coauthor_file:
	elements = line.split('\t')
	if int(elements[0]) < int(elements[1]):
	    whole_network.add_edge((int(elements[0]), int(elements[1])))
coauthor_file.close()
print "synthetic data loaded!"

server = SimpleJSONRPCServer (("127.0.0.1", 9000))
server.register_function (nodes)
server.register_function (size)
server.register_function (neighbors)
try:
    print "Start working"
    server.serve_forever()
except KeyboardInterrupt:
    pass

