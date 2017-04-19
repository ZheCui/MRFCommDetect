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
whole_network.add_nodes(range(1, 30123))
coauthor_file = open("dataset/youtube/youtube_consecutive", 'rb')
for line in coauthor_file:
	elements = line.split('\t')
	whole_network.add_edge((int(elements[0]), int(elements[1])))
coauthor_file.close()
print "youtube data loaded!"

server = SimpleJSONRPCServer (("127.0.0.1", 9000))
server.register_function (nodes)
server.register_function (size)
server.register_function (neighbors)
try:
    print "Start working"
    server.serve_forever()
except KeyboardInterrupt:
    pass

