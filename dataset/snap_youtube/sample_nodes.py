from CommunityDetection import network
import random
label = dict()

# sample target set of nodes
def nodes():
	return whole_network.nodes()

def size():
	return whole_network.size()

def neighbors(node):
	return whole_network.neighbors(node)

whole_network = network()
# synthetic dataset
# whole_network.add_nodes(range(1, 36001))
# coauthor_file = open("dataset/binary_networks/network_N36000_k15_maxk50_mu0.3_on0_om0.dat", 'rb')

# coauthorship dataset
whole_network.add_nodes(range(0, 30123))
coauthor_file = open("dataset/youtube/youtube_consecutive", 'rb')

for line in coauthor_file:
	elements = line.split('\t')
	# if int(elements[0]) < int(elements[1]):
	whole_network.add_edge((int(elements[0]), int(elements[1])))
coauthor_file.close()
print "data loaded!"

# coauthorship community file
community_file = open("dataset/youtube/youtube_community_consecutive", 'rb')
community = 1
for line in community_file:
    elements = line.split('\t')
    elements.pop()
    for element in elements:
    	if element != ' ' and int(element) not in label:
    		label[int(element)] = community
    community += 1
print 'community info loaded!'

#synthetic community file
# community_file = open("dataset/binary_networks/community_N36000_k15_maxk50_mu0.3_on0_om0.dat", 'rb')
# for line in community_file:
#     elements = line.split('\t')
#     label[int(elements[0])] = int(elements[1])
# print 'label generated!'

def sampleNodesFromOneCommunity(startNode, targetNum, communities, target_community):
	sampled_nodes = list()
	candidate_nodes = list()
	candidate_nodes = neighbors(startNode)[:]
	curr_communities = list()
	curr_communities.append(communities)
	# if len(candidate_nodes) < 10:
	# 	return sampled_nodes
	sampled_nodes.append(startNode)
	# print sampled_nodes
	# print neighbors(startNode)[:]
	pre_candidate_size = 0
	while len(sampled_nodes) < targetNum * target_community:
		i = 0
		sampleNo = 0
		random.shuffle(candidate_nodes)
		# if pre_candidate_size >= candidate_size:
		# 	break
		while len(candidate_nodes) > 0:
			curr_node = candidate_nodes[0]
			del candidate_nodes[0]
			if len(curr_communities) <= target_community:
				if label[curr_node] not in curr_communities:
					curr_communities.append(label[curr_node])
			if curr_node not in sampled_nodes and label[curr_node] in curr_communities:
				print 'label is ' + str(label[curr_node])

				sampled_nodes.append(curr_node)
				sampleNo += 1
				if sampleNo >= target_community * targetNum:
					break
				candidate_nodes.extend(whole_network.neighbors(curr_node))
			i += 1
		# pre_candidate_size = len(candidate_nodes)
		return sampled_nodes, curr_communities

# synthetic dataset
# community_num = range(1, 66)
# community_sampled = 10
# done = True
# nodespercommunity = 70
# shuffled_nodes = range(1, 36001)
# target_file = open("dataset/binary_networks/target_community_N36000_k15_maxk50_mu0.3_on0_om0.dat", 'wb')


# coauthorship dataset
community_num = range(1, community)
community_sampled = 20
done = True
nodespercommunity = 30
shuffled_nodes = range(1, 30123)
target_file = open("dataset/youtube/target_community", 'wb')
while done:
	rand_community = random.sample(community_num, 1)
	target_nodes = list()
	random.shuffle(shuffled_nodes)
	print rand_community
	for node in shuffled_nodes:
		if node in label and label[node] in rand_community:
			nodes, sampled_communities = sampleNodesFromOneCommunity(node, nodespercommunity, rand_community, community_sampled)
			# nodes = results['nodes']
			# sampled_communities = results['community']
			if len(nodes) >= nodespercommunity:
				for each in nodes:
					if each not in target_nodes:
						target_nodes.append(each)
			if len(nodes) >= nodespercommunity * community_sampled:
				for node in target_nodes:
					# if label[node] not in fix_community:
					# 	print label[node]
					# if label[node] in fix_community:
					target_file.write(str(node) + "\t" + str(label[node]) + "\n")
				done = False
				print 'sampled nodes generation done!'
				break
