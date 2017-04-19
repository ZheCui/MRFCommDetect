import networkx as nx

def read_football(edgelist, community):
        """
                Only takes the vertices in the community list,
                since we don't know the ground truth not in the list
        """
        # Two files concatenated with ','

        # Create a iGraph with first file: Edge-list
        G=nx.read_edgelist(edgelist, nodetype=int, create_using=nx.Graph())

        # If community list exists, then parse it
        if True:
                # for first time of the dataset
                with open(community, 'r') as c:
                        cnt = 0
                        for l in c:
                                l = [int(x) for x in l.split()]
                                for v in l:
                                        G.node[v]['group'] = cnt
                                        G.node[v]['id'] = v
                                cnt += 1
                sv = []
                for n in G:
                        if 'id' not in G.node[n]:
                                sv.append(n)

                # sv = (n for n in G if 'id' in G.node[n])
                G.remove_nodes_from(sv)
                self_loop_nodes = G.nodes_with_selfloops()
                G.remove_nodes_from(self_loop_nodes)

        # Output file
        # print G.is_directed(), G.vcount(), G.ecount()
        return G

def community_remove(community_file, graph, removed_file):
        # write_back = open(removed_file, 'wb')
        with open(community_file, 'r') as c:
                for line in c:
                        line = [int(x) for x in line.split()]
                        removed_line = []
                        for v in line:
                                if graph.has_node(v):
                                        removed_line.append(v)
                        if len(removed_line) >= 1:
                                for item in removed_line:
                                        removed_file.write(str(item) + '\t')
                                removed_file.write('\n')



# edgelist and community are file names.
edgelist = 'amazon_ungraph.txt'
community = 'amazon_community.txt'
# edgelist = 'amazon_sampled_4'
# community = 'community_sampled_4'
G = read_football(edgelist, community)
iter = 19
while iter > 0:
        for i in range(4, -1, -1):
                sv = []
                for n in G:
                        if G.degree(n) == i:
                                sv.append(n)
                # sv = (n for n in G if G.degree(n) == i)
                G.remove_nodes_from(sv)
        iter -= 1
for i in range(1, -1, -1):
        sv = []
        for n in G:
                if G.degree(n) == i:
                        sv.append(n)
        # sv = (n for n in G if G.degree(n) == i)
        G.remove_nodes_from(sv)
# G.vs.select(_degree = 2).delete()
# G.vs.select(_degree = 1).delete()
# G.vs.select(_degree = 0).delete()
# G.vs.select(_degree = 1).delete()
# G.vs.select(_degree = 0).delete()

print G.number_of_nodes(), G.number_of_edges()

target_file = open("amazon_sampled_1", 'wb')
community_file = open('community_sampled_1', 'wb')
print 'writing reduced edge list into file'
nx.write_edgelist(G, target_file)
community_remove(community, G, community_file)
# for v in G:
#         if 'id' in G.node[v]:
#                 community_file.write(str(G.node[v]['id']) + '\t' + str(G.node[v]['group']) + '\n')
target_file.close()
community_file.close()
print 'done!'
        # print e.tuple[0], e.tuple[1]