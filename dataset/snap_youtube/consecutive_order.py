import networkx as nx

def read_graph(edgelist):
        # Create a networkx with first file: Edge-list
        G=nx.read_edgelist(edgelist, nodetype=int, create_using=nx.Graph())
        ind = 1
        print G.number_of_nodes(), G.number_of_edges()
        node_index = dict()

        for node in G:
                node_index[node] = ind
                ind += 1

        return G, node_index

def write_edgelist(edgelist, graph, new_edgelist, node_index):
        # write_back = open(removed_file, 'wb')
        with open(edgelist, 'r') as c:
                for line in c:
                        line = [int(x) for x in line.split()]
                        removed_line = []
                        for v in line:
                                removed_line.append(node_index[v])
                        for item in removed_line:
                                new_edgelist.write(str(item) + '\t')
                        new_edgelist.write('\n')

# edgelist and community are file names.

# edgelist = open("amazon_sampled_1", 'r')
edgelist = 'youtube_sampled_1'
community_file = 'youtube_community_1'
# community_file = open('community_sampled_1', 'r')

ordered_list = open("youtube_consecutive", 'wb')
ordered_community = open('youtube_community_consecutive', 'wb')

G, node_index = read_graph(edgelist)

print 'writing into file'
write_edgelist(edgelist, G, ordered_list, node_index)
write_edgelist(community_file, G, ordered_community, node_index)
print 'done!'

ordered_list.close()
ordered_community.close()
# edgelist.close()
# community_file.close()