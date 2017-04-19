import networkx as nx

def read_graph(edgelist, community):
        # Create a networkx with first file: Edge-list
        G=nx.read_edgelist(edgelist, nodetype=int, create_using=nx.Graph())
        print G.number_of_nodes(), G.number_of_edges()
        for n in G:
                G.node[n]['groups'] = 0
        with open(community, 'r') as c:
                cnt = 1
                for l in c:
                        l = [int(x) for x in l.split()]
                        for v in l:
                                G.node[v]['groups'] += 1
                                G.node[v]['community'] = cnt
                                cnt += 1

        return G

def write_community(graph, save_file):
        # write_back = open(removed_file, 'wb')
        for node in graph:
                if graph.node[node]['groups'] == 1:
                    save_file.write(str(node) + '\t' + \
                    	str(graph.node[node]['community']) + '\n')


ordered_list = "amazon_consecutive"
ordered_community = 'community_consecutive'

G = read_graph(ordered_list, ordered_community)

save_file = open('community_samples', 'wb')
write_community(G, save_file)
save_file.close()


