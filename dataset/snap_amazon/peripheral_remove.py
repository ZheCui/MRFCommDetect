import igraph as ig

def read_football(edgelist, community):
        """
                Only takes the vertices in the community list,
                since we don't know the ground truth not in the list
        """
        # Two files concatenated with ','

        # Create a iGraph with first file: Edge-list
        G = ig.Graph.Read_Edgelist(edgelist, directed=False)

        # If community list exists, then parse it
        if True:
                with open(community, 'r') as c:
                        cnt = 0
                        for l in c:
                                l = [int(x) for x in l.split()]
                                for v in l:
                                        G.vs[v]['group'] = cnt
                                        G.vs[v]['id'] = v
                                cnt += 1
                sv = G.vs.select(id_eq=None)
                G.delete_vertices(sv)

        # Output file
        # print G.is_directed(), G.vcount(), G.ecount()
        return G.simplify(combine_edges="first")

def community_remove(community_file, graph, removed_file):
        # write_back = open(removed_file, 'wb')
        with open(community_file, 'r') as c:
                for line in c:
                        line = [int(x) for x in line.split()]
                        removed_line = []
                        for v in line:
                                if graph.vs.select(id_eq = v) != None:
                                        removed_line.append(v)
                        if len(removed_line) >= 1:
                                for item in removed_line:
                                        removed_file.write(str(item) + '\t')
                                removed_file.write('\n')

# edgelist and community are file names.
edgelist = 'amazon_ungraph.txt'
community = 'amazon_community.txt'
G = read_football(edgelist, community).as_undirected()
iter = 20
while iter > 0:
        G.vs.select(_degree = 4).delete()
        G.vs.select(_degree = 3).delete()
        G.vs.select(_degree = 2).delete()
        G.vs.select(_degree = 1).delete()
        G.vs.select(_degree = 0).delete()
        iter -= 1

# iter = 20
# while iter > 0:
#         G.vs.select(_degree = 3).delete()
#         G.vs.select(_degree = 2).delete()
#         G.vs.select(_degree = 1).delete()
#         G.vs.select(_degree = 0).delete()
#         iter -= 1

print G.is_directed(), G.vcount(), G.ecount()
target_file = open("amazon_sampled_1", 'wb')
community_file = open('community_sampled_1', 'wb')
print 'writing reduced edge list into file'
for e in G.es:
        target_file.write(str(e.tuple[0]) + "\t" + str(e.tuple[1]) + "\n")
community_remove(community, G, community_file)
# for v in G.vs:
#         community_file.write(str(v.index) + '\t' + str(v['group']) + '\n')
target_file.close()
community_file.close()
print 'done!'
        # print e.tuple[0], e.tuple[1]