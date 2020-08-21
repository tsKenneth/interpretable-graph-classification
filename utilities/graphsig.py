from utilities.GNNGraph import GNNGraph
import networkx as nx

def convert_graphsig_to_gnn_graph(sample_path, node_feature_exist=False):
    graph_list = []
    with open(sample_path, 'r') as f:
        filedata = f.read().splitlines()
        i = 0
        while(True):
            if filedata[i] != '' and filedata[i][0] == '#':
                feat_dict = {}
                node_tags = []
                node_tags_dict = {}
                node_features = []
                G = nx.Graph()
                i += 1
                no_of_nodes = int(filedata[i])
                for j in range(no_of_nodes):
                    G.add_node(j)
                    i += 1
                    # check if node is in feat dict
                    if not filedata[i] in feat_dict:
                        mapped = len(feat_dict)
                        feat_dict[filedata[i]] = mapped
                    node_tags.append(feat_dict[filedata[i]])
                    if feat_dict[filedata[i]] not in node_tags_dict.keys():
                        node_tags_dict[feat_dict[filedata[i]]] = filedata[i]

                i += 1
                no_of_edges = int(filedata[i])
                for j in range(no_of_edges):
                    i += 1
                    edge_data = filedata[i].split()
                    G.add_edge(edge_data[0], edge_data[1])
                graph_list.append(
                    GNNGraph(G, '', node_tags, node_tags_dict, node_features))
            else:
                i += 1
            if i >= len(filedata):
                break
    return graph_list
