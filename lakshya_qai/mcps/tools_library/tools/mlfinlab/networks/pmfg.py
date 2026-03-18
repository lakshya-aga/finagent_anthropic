"""
This file contains Graph classes, which create NetworkX's Graph objects from matrices.
"""

import heapq
import itertools
from itertools import count
import warnings

import networkx as nx
from matplotlib import pyplot as plt

from mlfinlab.networks.graph import Graph


class PMFG(Graph):
    """
    PMFG class creates and stores the PMFG as an attribute.
    """

    def __init__(self, input_matrix, matrix_type):
        """
        PMFG class creates the Planar Maximally Filtered Graph and stores it as an attribute.

        :param input_matrix: (pd.Dataframe) Input distance matrix
        :param matrix_type: (str) Matrix type name (e.g. "distance").
        """

        super().__init__(matrix_type)
        self.graph = self.create_pmfg(input_matrix)
        self.mst_edges = list(nx.minimum_spanning_tree(nx.from_pandas_adjacency(input_matrix)).edges())
        self.disparity = self._calculate_disparity()

    def get_disparity_measure(self):
        """
        Getter method for the dictionary of disparity measure values of cliques.

        :return: (Dict) Returns a dictionary of clique to the disparity measure.
        """

        return self.disparity

    def _calculate_disparity(self):
        """
        Calculate disparity given in Tumminello M, Aste T, Di Matteo T, Mantegna RN.
        A tool for filtering information in complex systems.
        https://arxiv.org/pdf/cond-mat/0501335.pdf

        :return: (Dict) Returns a dictionary of clique to the disparity measure.
        """

        disp = {}
        for cl in self._generate_cliques():
            sub = self.graph.subgraph(cl)
            weights = [d.get('weight', 1.0) for _, _, d in sub.edges(data=True)]
            s = sum(weights)
            disp[tuple(cl)] = sum((w / s) ** 2 for w in weights) if s > 0 else 0
        return disp

    def _generate_cliques(self):
        """
        Generate cliques from all of the nodes in the PMFG.
        """

        return list(nx.find_cliques(self.graph))

    def create_pmfg(self, input_matrix):
        """
        Creates the PMFG matrix from the input matrix of all edges.

        :param input_matrix: (pd.Dataframe) Input matrix with all edges
        :return: (nx.Graph) Output PMFG matrix
        """

        g = nx.from_pandas_adjacency(input_matrix)
        return g

    def get_mst_edges(self):
        """
        Returns the list of MST edges.

        :return: (list) Returns a list of tuples of edges.
        """

        return self.mst_edges

    def edge_in_mst(self, node1, node2):
        """
        Checks whether the edge from node1 to node2 is a part of the MST.

        :param node1: (str) Name of the first node in the edge.
        :param node2: (str) Name of the second node in the edge.
        :return: (bool) Returns true if the edge is in the MST. False otherwise.
        """

        return (node1, node2) in self.mst_edges or (node2, node1) in self.mst_edges

    def get_graph_plot(self):
        """
        Overrides parent get_graph_plot to plot it in a planar format.

        Returns the graph of the MST with labels.
        Assumes that the matrix contains stock names as headers.

        :return: (AxesSubplot) Axes with graph plot. Call plt.show() to display this graph.
        """

        pos = nx.planar_layout(self.graph) if nx.check_planarity(self.graph)[0] else nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=True)
        return plt.gca()
