"""
This file contains Graph classes, which create NetworkX's Graph objects from matrices.
"""

import networkx as nx
from mlfinlab.networks.graph import Graph


class MST(Graph):
    """
    MST is a subclass of Graph which creates a MST Graph object.
    """

    def __init__(self, matrix, matrix_type, mst_algorithm='kruskal'):
        """
        Creates a MST Graph object and stores the MST inside graph attribute.

        :param matrix: (pd.Dataframe) Input matrices such as a distance or correlation matrix.
        :param matrix_type: (str) Name of the matrix type (e.g. "distance" or "correlation").
        :param mst_algorithm: (str) Valid MST algorithm types include 'kruskal', 'prim', or 'boruvka'.
            By default, MST algorithm uses Kruskal's.
        """

        super().__init__(matrix_type)
        self.graph = self.create_mst(matrix, mst_algorithm)

    @staticmethod
    def create_mst(matrix, algorithm='kruskal'):
        """
        This method converts the input matrix into a MST graph.

        :param matrix: (pd.Dataframe) Input matrix.
        :param algorithm: (str) Valid MST algorithm types include 'kruskal', 'prim', or 'boruvka'.
            By default, MST algorithm uses Kruskal's.
        """

        g = nx.from_pandas_adjacency(matrix)
        return nx.minimum_spanning_tree(g, algorithm=algorithm)
