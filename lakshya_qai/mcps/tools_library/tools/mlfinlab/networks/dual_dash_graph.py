"""
This class takes in a Graph object and creates interactive visualisations using Plotly's Dash.
The DualDashGraph class contains private functions used to generate the frontend components needed to create the UI.

Running run_server() will produce the warning "Warning: This is a development server. Do not use app.run_server
in production, use a production WSGI server like gunicorn instead.".
However, this is okay and the Dash server will run without a problem.
"""

import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
import dash_html_components as html
from dash import Dash
from dash.dependencies import Input, Output, State
from jupyter_dash import JupyterDash

class DualDashGraph:
    """
    The DualDashGraph class is the inerface for comparing and highlighting the difference between two graphs.
    Two Graph class objects should be supplied - such as MST and ALMST graphs.
    """

    def __init__(self, graph_one, graph_two, app_display='default'):
        """
        Initialises the dual graph interface and generates the interface layout.

        :param graph_one: (Graph) The first graph for the comparison interface.
        :param graph_two: (Graph) The second graph for the comparison interface.
        :param app_display: (str) 'default' by default and 'jupyter notebook' for running Dash inside Jupyter Notebook.
        """

        self.graph_one = graph_one
        self.graph_two = graph_two
        self.difference = graph_one.get_difference(graph_two)

        app_cls = JupyterDash if app_display == 'jupyter' else Dash
        self.app = app_cls(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

        self.elements_one = []
        self.elements_two = []
        self.stylesheet = self._get_default_stylesheet([])
        self._set_cyto_graph()
        self.app.layout = self._generate_comparison_layout(graph_one, graph_two)

        self.app.callback(
            Output("cytoscape-two", "elements"),
            Input("cytoscape-one", "tapNodeData"),
            State("cytoscape-two", "elements"),
        )(self._select_other_graph_node)
        self.app.callback(
            Output("cytoscape-one", "elements"),
            Input("cytoscape-two", "tapNodeData"),
            State("cytoscape-one", "elements"),
        )(self._select_other_graph_node)

    @staticmethod
    def _select_other_graph_node(data, elements):
        """
        Callback function to select the other graph node when a graph node
        is selected by setting selected to True.

        :param data: (Dict) Dictionary of "tapped" or selected node.
        :param elements: (Dict) Dictionary of elements.
        :return: (Dict) Returns updates dictionary of elements.
        """

        if data is None:
            return elements
        node_id = data.get("id")
        for elem in elements:
            if "source" in elem.get("data", {}):
                continue
            elem["selected"] = elem["data"].get("id") == node_id
        return elements

    def _generate_comparison_layout(self, graph_one, graph_two):
        """
        Returns and generates a dual comparison layout.

        :param graph_one: (Graph) The first graph object for the dual interface.
        :param graph_two: (Graph) Comparison graph object for the dual interface.
        :return: (html.Div) Returns a Div containing the interface.
        """

        return dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            cyto.Cytoscape(
                                id="cytoscape-one",
                                elements=self.elements_one,
                                layout={"name": "cose"},
                                stylesheet=self.stylesheet,
                                style={"width": "100%", "height": "650px"},
                            ),
                            width=6,
                        ),
                        dbc.Col(
                            cyto.Cytoscape(
                                id="cytoscape-two",
                                elements=self.elements_two,
                                layout={"name": "cose"},
                                stylesheet=self.stylesheet,
                                style={"width": "100%", "height": "650px"},
                            ),
                            width=6,
                        ),
                    ]
                )
            ],
            fluid=True,
        )

    @staticmethod
    def _get_default_stylesheet(weights):
        """
        Returns the default stylesheet for initialisation.

        :param weights: (List) A list of weights of the edges.
        :return: (List) A List of definitions used for Dash styling.
        """

        return [
            {
                "selector": "node",
                "style": {
                    "label": "data(label)",
                    "background-color": "#2a9d8f",
                    "font-size": "10px",
                    "color": "#222",
                },
            },
            {
                "selector": "edge",
                "style": {
                    "label": "data(weight)",
                    "line-color": "#888",
                    "width": 2,
                    "curve-style": "bezier",
                    "font-size": "8px",
                },
            },
            {
                "selector": ".diff-edge",
                "style": {"line-color": "#e76f51", "width": 4},
            },
        ]

    def _set_cyto_graph(self):
        """
        Updates and sets the two cytoscape graphs using the corresponding components.
        """

        self.elements_one = self._update_elements_dual(self.graph_one, self.difference, 1)
        self.elements_two = self._update_elements_dual(self.graph_two, self.difference, 2)

    def _update_elements_dual(self, graph, difference, graph_number):
        """
        Updates the elements needed for the Dash Cytoscape Graph object.

        :param graph: (Graph) Graph object such as MST or ALMST.
        :param difference: (List) List of edges where the two graphs differ.
        :param graph_number: (Int) Graph number to update the correct graph.
        """

        g = graph.get_graph()
        if g is None:
            return []
        diff_edges = set(tuple(sorted(e)) for e in difference)
        elements = []
        for node in g.nodes():
            elements.append({
                "data": {"id": str(node), "label": str(node)}
            })
        for u, v, data in g.edges(data=True):
            weight = float(data.get("weight", 1.0))
            classes = "diff-edge" if tuple(sorted((u, v))) in diff_edges else ""
            elements.append({
                "data": {"source": str(u), "target": str(v), "weight": weight},
                "classes": classes,
            })
        return elements

    def get_server(self):
        """
        Returns the comparison interface server

        :return: (Dash) Returns the Dash app object, which can be run using run_server.
            Returns a Jupyter Dash object if DashGraph has been initialised for Jupyter Notebook.
        """

        return self.app
