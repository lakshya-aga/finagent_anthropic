"""
This class takes in a Graph object and creates interactive visualisations using Plotly's Dash.
The DashGraph class contains private functions used to generate the frontend components needed to create the UI.

Running run_server() will produce the warning "Warning: This is a development server. Do not use app.run_server
in production, use a production WSGI server like gunicorn instead.".
However, this is okay and the Dash server will run without a problem.
"""

import json
import random

import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_cytoscape as cyto
import dash_html_components as html
from dash import Dash
from dash.dependencies import Input, Output
from jupyter_dash import JupyterDash
import networkx as nx


class DashGraph:
    """
    This DashGraph class creates a server for Dash cytoscape visualisations.
    """

    def __init__(self, input_graph, app_display='default'):
        """
        Initialises the DashGraph object from the Graph class object.
        Dash creates a mini Flask server to visualise the graphs.

        :param app_display: (str) 'default' by default and 'jupyter notebook' for running Dash inside Jupyter Notebook.
        :param input_graph: (Graph) Graph class from graph.py.
        """

        self.input_graph = input_graph
        self.graph = input_graph.get_graph()
        self.pos = input_graph.get_pos()
        self.node_groups = input_graph.get_node_colours() or {}
        self.node_sizes = input_graph.get_node_sizes()
        self.node_group_lookup = {}
        nodes = list(self.graph.nodes()) if self.graph is not None else []
        for group, members in self.node_groups.items():
            for m in members:
                if isinstance(m, int) and 0 <= m < len(nodes):
                    self.node_group_lookup[nodes[m]] = group
                else:
                    self.node_group_lookup[m] = group

        app_cls = JupyterDash if app_display == 'jupyter' else Dash
        self.app = app_cls(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

        self.elements = []
        self.stylesheet = self._get_default_stylesheet()
        self.group_colours = {}
        if self.node_groups:
            self._assign_colours_to_groups(list(self.node_groups.keys()))
            self._style_colours()
        if self.node_sizes is not None:
            self._assign_sizes()

        self._set_cyto_graph()
        self.app.layout = self._generate_layout()

        self.app.callback(Output("cytoscape", "layout"), Input("layout-dropdown", "value"))(
            self._update_cytoscape_layout
        )
        self.app.callback(Output("cytoscape", "elements"), Input("decimal-slider", "value"))(
            self._round_decimals
        )
        self.app.callback(Output("stat-json", "children"), Input("stat-dropdown", "value"))(
            self._update_stat_json
        )

    def _set_cyto_graph(self):
        """
        Sets the cytoscape graph elements.
        """

        self._update_elements()

    def _get_node_group(self, node_name):
        """
        Returns the industry or sector name for a given node name.

        :param node_name: (str) Name of a given node in the graph.
        :return: (str) Name of industry that the node is in or "default" for nodes which haven't been assigned a group.
        """

        return self.node_group_lookup.get(node_name, "default")

    def _get_node_size(self, index):
        """
        Returns the node size for given node index if the node sizes have been set.

        :param index: (int) The index of the node.
        :return: (float) Returns size of node set, 0 if it has not been set.
        """

        if self.node_sizes is None:
            return 25
        return float(self.node_sizes[index])

    def _update_elements(self, dps=4):
        """
        Updates the elements needed for the Dash Cytoscape Graph object.

        :param dps: (int) Decimal places to round the edge values.
        """

        if self.graph is None:
            self.elements = []
            return
        nodes = list(self.graph.nodes())
        elements = []
        for i, node in enumerate(nodes):
            elements.append({
                "data": {
                    "id": str(node),
                    "label": str(node),
                    "group": self._get_node_group(node),
                    "size": self._get_node_size(i),
                }
            })
        for u, v, data in self.graph.edges(data=True):
            weight = round(float(data.get("weight", 1.0)), dps)
            elements.append({
                "data": {"source": str(u), "target": str(v), "weight": weight}
            })
        self.elements = elements

    def _generate_layout(self):
        """
        Generates the layout for cytoscape.

        :return: (dbc.Container) Returns Dash Bootstrap Component Container containing the layout of UI.
        """

        return dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(self._get_default_controls(), width=3),
                        dbc.Col(
                            [
                                cyto.Cytoscape(
                                    id="cytoscape",
                                    elements=self.elements,
                                    layout={"name": "cose"},
                                    stylesheet=self.stylesheet,
                                    style={"width": "100%", "height": "700px"},
                                ),
                                self._get_toast(),
                            ],
                            width=9,
                        ),
                    ]
                )
            ],
            fluid=True,
        )

    def _assign_colours_to_groups(self, groups):
        """
        Assigns the colours to industry or sector groups by creating a dictionary of group name to colour.

        :param groups: (List) List of industry groups as strings.
        """

        colours = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51",
                   "#577590", "#43aa8b", "#f94144", "#90be6d", "#f9c74f"]
        random.shuffle(colours)
        self.group_colours = {g: colours[i % len(colours)] for i, g in enumerate(groups)}

    def _style_colours(self):
        """
        Appends the colour styling to stylesheet for the different groups.
        """

        for group, colour in self.group_colours.items():
            self.stylesheet.append({
                "selector": f'node[group = "{group}"]',
                "style": {"background-color": colour},
            })

    def _assign_sizes(self):
        """
        Assigns the node sizing by appending to the stylesheet.
        """

        for style in self.stylesheet:
            if style.get("selector") == "node":
                style["style"]["width"] = "data(size)"
                style["style"]["height"] = "data(size)"

    def get_server(self):
        """
        Returns a small Flask server.

        :return: (Dash) Returns the Dash app object, which can be run using run_server.
            Returns a Jupyter Dash object if DashGraph has been initialised for Jupyter Notebook.
        """

        return self.app

    @staticmethod
    def _update_cytoscape_layout(layout):
        """
        Callback function for updating the cytoscape layout.
        The useful layouts for MST have been included as options (cola, cose-bilkent, spread).

        :return: (Dict) Dictionary of the key 'name' to the desired layout (e.g. cola, spread).
        """

        return {"name": layout}

    def _update_stat_json(self, stat_name):
        """
        Callback function for updating the statistic shown.

        :param stat_name: (str) Name of the statistic to display (e.g. graph_summary).
        :return: (json) Json of the graph information depending on chosen statistic.
        """

        if stat_name == "graph_summary":
            return json.dumps(self.get_graph_summary(), indent=2)
        return json.dumps({}, indent=2)

    def get_graph_summary(self):
        """
        Returns the Graph Summary statistics.
        The following statistics are included - the number of nodes and edges, smallest and largest edge,
        average node connectivity, normalised tree length and the average shortest path.

        :return: (Dict) Dictionary of graph summary statistics.
        """

        if self.graph is None:
            return {}
        weights = [d.get("weight", 1.0) for _, _, d in self.graph.edges(data=True)]
        n_nodes = self.graph.number_of_nodes()
        n_edges = self.graph.number_of_edges()
        avg_degree = sum(dict(self.graph.degree()).values()) / n_nodes if n_nodes else 0
        avg_connectivity = nx.average_node_connectivity(self.graph) if n_nodes > 1 else 0
        avg_shortest = nx.average_shortest_path_length(self.graph) if n_nodes > 1 and nx.is_connected(self.graph) else None
        total_weight = sum(weights) if weights else 0
        norm_tree_len = total_weight / (n_nodes - 1) if n_nodes > 1 else 0
        return {
            "nodes": n_nodes,
            "edges": n_edges,
            "min_edge": min(weights) if weights else None,
            "max_edge": max(weights) if weights else None,
            "avg_degree": avg_degree,
            "avg_connectivity": avg_connectivity,
            "normalized_tree_length": norm_tree_len,
            "avg_shortest_path": avg_shortest,
        }

    def _round_decimals(self, dps):
        """
        Callback function for updating decimal places.
        Updates the elements to modify the rounding of edge values.

        :param dps: (int) Number of decimals places to round to.
        :return: (List) Returns the list of elements used to define graph.
        """

        self._update_elements(dps=int(dps))
        return self.elements

    def _get_default_stylesheet(self):
        """
        Returns the default stylesheet for initialisation.

        :return: (List) A List of definitions used for Dash styling.
        """

        return [
            {
                "selector": "node",
                "style": {
                    "label": "data(label)",
                    "background-color": "#2a9d8f",
                    "width": "data(size)",
                    "height": "data(size)",
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
        ]

    def _get_toast(self):
        """
        Toast is the floating colour legend to display when industry groups have been added.
        This method returns the toast component with the styled colour legend.

        :return: (html.Div) Returns Div containing colour legend.
        """

        if not self.group_colours:
            return html.Div()
        items = []
        for group, colour in self.group_colours.items():
            items.append(
                html.Div(
                    [
                        html.Span(style={
                            "display": "inline-block",
                            "width": "12px",
                            "height": "12px",
                            "backgroundColor": colour,
                            "marginRight": "6px",
                        }),
                        html.Span(group),
                    ],
                    style={"marginBottom": "4px"},
                )
            )
        return html.Div(items, style={"paddingTop": "10px"})

    def _get_default_controls(self):
        """
        Returns the default controls for initialisation.

        :return: (dbc.Card) Dash Bootstrap Component Card which defines the side panel.
        """

        return dbc.Card(
            dbc.CardBody(
                [
                    html.H5("Layout"),
                    dcc.Dropdown(
                        id="layout-dropdown",
                        options=[{"label": n, "value": n} for n in
                                 ["cose", "cose-bilkent", "cola", "circle", "concentric", "breadthfirst", "grid",
                                  "random", "spread"]],
                        value="cose",
                        clearable=False,
                    ),
                    html.Hr(),
                    html.H5("Edge Precision"),
                    dcc.Slider(
                        id="decimal-slider",
                        min=0,
                        max=6,
                        step=1,
                        value=4,
                        marks={i: str(i) for i in range(7)},
                    ),
                    html.Hr(),
                    html.H5("Statistics"),
                    dcc.Dropdown(
                        id="stat-dropdown",
                        options=[{"label": "Graph Summary", "value": "graph_summary"}],
                        value="graph_summary",
                        clearable=False,
                    ),
                    html.Pre(id="stat-json", style={"whiteSpace": "pre-wrap"}),
                ]
            )
        )


class PMFGDash(DashGraph):
    """
    PMFGDash class, a child of DashGraph, is the Dash interface class to display the PMFG.
    """

    def __init__(self, input_graph, app_display='default'):
        """
        Initialise the PMFGDash class but override the layout options.
        """

        super().__init__(input_graph, app_display=app_display)

    def _update_elements(self, dps=4):
        """
        Overrides the parent DashGraph class method _update_elements, to add styling for the MST edges.
        Updates the elements needed for the Dash Cytoscape Graph object.

        :param dps: (int) Decimal places to round the edge values. By default, this will round to 4 d.p's.
        """

        if self.graph is None:
            self.elements = []
            return
        mst_edges = set(tuple(sorted(e)) for e in self.input_graph.get_mst_edges())
        nodes = list(self.graph.nodes())
        elements = []
        for i, node in enumerate(nodes):
            elements.append({
                "data": {
                    "id": str(node),
                    "label": str(node),
                    "group": self._get_node_group(node),
                    "size": self._get_node_size(i),
                }
            })
        for u, v, data in self.graph.edges(data=True):
            weight = round(float(data.get("weight", 1.0)), dps)
            classes = "mst-edge" if tuple(sorted((u, v))) in mst_edges else ""
            elements.append({
                "data": {"source": str(u), "target": str(v), "weight": weight},
                "classes": classes,
            })
        self.elements = elements

    def _get_default_stylesheet(self):
        """
        Gets the default stylesheet and adds the MST styling.

        :return: (List) Returns the stylesheet to be added to the graph.
        """

        stylesheet = super()._get_default_stylesheet()
        stylesheet.append({
            "selector": ".mst-edge",
            "style": {"line-color": "#e76f51", "width": 4},
        })
        return stylesheet
