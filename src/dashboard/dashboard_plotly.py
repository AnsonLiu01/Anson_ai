import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from loguru import logger


class MainDashboard:
    """
    Simple dashboard that works directly with GetTopics instance
    """

    def __init__(self, topic_model, port: int = 8050):
        """
        Args:
            topic_model: GetTopics instance after running runner()
            port: Port to run dashboard on
        """
        self.model = topic_model
        self.port = port
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            suppress_callback_exceptions=True
        )
        self.setup_layout()
        self.setup_callbacks()

    def setup_layout(self):
        """Setup dashboard layout"""
        self.app.layout = dbc.Container([
            # Header
            dbc.Row([
                dbc.Col([
                    html.H1("📊 Topic Analysis Dashboard", className="text-primary mb-3"),
                    html.Hr()
                ])
            ], className="mb-4"),

            # Controls and Content
            dbc.Row([
                # Sidebar
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H4("⚙️ Controls")),
                        dbc.CardBody([
                            html.Label("Number of Topics:", className="fw-bold"),
                            dcc.Slider(
                                id='topic-slider',
                                min=5, max=20, step=1, value=10,
                                marks={i: str(i) for i in range(5, 21, 5)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            ),
                            html.Br(),
                            dbc.Button("🔄 Refresh", id="refresh-btn", color="primary", className="w-100")
                        ])
                    ])
                ], width=3),

                # Main content
                dbc.Col([
                    dbc.Tabs([
                        dbc.Tab(label="📍 Data Map", tab_id="datamap"),
                        dbc.Tab(label="📊 Topics", tab_id="topics"),
                        dbc.Tab(label="🔥 Similarity", tab_id="similarity"),
                    ], id="tabs", active_tab="datamap"),
                    html.Div(id="tab-content", className="mt-3")
                ], width=9)
            ])
        ], fluid=True)

    def setup_callbacks(self):
        """Setup callbacks"""

        @self.app.callback(
            Output("tab-content", "children"),
            [Input("tabs", "active_tab")]
        )
        def render_tab(active_tab):
            """Render tab content"""
            if active_tab == "datamap":
                return dbc.Card([
                    dbc.CardBody([
                        dcc.Loading(
                            dcc.Graph(id="datamap-graph", style={'height': '600px'})
                        )
                    ])
                ])
            elif active_tab == "topics":
                return dbc.Card([
                    dbc.CardBody([
                        dcc.Loading(
                            dcc.Graph(id="topics-graph", style={'height': '600px'})
                        )
                    ])
                ])
            elif active_tab == "similarity":
                return dbc.Card([
                    dbc.CardBody([
                        dcc.Loading(
                            dcc.Graph(id="similarity-graph", style={'height': '600px'})
                        )
                    ])
                ])

        @self.app.callback(
            Output("datamap-graph", "figure"),
            [Input("refresh-btn", "n_clicks"),
             Input("tabs", "active_tab")],
            prevent_initial_call=False
        )
        def update_datamap(n_clicks, active_tab):
            """Update data map visualization"""
            if active_tab != "datamap":
                return go.Figure()

            try:
                logger.info("Updating data map visualization")

                # Check if method exists
                if not hasattr(self.model, 'get_datamap_vis'):
                    raise AttributeError("get_datamap_vis method not found")

                # Call the method
                fig = self.model.get_datamap_vis()

                if fig is None:
                    raise ValueError("get_datamap_vis returned None")

                logger.success("Data map generated successfully")
                return fig

            except Exception as e:
                logger.error(f"Error generating data map: {e}")
                import traceback
                traceback.print_exc()

                # Return error figure
                fig = go.Figure()
                fig.add_annotation(
                    text=f"⚠️ Error loading data map:<br>{str(e)}<br><br>Check the terminal for details",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(size=14, color="red")
                )
                fig.update_layout(
                    title="Data Map Error",
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    height=600
                )
                return fig

        @self.app.callback(
            Output("topics-graph", "figure"),
            [Input("topic-slider", "value"),
             Input("refresh-btn", "n_clicks")],
            prevent_initial_call=False
        )
        def update_topics(top_n, n_clicks):
            """Update topics barchart"""
            try:
                logger.info(f"Updating topics barchart with top_n={top_n}")

                # Check if bert exists
                if not hasattr(self.model, 'bert') or self.model.bert is None:
                    raise ValueError("BERTopic model not found. Did you run topic_model.runner()?")

                # Check if labelled_topics exists
                if not hasattr(self.model, 'labelled_topics') or self.model.labelled_topics is None:
                    raise ValueError("Labelled topics not found. Did you run topic_model.runner()?")

                logger.info(f"Found {len(self.model.labelled_topics)} labelled topics")

                # Generate the figure
                fig = self.model.bert.visualize_barchart(
                    custom_labels=self.model.labelled_topics,
                    top_n_topics=top_n
                )

                if fig is None:
                    raise ValueError("visualize_barchart returned None")

                logger.success("Topics barchart generated successfully")
                return fig

            except Exception as e:
                logger.error(f"Error generating topics: {e}")
                import traceback
                traceback.print_exc()

                # Return error figure
                fig = go.Figure()
                fig.add_annotation(
                    text=f"⚠️ Error loading topics:<br>{str(e)}<br><br>Check the terminal for details",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(size=14, color="red")
                )
                fig.update_layout(
                    title="Topics Error",
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    height=600
                )
                return fig

        @self.app.callback(
            Output("similarity-graph", "figure"),
            [Input("refresh-btn", "n_clicks"),
             Input("tabs", "active_tab")],
            prevent_initial_call=False
        )
        def update_similarity(n_clicks, active_tab):
            """Update similarity heatmap"""
            if active_tab != "similarity":
                return go.Figure()

            try:
                logger.info("Updating similarity heatmap")

                # Check if method exists
                if not hasattr(self.model, 'eda_visual_similarity_heatmap'):
                    raise AttributeError("eda_visual_similarity_heatmap method not found")

                # Call the method
                fig = self.model.eda_visual_similarity_heatmap()

                if fig is None:
                    raise ValueError("eda_visual_similarity_heatmap returned None")

                logger.success("Similarity heatmap generated successfully")
                return fig

            except Exception as e:
                logger.error(f"Error generating similarity heatmap: {e}")
                import traceback
                traceback.print_exc()

                # Return error figure
                fig = go.Figure()
                fig.add_annotation(
                    text=f"⚠️ Error loading similarity:<br>{str(e)}<br><br>Check the terminal for details",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(size=14, color="red")
                )
                fig.update_layout(
                    title="Similarity Error",
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    height=600
                )
                return fig

    def runner(self, debug: bool = True):
        """
        Run the dashboard
        """
        logger.info(f"Starting dashboard at http://127.0.0.1:{self.port}")
        self.app.run(debug=debug, port=self.port)
