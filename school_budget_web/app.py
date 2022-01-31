# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_html_components as html
import pandas as pd
import dash_table
import dash_core_components as dcc
import dash_bootstrap_components as dbc


# app = dash.Dash(
#     __name__,
#     meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
# )
csv_path = "notebooks/train_v1.csv"
df = pd.read_csv(csv_path, index_col=0)
# Main App
# Define app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
# Define Layout
app.layout = html.Div(
    dbc.Container(
        fluid=True,
        children=[
            dbc.Navbar(
                dbc.Container(
                    [
                        html.A(
                            # Use row and col to control vertical alignment of logo / brand
                            dbc.Row(
                                [
                                    dbc.Col(html.Img(src=app.get_asset_url("school.png"), height="30px")),
                                    dbc.Col(dbc.NavbarBrand("School Budget", className="ms-2")),
                                ],
                                align="center",
                                className="g-0",
                            ),
                            href="/",
                            style={"textDecoration": "none"},
                        ),
                    ]
                ),
             ),
            html.Div(
                [
                    html.Br(),
                    html.P(
                        "Budgets for schools are huge, complex, and not standartized.",
                        className="lead",
                    ),
                    html.P(
                        "Hundreds of hours each year are spent manually labelling.",
                        className="lead",
                    ),
                    html.Img(src=app.get_asset_url("budget.png"), height="30px"),
                ],
                style={'textAlign': 'center'}
            ),
            dbc.Table.from_dataframe(df.head(), striped=True, bordered=True, hover=True)
        ],
        style={"margin": "auto"},
    )
)


if __name__ == '__main__':
    app.run_server(debug=True)
