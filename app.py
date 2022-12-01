# %% [markdown]
# # PART 0. **Imports & Functions**

# %% [markdown]
# ## **Import Libraries**

# %%
#################################################################################################################################################
######################### ----------- Import Python Libraries
import dash
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import dash_bootstrap_components as dbc

from functools import reduce
from dash_iconify import DashIconify
from plotly.subplots import make_subplots
from dash.dependencies import Output, Input
from dash import Dash, dcc, html
from datetime import datetime, timedelta, date

warnings.filterwarnings("ignore")


# %% [markdown]
# ## **Import Datasets**

# %%
def label_engagement(row):
    if (
        row["questions_posted"] > 0
        or row["engagement_quizzes"] > 0
        or row["engagement_exams"] > 0
        or row["engagement_lessons"] > 0
    ):
        return 1

def label_payer(row):
    if row["tot_purchases"] > 0:
        return 1

# %%
def load_dfs():
    #############################################
    global course_ratings
    global course_info
    global student_learning
    global map_df
    global course_info_names
    global students_info_df
    global student_purchases_df
    global dates_df
    global date_df_labels
    global special_dates
    global countries
    #############################################
    course_ratings = pd.read_csv("365_database/365_course_ratings.csv")
    course_info = pd.read_csv("365_database/365_course_info.csv")
    student_learning = pd.read_csv("365_database/365_student_learning.csv")
    map_df = pd.read_csv("365_database/map_df.csv")
    course_info_names = pd.read_csv("365_database/course_info_names.csv").drop(
        columns="Unnamed: 0"
    )
    students_info_df = pd.read_csv("365_database/students_info_df.csv").drop(
        columns="Unnamed: 0"
    )
    students_info_df["engaged_somehow"] = students_info_df.apply(
        lambda row: label_engagement(row), axis=1
    )
    students_info_df["pay_user"] = students_info_df.apply(
        lambda row: label_payer(row), axis=1
    )
    
    student_purchases_df = pd.read_csv("365_database/student_purchases_df.csv").drop(
        columns="Unnamed: 0"
    )
    dates_df = pd.read_csv("365_database/dates_df.csv").drop(columns="Unnamed: 0")
    dates_df["tot_hours_watched"] = dates_df["tot_minutes_watched"] / 60
    dates_df["mins_by_student"] = dates_df["tot_minutes_watched"] / 60
    date_df_labels = eval(open("365_database/date_df_labels.json").read())

    special_dates = [
        ["2022-01-17", "2022-01-20"],
        ["2022-03-21", "2022-04-01"],
        ["2022-05-20", "2022-06-02"],
        ["2022-07-18", "2022-07-30"],
        ["2022-09-16", "2022-09-17"],
        ["2022-09-19", "2022-10-01"],
    ]


# %% [markdown]
# ## **Functions**

# %% [markdown]
# ### Filter by Dates

# %%
def filter_by_dates(filt_df, range_dates, date_column_name):
    filt_df[date_column_name] = pd.to_datetime(filt_df[date_column_name])
    mask = (filt_df[date_column_name] >= pd.to_datetime(range_dates[0])) & (
        filt_df[date_column_name] <= pd.to_datetime(range_dates[1])
    )
    filtered_df = filt_df.loc[mask]
    filtered_df.sort_values([date_column_name], ascending=True, inplace=True)
    return filtered_df

# %% [markdown]
# ### Generate World Map

# %%
def plot_map(map_plot_df, date_column_name, range_dates, colors, hover_names):
    filtered_df = filter_by_dates(map_plot_df, range_dates, date_column_name)
    df_map = filtered_df.groupby(["Country_Name", "Country_ISO_A3"]).sum().reset_index()
    fig = px.treemap(
        df_map,
        path=[px.Constant("world"), "Country_Name"],
        values="N° of Registrations",
        color="N° of Registrations",
        color_continuous_scale="RdBu",
        color_continuous_midpoint=np.average(
            df_map["N° of Registrations"], weights=df_map["N° of Registrations"]
        ),
    )
    fig.update_layout(
        title='Information by Country',
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        margin = dict(t=35, l=15, r=15, b=15)
    )
    fig.update_coloraxes(showscale=False)

    return fig


# %% [markdown]
# ### Lines PLot

# %%
def courses_popularity(range_dates):
    course_rating_1 = filter_by_dates(course_ratings, range_dates, 'date_rated').groupby('course_id').mean().reset_index().drop(columns='student_id')
    course_rating_2 = filter_by_dates(student_learning, range_dates, 'date_watched').groupby('course_id').sum().reset_index().drop(columns='student_id')
    course_rating_3 = filter_by_dates(student_learning, range_dates, 'date_watched').groupby('course_id').mean().reset_index().drop(columns='student_id')
    course_rating_4 = filter_by_dates(student_learning, range_dates, 'date_watched').groupby('course_id').count().reset_index().drop(columns='student_id')
    df_to_merge =[
        course_rating_1,
        course_rating_2,
        course_rating_3,
        course_rating_4,
        course_info
    ]

    course_info_names = reduce(lambda  left,right: pd.merge(left,right,on=['course_id'], how='outer'), df_to_merge).drop(columns='date_watched')
    course_info_names.columns = ['course_id', 'avg_course_rating', 'tot_minutes_watched', 'avg_minutes_watched', 'times_watched', 'course_title']
    course_info_names['intro_word'] =  course_info_names['course_title'].str.contains(pat ='Intro[a-z]', regex = True)
    course_info_names['ml'] =  course_info_names['course_title'].str.contains(pat ='Mach[a-z]', regex = True)
    course_info_names['popularity_1'] = course_info_names['times_watched'] / course_info_names['avg_course_rating']
    course_info_names['popularity_2'] = course_info_names['times_watched'] * course_info_names['avg_course_rating']
    course_info_names['popularity_3'] = course_info_names['popularity_2'] / course_info_names['tot_minutes_watched']
    course_info_names['popularity_normal']=(course_info_names['popularity_1']-course_info_names['popularity_1'].min())/(course_info_names['popularity_1'].max()-course_info_names['popularity_1'].min())
    course_info_names.sort_values(by='popularity_2', ascending=False)
    return course_info_names.sort_values(by='popularity_2', ascending=False)

# %%
def lines_plot(df_name, chart_title, x_params, x_title, y_params, fill_colors, trendline, mk_size, legend, hovermode, new_labels, range_dates, agg_parameter):
    #############################################
    #############################################
    if df_name == 'dates_df':
        df_to_plot = filter_by_dates(dates_df, range_dates, 'date')
    if df_name == 'course_ratings':
        df_to_plot = courses_popularity(range_dates).head(15)
    else:
        if agg_parameter == 'sum':
            df_to_plot = filter_by_dates(students_info_df, range_dates, 'date_registered').groupby('student_country').sum().reset_index()
        if agg_parameter == 'mean':
            df_to_plot = filter_by_dates(students_info_df, range_dates, 'date_registered').groupby('student_country').mean().reset_index()
        if agg_parameter == 'count':
            df_to_plot = filter_by_dates(students_info_df, range_dates, 'date_registered').groupby('student_country').count().reset_index()
    #############################################
    if trendline:
        trend = 'ols'
    else:
        trend = None
    #############################################
    fig = px.scatter(
            df_to_plot,
            x=x_params,
            y=y_params,
            trendline=trend,
            # markers=mk,
            color_discrete_sequence=fill_colors,
        )
    #############################################
    fig.update_traces(
        hovertemplate="%{y}",
        textposition="bottom right",
        marker=dict(
            size=mk_size,
        ),
    )
    #############################################
    fig.update_layout(
        title=chart_title,
        hovermode=hovermode,
        showlegend=legend,
        transition_duration=500,
        xaxis_title=x_title,
        yaxis_title=None,
        legend_title=None,
        paper_bgcolor='#FFFFFF',
        plot_bgcolor="#FFFFFF",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(
            l=2,
            r=2,
            b=2,
            t=50,
        ),
        hoverlabel=dict(
            bgcolor="#FFFFFF",
            font_size=12,
            font=dict(color="#020C1E"),
        ),
    )

    #############################################
    fig.for_each_trace(
        lambda t: t.update(
            name=new_labels[t.name],
            legendgroup=new_labels[t.name],
            hovertemplate=t.hovertemplate.replace(t.name, new_labels[t.name]),
        )
    )
    #############################################
    return fig

# %% [markdown]
# # PART 1. **Dash Body**

# %% [markdown]
# ## Time dataframe

# %%
def fig_map(range_dates):
    date_column_name = 'date'
    colors = "N° of Registrations"
    hover_names =  "Country_Name"
    return plot_map(map_df, date_column_name, range_dates, colors, hover_names)

# %%
def eval_special_dates(s_date, range_dates):
    s_date[0], s_date[1] = date.fromisoformat(str(s_date[0])), date.fromisoformat(str(s_date[1]))
    if s_date[0] >= range_dates[0] and s_date[1] <= range_dates[1]:
        x_0 = s_date[0]
        x_1 = s_date[1]
    if s_date[0] < range_dates[0] and s_date[1] <= range_dates[1]:
        x_0 = range_dates[0]
        x_1 = s_date[1]
    if s_date[0] >= range_dates[0] and s_date[1] > range_dates[1]:
        x_0 = s_date[0]
        x_1 = range_dates[1]
    if s_date[0] < range_dates[0] and s_date[1] > range_dates[1]:
        x_0 = range_dates[0]
        x_1 = range_dates[1]
    if s_date[0] < range_dates[0] and s_date[1] < range_dates[0]:
        x_0, x_1 = None, None
    if s_date[0] > range_dates[1] and s_date[1] > range_dates[1]:
        x_0, x_1 = None, None
    return x_0, x_1


# %%
#################################################################################################################################################
def registrations_chart(range_dates):
    fig_args = [
        "dates_df",  # df_name
        "Registrations, Courses and Exams Presented",  # chart_title
        "date",  # x_params
        "Timeline",  # x_title
        # y_params
        [
            "students_registered",
            "tot_courses_watched",
            "tot_exams_presented",
        ],
        # fill_colors
        [
            "#020C1E",
            "#72BDC5",
            "#7762F6",
            "#C7A638",
            "#152A3A",
            "#848A93",
        ],
        True,  # Tendline Yes or No
        5,  # Marker Size
        False,  # Legend Yes or No
        "x unified",  # Hovermode
        date_df_labels,  # new_labels
        range_dates,
        "agg_parameter",
    ]
    fig = lines_plot(*fig_args)
    fig.update_traces(mode="lines")
    for s_date in special_dates:
        x_0, x_1 = eval_special_dates(s_date, range_dates)
        if x_0 == None:
            continue
        fig.add_vrect(
            x0=x_0,
            x1=x_1,
            fillcolor="#002EEB",
            opacity=0.3,
            layer="below",
            line_width=0,
        )
    return fig


# %%
#################################################################################################################################################
def mins_courses_chart(range_dates):
    fig_args = [
        "dates_df",  # df_name
        "Courses and Minutes Watched Over Time",  # chart_title
        "date",  # x_params
        "Timeline",  # x_title
        # y_params
        [
            "tot_hours_watched",
            "tot_courses_watched",
        ],
        # fill_colors
        [
            "#020C1E",
            "#72BDC5",
            "#7762F6",
            "#C7A638",
            "#152A3A",
            "#848A93",
        ],
        True,  # Tendline Yes or No
        6,  # Marker Size
        False,  # Legend Yes or No
        "x unified",  # Hovermode
        date_df_labels,  # new_labels
        range_dates,
        'agg_parameter'
    ]
    fig = lines_plot(*fig_args)
    fig.update_traces(mode = 'lines')

    for s_date in special_dates:
        x_0, x_1 = eval_special_dates(s_date, range_dates)
        if x_0 == None:
            continue
        fig.add_vrect(
            x0=x_0,
            x1=x_1,
            fillcolor="#002EEB",
            opacity=0.3,
            layer="below",
            line_width=0,
        )
    return fig


# %%
#################################################################################################################################################
def engagement_chart(range_dates):
    fig_args = [
        "dates_df",  # df_name
        "Engagement Over Time",  # chart_title
        "date",  # x_params
        "Timeline",  # x_title
        # y_params
        [
            "engagement_quizzes",
            "engagement_exams",
            "engagement_lessons"
        ],
        # fill_colors
        [
            "#020C1E",
            "#72BDC5",
            "#7762F6",
            "#C7A638",
            "#152A3A",
            "#848A93",
        ],
        True,  # Tendline Yes or No
        6,  # Marker Size
        False,  # Legend Yes or No
        "x unified",  # Hovermode
        date_df_labels,  # new_labels
        range_dates,
        'agg_parameter'
    ]
    fig = lines_plot(*fig_args)
    fig.update_traces(mode = 'lines')
    
    for s_date in special_dates:
        x_0, x_1 = eval_special_dates(s_date, range_dates)
        if x_0 == None:
            continue
        fig.add_vrect(
            x0=x_0,
            x1=x_1,
            fillcolor="#002EEB",
            opacity=0.3,
            layer="below",
            line_width=0,
        )
    return fig


# %%
#################################################################################################################################################
def purchases_chart(range_dates):
    fig_args = [
        "dates_df",  # df_name
        "Purchases Over Time",  # chart_title
        "date",  # x_params
        "Timeline",  # x_title
        # y_params
        [
            "annual_purchases",
            "monthly_purchases",
            "quarterly_purchases"
        ],
        # fill_colors
        [
            "#020C1E",
            "#72BDC5",
            "#7762F6",
            "#C7A638",
            "#152A3A",
            "#848A93",
        ],
        False,  # Tendline Yes or No
        6,  # Marker Size
        False,  # Legend Yes or No
        "x unified",  # Hovermode
        date_df_labels,  # new_labels
        range_dates,
        'agg_parameter'
    ]
    fig = lines_plot(*fig_args)
    fig.update_traces(mode = 'lines')
    
    for s_date in special_dates:
        x_0, x_1 = eval_special_dates(s_date, range_dates)
        if x_0 == None:
            continue
        fig.add_vrect(
            x0=x_0,
            x1=x_1,
            fillcolor="#002EEB",
            opacity=0.3,
            layer="below",
            line_width=0,
        )
    return fig


# %%
#################################################################################################################################################
def academic_chart(range_dates):
    fig_args = [
        "dates_df",  # df_name
        "Academic Performance Over Time",  # chart_title
        "date",  # x_params
        "Timeline",  # x_title
        # y_params
        [
            "avg_exam_results",
            "avg_exam_completion_time",
        ],
        # fill_colors
        [
            "#020C1E",
            "#72BDC5",
            "#7762F6",
            "#C7A638",
            "#152A3A",
            "#848A93",
        ],
        True,  # Tendline Yes or No
        6,  # Marker Size
        False,  # Legend Yes or No
        "x unified",  # Hovermode
        date_df_labels,  # new_labels
        range_dates,
        'agg_parameter'
    ]
    fig = lines_plot(*fig_args)
    fig.update_traces(mode = 'lines')
    for s_date in special_dates:
        x_0, x_1 = eval_special_dates(s_date, range_dates)
        if x_0 == None:
            continue
        fig.add_vrect(
            x0=x_0,
            x1=x_1,
            fillcolor="#002EEB",
            opacity=0.3,
            layer="below",
            line_width=0,
        )
    return fig


# %% [markdown]
# ## Total Numbers

# %%
def students_registered(range_dates):
    filtered_df = filter_by_dates(dates_df, range_dates, 'date')
    total = filtered_df.students_registered.sum()
    return int(total)

def students_interactions(range_dates):
    filtered_df = filter_by_dates(dates_df, range_dates, 'date')
    total = filtered_df.questions_posted.sum() + filtered_df.engagement_quizzes.sum() + filtered_df.engagement_exams.sum() + filtered_df.engagement_lessons.sum()
    return int(total)

def total_purchases(range_dates):
    filtered_df = filter_by_dates(dates_df, range_dates, 'date')
    total = filtered_df.annual_purchases.sum() + filtered_df.monthly_purchases.sum() + filtered_df.quarterly_purchases.sum()
    annual = filtered_df.annual_purchases.sum()
    quarterly = filtered_df.quarterly_purchases.sum()
    monthly = filtered_df.monthly_purchases.sum()
    return int(total), annual, quarterly, monthly

def hours_watched(range_dates):
    filtered_df = filter_by_dates(dates_df, range_dates, 'date')
    total = filtered_df.tot_hours_watched.sum()
    return int(total)

def avg_exam_results(range_dates):
    filtered_df = filter_by_dates(dates_df, range_dates, 'date')
    total = f'{filtered_df.avg_exam_results.mean().round(2)}%'
    return total

def avg_course_popularity():
    course_info_names['popularity']=(course_info_names['popularity']-course_info_names['popularity'].min())/(course_info_names['popularity'].max()-course_info_names['popularity'].min())
    total = f'{course_info_names.popularity.mean().avg_exam_results.mean().round(2)}%'
    return total

# %% [markdown]
# ## Bar plot

# %%
def bars_plot(df_name, chart_title, x_params, x_title, y_params, fill_colors, legend, hovermode, new_labels, range_dates):
    #############################################
    #############################################
    if df_name == 'course_ratings':
        df_to_plot = courses_popularity(range_dates).head(15)
        # df_to_plot['popularity_normal'] = df_to_plot['popularity_normal'].round(2).astype(str) + '%'
    #############################################
    fig = px.bar(
        df_to_plot,
        barmode="group",
        x=x_params,
        y=y_params,
        text_auto=".2s",
        color_discrete_sequence=fill_colors,
    )
    #############################################
    fig.update_layout(
        title=chart_title,
        hovermode=hovermode,
        showlegend=legend,
        transition_duration=500,
        xaxis_title=x_title,
        yaxis_title=None,
        legend_title=None,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        bargap=0.15,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(
            l=2,
            r=2,
            b=2,
            t=50,
        ),
        hoverlabel=dict(
            bgcolor='#FFFFFF',
            font_size=12,
            font=dict(color='#000000'),
        ),
    )
    #############################################
    fig.update_traces(
        hovertemplate="%{y}",
        textangle=-70,
        textposition="outside",
        texttemplate="%{y:.1f}",
        cliponaxis=False,
    )
    #############################################
    fig.for_each_trace(
        lambda t: t.update(
            name=new_labels[t.name],
            legendgroup=new_labels[t.name],
            hovertemplate=t.hovertemplate.replace(t.name, new_labels[t.name]),
        )
    )
    #############################################
    return fig

# %% [markdown]
# ## Courses

# %%
#################################################################################################################################################
def top_courses(range_dates):
    new_labels = {
        "popularity_normal": "Popularity",
        "times_watched": "Times Watched",
        "avg_course_rating": "Course Rating (avg)",
    }
    fig_A_args = [
        "course_ratings",  # df_name
        "15 Most Popular Courses",  # chart_title
        "course_title",  # x_params
        "Course",  # x_title
        # y_params
        [
            "times_watched",
        ],
        # fill_colors
        [
            "#152A3A",
            "#848A93",
        ],
        False,  # Tendline Yes or No
        6,  # Marker Size
        False,  # Legend Yes or No
        "x unified",  # Hovermode
        new_labels,  # new_labels
        range_dates, # Dates
        "agg_parameter",
    ]
    ################################################################
    ################################################################
    fig_B_args = [
        "course_ratings",  # df_name
        "15 Most Popular Courses",  # chart_title
        "course_title",  # x_params
        "Course",  # x_title
        # y_params
        ["popularity_normal", "avg_course_rating"],
        # fill_colors
        [
            "#020C1E",
            "#72BDC5",
            "#7762F6",
            "#C7A638",
            "#152A3A",
            "#848A93",
        ],
        False,  # Legend Yes or No
        "x unified",  # Hovermode
        new_labels,  # new_labels
        range_dates,
    ]
    ################################################################
    ################################################################
    fig_A = lines_plot(*fig_A_args)
    fig_A.update_traces(mode="lines")
    fig_A.update_traces(yaxis="y2")
    fig_B = bars_plot(*fig_B_args)
    for s_date in special_dates:
        x_0, x_1 = eval_special_dates(s_date, range_dates)
        if x_0 == None:
            continue
        fig_B.add_vrect(
            x0=x_0,
            x1=x_1,
            fillcolor="#002EEB",
            opacity=0.3,
            layer="below",
            line_width=0,
        )

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_traces(fig_A.data + fig_B.data)

    return fig

# %% [markdown]
# # PART 2. **Dash APP**

# %% [markdown]
# ## **Update all Charts** (Main Function)

# %%
def update_charts(range_dates):
    csr_charts = html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H3("Registrations"),
                                    html.H4(students_registered(range_dates)),
                                ],
                                className="csr-kpi",
                            ),
                            html.Div(
                                [
                                    html.H3("Purchases"),
                                    html.H4(total_purchases(range_dates)[0]),
                                ],
                                className="csr-kpi",
                            ),
                            html.Div(
                                [
                                    html.H3("Interactions"),
                                    html.H4(students_interactions(range_dates)),
                                ],
                                className="csr-kpi",
                            ),
                            html.Div(
                                [
                                    html.H3("Hours Watched"),
                                    html.H4(hours_watched(range_dates)),
                                ],
                                className="csr-kpi",
                            ),
                            html.Div(
                                [
                                    html.H3("Exams Results (avg)"),
                                    html.H4(avg_exam_results(range_dates)),
                                ],
                                className="csr-kpi",
                            ),
                        ],
                        className="some-kpis",
                    ),
                    html.Div(
                        [
                            dcc.Graph(
                                figure=purchases_chart(range_dates),
                                className="chart_style",
                            ),
                            dcc.Graph(
                                figure=academic_chart(range_dates),
                                className="chart_style",
                            ),
                            dcc.Graph(
                                figure=fig_map(range_dates), className="chart_style"
                            ),
                        ],
                        className="csr-3-columns",
                    ),
                    html.Div(
                        [
                            dcc.Graph(
                                figure=registrations_chart(range_dates),
                                className="chart_style",
                            ),
                            dcc.Graph(
                                figure=engagement_chart(range_dates),
                                className="chart_style",
                            ),
                            dcc.Graph(
                                figure=mins_courses_chart(range_dates),
                                className="chart_style",
                            ),
                        ],
                        className="csr-3-columns",
                    ),
                    html.Div(
                        [   
                            dcc.Graph(
                                figure=top_courses(range_dates),
                                className="chart_style",
                            ),
                        ], className='chart-1-alone'
                    ),
                ],
                id="container-charts-time",
            ),
        ]
    )

    return csr_charts


######################################################################
######################################################################
# fig_map(range_dates)
# registrations_chart(range_dates)
# mins_courses_chart(range_dates)
# engagement_chart(range_dates)
# purchases_chart(range_dates)
# academic_chart(range_dates)


# %% [markdown]
# ## Start Application

# %%
#################################################################################################################################################
######################### ----------- Start Application
external_scripts = [
    '/assets/aos.js',
    ]

app = dash.Dash(
    __name__,
    meta_tags=[
        {
            "name": "viewport",
            "content": "width=device-width, initial-scale=1.0, maximum-scale=1.1, minimum-scale=0.9,",
        }
    ],
    external_scripts=external_scripts,
)
server = app.server

# %% [markdown]
# ## App Layout

# %%
#################################################################################################################################################
#################################################################################################################################################
######################### -----------  App Layout
def serve_layout():
    # range_dates = ["2022-01-01", "2022-10-20"]
    load_dfs()
    csr_layout = html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        DashIconify(
                                            icon="flat-color-icons:calendar",
                                            id="cal-icon",
                                        ),
                                        className="cal-icon",
                                    ),
                                    #############################################
                                    dbc.Col(
                                        dcc.DatePickerRange(
                                            id="csr-date-picker",
                                            display_format="MMM DD, YYYY",
                                            min_date_allowed="2022-01-01",
                                            max_date_allowed="2022-10-20",
                                            updatemode = 'singledate',
                                            start_date="2022-01-01",
                                            end_date="2022-10-20",
                                            with_portal=False,
                                            show_outside_days=True,
                                        ),
                                    ),
                                ],
                                className="calendar-kit",
                            ),
                        ],
                        id="container-controls",
                    ),
                    html.Div(id="all_charts_body"),
                ],
                id="csr_subbody",
            ),
        ],
        id="csr_body",
    )
    return csr_layout


#############################################
#############################################
app.layout = serve_layout


# %% [markdown]
# ## **Callbacks**

# %%
###########################################################################################################
###########################################################################################################
######################### ----------- Tabs Callback
@app.callback(
    Output("all_charts_body", "children"),
    [Input("csr-date-picker", "start_date"), Input("csr-date-picker", "end_date")],
)
def populate_tabs(start_date, end_date):
    ##################################################
    range_dates = (date.fromisoformat(start_date), date.fromisoformat(end_date))
    ##################################################
    return update_charts(range_dates)


# %% [markdown]
# ## **Run Application**

# %%
#################################################################
######################### -----------  Server Initiation
if __name__ == "__main__":
    app.run_server(host='0.0.0.0', port=8080, debug=False, use_reloader=False)


