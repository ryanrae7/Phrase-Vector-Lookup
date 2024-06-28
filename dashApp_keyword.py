import pandas as pd
import plotly.express as px
from plotly import graph_objects as go
import ipywidgets as widgets
from IPython.display import display

import numpy as np
import spacy
from collections import Counter

import dash
from dash import dcc, html
from dash.dependencies import Input, Output

import dash_bootstrap_components as dbc #used for formatting layout of the dash

nlp = spacy.load("en_core_web_sm")

#File path to your Excel file
file_path = 'C:/Users/rgae/OneDrive - QuidelOrtho/Documents/All QuidelFiles/Excel Files/6-21-2024 Copy of CTS.xlsx'
#Read the Excel file and extract necessary columns
df = pd.read_excel(file_path, index_col = None, na_values = ['NA'], usecols = 'AL, BN, BM')

#use pd to date time to utilize in the mapping of the data over time
df['Complete Loc Dt'] = pd.to_datetime(df['Complete Loc Dt'])
#Combine the text columns into a single text column for analysis
df['combined_text'] = df.astype(str).agg(' '.join, axis=1)



#find frequency of text
combined_text = ' '.join(df['combined_text'])
doc = nlp(combined_text)

#Extract words and calculate their frequencies
words = [token.text.lower() for token in doc if token.is_alpha]
word_freq = Counter(words)

#convert the frequency data to a DataFrame for better visualization
word_freq_df = pd.DataFrame(word_freq.items(), columns=['Word', 'Frequency']).sort_values(by='Frequency', ascending=False).reset_index()

words_to_filter = ['the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'I', 'i']
word_freq_df = word_freq_df[~word_freq_df['Word'].isin(words_to_filter)]
word_freq_df = word_freq_df.drop(word_freq_df[word_freq_df['Count'] <= 1].index)

#Define the function to count keyword frequencies
def count_keywords(text, keywords):
    text = text.lower()
    return {keyword: 1 if keyword in text else 0 for keyword in keywords} #binary values
#############################

#initialize app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
app.layout = html.Div([  # dbc rows and col where number of columns and rows are determined by how many rows and columns are in the parameters
    dbc.Row(  # e.g. row(col col col) <-- 3 columns || row (col) <-- 1 column https://dash-bootstrap-components.opensource.faculty.ai/docs/components/layout/
        [
            dbc.Col([
                html.H2("Keyword Analysis", style={'font-size': '20px'}),  # H2 indicates sub heading with the following properties
                dcc.Dropdown(
                    id = 'keyword-dropdown',
                    options = [{'label': f"{word} ----- {freq}", 'value': word} for word, freq in zip(word_freq_df['Word'], word_freq_df['Frequency'])],
                    multi = True,
                    placeholder = 'Select keywords',
                    value = [],
                    style = {'width': '100%'}
                ),
                dbc.Button(
                    id = 'submit-button',
                    n_clicks = 0,
                    children = 'Submit',
                    color = 'primary',
                    style = {'margin-top': '10px'}
                ),
            ], width = {'size': 'auto'},
                style = {'font-size': '16px'}),

            dbc.Col([
                html.H2("Toggle Stack", style={'font-size': '20px'}),
                dcc.Checklist(
                    id = 'toggle-checklist',
                    options = [
                        {'label': 'Grouped', 'value': 'group'},
                    ], value = ['True'],
                    style = {'font-size': '16px'},
                )
            ])
        ], justify = 'between'
    ),

    dbc.Row([
        dbc.Col([
            dcc.Graph(id='keyword-graph')  # Populate graph based on the dropdown selection
        ]),
    ]),
])


#set up an callback function that updates the output based on our input
@app.callback(
    Output(component_id = 'keyword-graph', component_property = 'figure'),
    [Input(component_id = 'submit-button', component_property = 'n_clicks'),
     Input(component_id = 'toggle-checklist', component_property = 'value')],
    [dash.dependencies.State('keyword-dropdown', 'value')]
)

#create function that updates figure
def update_graph(n_clicks, toggle, keywords):  
    #want it so that for each click, updates using property of n_clicks and changing the keywords input

    #consider the case that exist empty string
    if not keywords:
        #return empty dict if exist empty string
        return {}

    #apply the function here which applies strip() and lower() while splitting by ','
    df['keyword_frequency'] = df['combined_text'].apply(lambda x: count_keywords(x, keywords))

    #convert back to dataframe and fill zero if missing(N/A)
    keyword_df = df['keyword_frequency'].apply(pd.Series).fillna(0)

    #group by date and reset the index like before
    keyword_bydate_df = df[['Complete Loc Dt']].join(keyword_df).groupby('Complete Loc Dt').sum().reset_index()
    keyword_bydate_df = keyword_bydate_df.melt(id_vars=['Complete Loc Dt'], var_name='Keyword', value_name='Service Calls')
    keyword_bydate_df = keyword_bydate_df.rename(columns = {'Complete Loc Dt': 'Date'})

    #set the barmode based on the toggle value
    barmode = 'group' if 'group' in toggle else 'stack'

    #create the histogram here
    fig = px.histogram(keyword_bydate_df, x = 'Date', y = 'Service Calls', color = 'Keyword', barmode = barmode, title = 'Keyword Presence Over Time')
    fig.update_xaxes(
        dtick=86400000.0 * 14 , #biweekly
        #tickformat="%b\n%Y",
        ticklabelmode="period"
    )

    #return the fig
    return fig
    
    
#############################

if __name__ == '__main__':
    app.run_server(debug = True, port = 8051) #specify port, couldn't terminate port 8050 so work around for now
    