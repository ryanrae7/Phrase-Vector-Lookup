import pandas as pd
import plotly.express as px
from plotly import graph_objects as go
import ipywidgets as widgets
from IPython.display import display

import numpy as np
import spacy
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

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

#Tokenize the text using spaCy and create a list of sentences
def tokenize(text):
    doc = nlp(text.lower())
    return ' '.join([token.text for token in doc]) #https://stackoverflow.com/questions/57187116/how-to-modify-spacy-tokens-doc-doc-tokens-with-pipeline-components-in-spacy

df['tokenized_text'] = df['combined_text'].apply(tokenize) #call the method above to the combined text column 
#cant do df['tokenized_text'] = tokenize(...) because you can't input series into a string parameter. Utilizing apply allows us to get around this and apply it for each row



#Use CountVectorizer to get common phrases
vectorizer = CountVectorizer(ngram_range = (1, 4))  #Bigrams and trigrams (how long each phrase can be i.e. 2 and 3)
X = vectorizer.fit_transform(df['tokenized_text'])

#Sum up the counts of each phrase
phrase_counts = X.sum(axis=0).A1
phrases = vectorizer.get_feature_names_out()

#Create a DataFrame with phrases and their counts
phrase_counts_df = pd.DataFrame({'Phrase': phrases, 'Count': phrase_counts}).sort_values(by = 'Count', ascending = False).reset_index()
phrase_counts_df = phrase_counts_df.drop(['index'], axis = 1)



#Define the function to count keyword frequencies
def count_keywords(text, keywords):
    text = text.lower()
    return {keyword: 1 if keyword in text else 0 for keyword in keywords} #binary values
#############################

#initialize app
app = dash.Dash(__name__, external_stylesheets = [dbc.themes.CYBORG]) 
app.layout = dbc.Container([ #dbc rows and col where number of columns and rows are determined by how many rows and columns are in the parameters
    dbc.Row([ #e.g. row(col col col) <-- 3 columns || row (col) <-- 1 column https://dash-bootstrap-components.opensource.faculty.ai/docs/components/layout/
        dbc.Col([
            html.H3("Phrase Analysis"), #H2 indicates sub heading with the following properties
            dcc.Dropdown(
                id='keyword-dropdown',
                options = [{'label': f"{phrase} ----- {freq}", 'value': phrase} for phrase, freq in zip(phrase_counts_df['Phrase'], phrase_counts_df['Count'])],
                multi=True,
                placeholder='Select keywords',
                value=[],
                style={'width': '100%'}
            ),
            dbc.Button(
                id='submit-button',
                n_clicks=0,
                children='Submit',
                color='primary',
                style={'margin-top': '10px'}
            ),
        ], width='auto'),
    ], justify='left', style={'margin-top': '20px'}),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='keyword-graph')  # Populate graph based on the dropdown selection
        ]),
    ]),
])

#set up an callback function that updates the output based on our input
@app.callback(
    Output(component_id = 'keyword-graph', component_property = 'figure'),
    [Input(component_id = 'submit-button', component_property = 'n_clicks')],
    [dash.dependencies.State('keyword-dropdown', 'value')]
)

#create function that autosuggests words/phrases?
#def word_suggestor

#create function that updates figure
def update_graph(n_clicks, keywords):  
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
    keyword_bydate_df = keyword_bydate_df.melt(id_vars = ['Complete Loc Dt'], var_name = 'Keyword', value_name = 'Service Calls')
    keyword_bydate_df = keyword_bydate_df.rename(columns = {'Complete Loc Dt': 'Date'})

    #create the histogram here
    fig = px.histogram(keyword_bydate_df, x = 'Date', y = 'Service Calls', color = 'Keyword', barmode = 'group', title = 'Phrase Presence Over Time')
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
    