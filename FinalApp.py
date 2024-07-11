# %%
#first import the excel sheet
import pandas as pd
import plotly.express as px
from plotly import graph_objects as go
import ipywidgets as widgets
from IPython.display import display

import numpy as np
import spacy
from collections import Counter

import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output

import dash_bootstrap_components as dbc #used for formatting layout of the dash

file_path = "C:/Users/rgae/OneDrive - QuidelOrtho/Documents/All QuidelFiles/Excel Files/6-27-2024 2nd Copy .xlsx"
df = pd.read_excel(file_path, usecols = 'AP, AV, AZ')

# %%
df_filtered_AZ = df.loc[df['Action Taken Description'] == 'CLOSE A CALL'].copy()

#Verify the length of the DataFrame
length_of_df = len(df_filtered_AZ)

# %%
#tokenize the text by combining it into one column for each row
#load nlp model
nlp = spacy.load("en_core_web_sm")

#use pd to date time to utilize in the mapping of the data over time
df_filtered_AZ['Incident Close Loc Dt'] = pd.to_datetime(df_filtered_AZ['Incident Close Loc Dt'])
#Combine the text columns into a single text column for analysis
df_filtered_AZ['combined_text'] = df_filtered_AZ.astype(str).agg(' '.join, axis = 1)

#analyze column 'Problem Dsc'
def tokenize(text):
    doc = nlp(text.lower())
    return ' '.join([token.text for token in doc]) #https://stackoverflow.com/questions/57187116/how-to-modify-spacy-tokens-doc-doc-tokens-with-pipeline-components-in-spacy

#df_filtered_AZ['tokenized_text'] = df_filtered_AZ['Problem Dsc'].apply(tokenize) #call the method above to the combined text column 
df_filtered_AZ.loc[:,'tokenized_text'] = df_filtered_AZ['Problem Dsc'].apply(tokenize)

display(df_filtered_AZ)

# %%
#vectorize the token to find phrases
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(ngram_range = (1, 5))  #Bigrams and trigrams (how long each phrase can be i.e. 2 and 3)
X = vectorizer.fit_transform(df_filtered_AZ['tokenized_text'])

#Sum up the counts of each phrase
phrase_counts = (X > 0).sum(axis=0).A1

phrases = vectorizer.get_feature_names_out()

phrase_counts_df = pd.DataFrame({'Phrase': phrases, 'Count': phrase_counts}).sort_values(by = 'Count', ascending = False).reset_index()
phrase_counts_df = phrase_counts_df.drop(['index'], axis = 1)
#phrase_counts_df = phrase_counts_df.drop(phrase_counts_df[phrase_counts_df['Count'] <= 10].index)
#display(phrase_counts_df)

# %%
from dash.dependencies import Input, Output, State
import random

def check_keywords(text, keywords):
    return int(any(keyword in text for keyword in keywords))


def count_keywords(text, keywords):
    text = text.lower()
    return {keyword: 1 if keyword in text else 0 for keyword in keywords} #binary values


#create dash app here
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = html.Div([
    dbc.Row([
        dbc.Col([
            html.H3("Phrase Analysis"),
            dcc.Dropdown(
                id='keyword-dropdown',
                options = [{'label': f"{phrase} ----- {freq}", 'value': phrase} for phrase, freq in zip(phrase_counts_df['Phrase'], phrase_counts_df['Count'])],
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
        ], width = {'size': '8'},
            style = {'font-size': '16px', 'text-align':'left'}),
        dbc.Col([
            html.H2("Toggle Stack", style={'font-size': '20px'}),
            dcc.Checklist(
                id = 'toggle-checklist',
                options = [
                    {'label': 'Grouped', 'value': 'group'},
                    {'label': 'Percentage', 'value': 'percentage'}
                ], value = ['True'],
                style = {'font-size': '16px'},
            )
        ],style = {'text-align': 'right'})
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id = 'keyword-graph')
        ]),
    ]),

    dbc.Row([
        #Input keywords for the overlap observation
        dbc.Col([ 
            html.H5("Input Keyword to Observe Overlap (comma separated)"),
            dcc.Dropdown(
                id = "input_2",
                options = [{'label': f"{phrase} ----- {freq}", 'value': phrase} for phrase, freq in zip(phrase_counts_df['Phrase'], phrase_counts_df['Count'])],
                multi = True,
                placeholder = "Select keywords"
            )
        ], width = {"size":6}), 

    ]),


    dbc.Row([
        dbc.Col([ #create an output for the pairs
            html.H6("Percentage for Pair 1 and 2"),
            html.Div(id = "output_pair_1_2")
        ], width = {"size":6}),
    ]),
    
    dbc.Row([
        dbc.Col([
            html.H6("Recommended Phrases for Pair 1 and 2"),
            html.Div(id = "recommendation_1_2")
        ], width = {"size": 6}),
        dbc.Col([
            html.H6("Example of Raw Cell Data"),
            #create a slider with min 0 and max 4 (5 total) and iterate for each lens to change values on callback for each value changed
            dcc.Slider(id = 'raw-text-slider', min = 0, max = 4, step = 1, value = 0, marks = {i: str(i+1) for i in range(5)}),
            html.Div(id = 'raw-text-display')
        ]),
    ]),

])



#create app callback and mark all inputs and output id's that needs to be updated
@app.callback(
    [Output(component_id = "output_pair_1_2", component_property = "children"),
    Output(component_id = "recommendation_1_2", component_property = "children")],
    [Input(component_id = 'submit-button', component_property = 'n_clicks'),
     Input(component_id = "keyword-dropdown", component_property = "value"),
    Input(component_id = "input_2", component_property = "value")]
)

def update_output(n_clicks, input_1, input_2):
    # Check if the submit button has been clicked
    if n_clicks is None or n_clicks == 0:
        return "N/A", "N/A"

    if not input_1 or not input_2:
        return "N/A", "N/A"
    else:
        keywords_1 = input_1
        keywords_2 = input_2

        #Apply the function to create binary columns
        df_filtered_AZ['keyword_1_present'] = df_filtered_AZ['tokenized_text'].apply(lambda text: check_keywords(text, keywords=keywords_1))
        df_filtered_AZ['keyword_2_present'] = df_filtered_AZ['tokenized_text'].apply(lambda text: check_keywords(text, keywords=keywords_2))

        #Calculate percentages
        keyword_percentage_pair_1 = (sum(df_filtered_AZ['keyword_1_present'] & df_filtered_AZ['keyword_2_present']) / sum(df_filtered_AZ['keyword_1_present'])) * 100

        #Get recommendations
        recommendation_1_2 = [phrase for phrase in phrase_counts_df['Phrase']
                          if any(keyword in phrase for keyword in keywords_1) and any(keyword in phrase for keyword in keywords_2)]

        recommendation_1_2 = recommendation_1_2[:10]
        recommendation_1_2_text = "\n".join(recommendation_1_2) if recommendation_1_2 else "No recommendations"

        return f"{keyword_percentage_pair_1:.2f}%", recommendation_1_2
    

#####################################################################################################

@app.callback(
    Output(component_id = 'keyword-graph', component_property = 'figure'),
    [Input(component_id = 'submit-button', component_property = 'n_clicks'),
    Input(component_id = 'toggle-checklist', component_property = 'value')],
    [dash.dependencies.State('keyword-dropdown', 'value')]
)

#create function that autosuggests words/phrases?
#def word_suggestor

#create function that updates figure
def update_graph(n_clicks, toggle, keywords):  
    #want it so that for each click, updates using property of n_clicks and changing the keywords input

    #consider the case that exist empty string
    if not keywords:
        #return empty dict if exist empty string
        return {}

    #apply the function here which applies strip() and lower() while splitting by ','
    df_filtered_AZ['keyword_frequency'] = df_filtered_AZ['combined_text'].apply(lambda x: count_keywords(x, keywords))

    #percentage =
    #convert back to dataframe and fill zero if missing(N/A)
    keyword_df = df_filtered_AZ['keyword_frequency'].apply(pd.Series)

    #group by date and reset the index like before
    keyword_bydate_df = df_filtered_AZ[['Incident Close Loc Dt']].join(keyword_df).groupby('Incident Close Loc Dt').sum().reset_index()
    keyword_bydate_df = keyword_bydate_df.melt(id_vars = ['Incident Close Loc Dt'], var_name = 'Keyword', value_name = 'Service Calls')
    keyword_bydate_df = keyword_bydate_df.rename(columns = {'Incident Close Loc Dt': 'Date'})

    yaxis_formatting = '2%' if 'percentage' in toggle else ' '

    barmode = 'group' if 'group' in toggle else 'stack'

    fig = px.histogram(keyword_bydate_df, x = 'Date', y = 'Service Calls', color = 'Keyword', barmode = barmode,  title='Phrase Presence Over Time')
    fig.update_xaxes(
        dtick=86400000.0 * 14,  # biweekly
        ticklabelmode="period"
    )
    fig.update_layout(yaxis_tickformat = yaxis_formatting, yaxis_title = 'Service Calls')


    #return the fig
    return fig



#######################################################################################################################################################


@app.callback(
    [Output(component_id = 'raw-text-display', component_property = 'children'),
     Output(component_id = 'raw-text-slider', component_property = 'max')],
    [Input(component_id= 'submit-button', component_property = 'n_clicks'),
     Input(component_id = 'raw-text-slider', component_property = 'value')],
    [State(component_id = 'keyword-dropdown', component_property ='value'),
     State(component_id = 'input_2', component_property = 'value')]
)

#method to update reommendation clel
def recommend_cell(n_clicks, slider_value, input_1, input_2):
    if not input_1 or not input_2:
        return "No keywords selected.", 0
    
    #consider for when submit button is clicked or not
    if n_clicks is None or n_clicks == 0:
        return "N/A", 0
    else:
        keywords_1 = input_1
        keywords_2 = input_2

        #Apply the function to create binary columns
        df_filtered_AZ['keyword_1_present'] = df_filtered_AZ['tokenized_text'].apply(lambda text: check_keywords(text, keywords_1))
        df_filtered_AZ['keyword_2_present'] = df_filtered_AZ['tokenized_text'].apply(lambda text: check_keywords(text, keywords_2))

        #Filter rows where both keywords are present
        df_filtered_both_keywords = df_filtered_AZ[(df_filtered_AZ['keyword_1_present'] == 1) & (df_filtered_AZ['keyword_2_present'] == 1)]

        #minimum number of generation is 5
        num_samples = min(5, len(df_filtered_both_keywords))
        random_indices = random.sample(range(len(df_filtered_both_keywords)), num_samples)

        #convert to a list and find the location of the random indices
        raw_texts = df_filtered_both_keywords['Problem Dsc'].iloc[random_indices].tolist()

        #minus length of raw_Texts for each slider value e.g. if 4 then subtract necessary amount to equal it
        if raw_texts:
            return raw_texts[slider_value], len(raw_texts) - 1
        else:
            return "No matching texts found.", 0



            

if __name__ == '__main__':
    app.run_server(debug = True, port = 8056)



