import re
import unicodedata
from collections import Counter
import time

import requests
from bs4 import BeautifulSoup

from datetime import datetime, timedelta, date

import pandas as pd
import numpy as np

import psycopg2
from psycopg2.extras import execute_values

import streamlit as st
# import altair as alt

from datetime import datetime, timedelta, date

# import nltk
from nltk.corpus import stopwords

import spacy

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import torch


### DATABASE CONNECTION
USER = 'postgres.viwputoyddcvgcvvkfzb'
PW = 'FADSsummer2024!'
HOST = 'aws-0-us-east-1.pooler.supabase.com'
PORT = 6543
DB = 'postgres'

english_stopwords = set(stopwords.words('english'))
nlp = spacy.load("en_core_web_sm")

st.set_page_config(
    page_title='Complinace Terms App',
    layout='wide',
    initial_sidebar_state='collapsed',
    # menu_items={'Get Help': 'https://www.google.com/',
    #             'Report a bug': 'https://www.someresource.com/bug',
    #             'About': '# This is a header. This is an *some* app!'}
)

def clean_text(doc):

    # normalize Text
    doc = doc.lower()

    # remove unnecessary whitespaces
    doc = re.sub('\s+', ' ', doc)
    doc = doc.strip()
    
    # remove html tags
    doc = re.sub('<.*?>', '', doc)
    
    # remove email addresses
    doc = re.sub(r'([a-z0-9+._-]+@[a-z0-9+._-]+\.[a-z0-9+_-]+)', '', doc)
    
    # remove url
    doc = re.sub(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', '', doc)
    
    # remove accented characters
    doc = unicodedata.normalize('NFKD', doc).encode('ascii', 'ignore').decode('utf-8', 'ignore')

    # remove special symbols/punctuation
    doc = re.sub(r'[^\w ]+', '', doc)

    # remove stopwords
    doc = ' '.join([word for word in doc.split() if word not in english_stopwords])

    # lemmatization
    text = []
    for tok in nlp(doc):
        text.append(tok.lemma_)    
    
    doc = ' '.join(text)

    return doc

def intro():

    st.write("# Welcome to DOJ Compliance Trends")
    st.sidebar.success("Select a page above.")

    st.markdown(
        """
        **DOJ Compliance Trends** is collaboration project for the FADS Program Summer 2024 with Indiana University.
        This project was done using an open-source software and app is built on the Stremlit framework.
        Project Team:
         - Wes Martin
         - Jagadeesh Kovi
         - Sreya Kolachalama
         - Professor Haugh, Todd
        
        ### Want to learn more about IU Data Science Program?
        - Visit IU https://datascience.indiana.edu/programs/ms-data-science-online/index.html
        """
    )

def keyword_plots():

    st.markdown(f'# compliance keywords plots')
    st.write(
        """
        This page illustrates two plots:
        1. A histogram of the keywords identified in prediction model
        """
    )
    
    # FUNCTIONS
    def get_true_preds(df, col, val):
        mask = df[col] == val
        temp_df = df[mask].copy()
        
        temp_df.reset_index(drop=True, inplace=True)
        
        return temp_df
    
    def get_cutoff_date(days_ago):
        selected_date = (datetime.today() - timedelta(days=days_ago)).strftime('%D')
    
        return selected_date

    conn = psycopg2.connect(dbname=DB, user=USER, password=PW, host=HOST, port=PORT)
    curs = conn.cursor()
    curs.execute('SELECT * FROM doj_scrape_data')

    cols = [desc[0] for desc in curs.description]
    
    if 'df_doj_scraped' not in st.session_state:
        st.session_state.df_doj_scraped = pd.DataFrame(curs.fetchall(), columns=cols)
    
    curs.close()
    conn.close()

    df_doj_scraped = st.session_state.df_doj_scraped
    df_doj_scraped['model_pred'] = pd.to_numeric(df_doj_scraped['model_pred'])

    
    min_date = df_doj_scraped['date_published'].min()
    max_date = df_doj_scraped['date_published'].max()

    expander_label = 'Keywords Found for ' + min_date.strftime('%B %Y') + '-' + max_date.strftime('%B %Y')
    with st.expander(expander_label):
        # Function call to filter true predictions on Bert model
        filtered_df = get_true_preds(df_doj_scraped, 'model_pred', 1)
        
        df_keywords = pd.read_csv('data/compliance_keywords.csv')
        df_keywords['term_lemmas'] = df_keywords['compliance_terms'].apply(clean_text)
        df_keywords.drop_duplicates(inplace=True)

        # Initalize counter column
        df_keywords['counter'] = 0
        df_keywords['row_indexer'] = [[] for _ in range(df_keywords.shape[0])]

        # Count word frequencies
        for t, text_row in filtered_df.iterrows():
            text = text_row['cleaned_title_summary']
            
            c = Counter(text.split())
            text_words_list = list(c.keys())
            
            for k, keyword_row in df_keywords.iterrows():
                keyword = keyword_row['term_lemmas']
                
                if keyword in text_words_list:            
                    df_keywords.loc[k, 'counter'] += 1
                    df_keywords.loc[k, 'row_indexer'].append(t)

        # Create dataframe for matched words
        df_matched_keywords = df_keywords[df_keywords['counter'] > 0].copy()
        df_matched_keywords.reset_index(drop=True, inplace=True)
        
        # Define list variables
        terms = list(df_matched_keywords['compliance_terms'])
        counts = list(df_matched_keywords['counter'])
        df_vals = pd.DataFrame({'Compliance Words': terms, 'Frequency': counts,})

        str_min_date = min_date.strftime('%B %Y')
        str_max_date = max_date.strftime('%B %Y')
        
        ### Bar Chart of matched keywords
        st.header('Compliance Words: ' + str_min_date + ' - ' + str_max_date,
                divider='blue')
        
        color = st.color_picker('Color', '#FF0000')
        st.divider()

        st.bar_chart(df_vals, x='Compliance Words', y='Frequency', color=color, height=400)
        
        # Initalize plot object
        counter = {}
        date_freq = {}
        date_keys = filtered_df['date_published'].unique()

        for k in reversed(date_keys):
            str_date = str(k)
            date_freq[str_date] = 0

        # Create plot object
        for i, df in df_matched_keywords.iterrows():
            
            row_list = df['row_indexer']
            term = df['compliance_terms']
            dates = date_freq.copy()
            counter[term] = dates
            
            for x in range(len(row_list)):

                ref_idx = int(row_list[x])

                d0 = filtered_df['date_published'][ref_idx]
                str_date = str(d0)
                
                date_idx = list(date_freq.keys()).index(str_date)
                counter[term][str_date] += 1
        
        # Create dataframe from plot object
        df_plot = pd.DataFrame(counter)
        df_plot = df_plot.loc[(df_plot!=0).any(axis=1)].copy()
        df_plot.index = pd.to_datetime(df_plot.index)


    st.write(
        """
        2. Frequency over time up two keywords at time comparison
        """
    )

    range_selector = {7: 'Past 7 Days', 30:'Past 30 Days', 90: 'Past 90 Days', 
                      365: 'Past 12 Months', 0: min_date.strftime('%B %Y - present')}
    
    group_selector = {7: 'D', 30:'D', 90: 'ME', 365: 'QE', 0: 'D'}
    
    # Date range by keyword(s)
    st.header('Data Range', divider='blue')
    view_range_option = st.selectbox(
        'Select View Range',
        list(range_selector.values()),
        index=0,
        label_visibility='hidden'
    )

    selected_index = list(range_selector.values()).index(view_range_option)
    selected_key = list(range_selector.keys())[selected_index]
    
    selected_date = get_cutoff_date(selected_key)
    group_param = list(group_selector.values())[selected_index]

    if selected_key != 0:
        temp_df = df_plot.loc[(df_plot.index >= selected_date)]
    else:
        temp_df = df_plot.loc[(df_plot.index <= selected_date)]

    grouped_data = list(temp_df.groupby(pd.Grouper(freq=group_param)))
    group_counter = {}
    group_labels = []
    format = '%D'

    if group_param == 'ME':
        format = '%B %Y'

    for i in range(len(grouped_data)):

        group_date = grouped_data[i][0]
        group_str_date = grouped_data[i][0].strftime(format)
        
        keys = grouped_data[i][1].keys()
        vals = grouped_data[i][1].sum(axis=0)

        results = {keys[i]: vals.iloc[i] for i in range(len(keys))}
        
        group_counter[group_date] = results
        group_labels.append(group_str_date)

    temp_df_plot = pd.DataFrame(group_counter).T

    if len(temp_df_plot.index):
        freq_min_date = temp_df_plot.index.min().strftime('%B %Y')
        freq_max_date = temp_df_plot.index.max().strftime('%B %Y')
    else:
        date_today = date.today()
        date_previous = date_today - timedelta(days=selected_key)

        freq_min_date = date_previous.strftime('%B %Y')
        freq_max_date = date_today.strftime('%B %Y')

    # Line chart plot
    st.header('Frequency over time: ' + freq_min_date + ' - ' + freq_max_date,
              divider='blue')
            
    keyword_options = list(temp_df_plot.columns)

    selected_keywords = st.multiselect(
        "What are your favorite colors",
        keyword_options,
        max_selections=2,
        placeholder='Choose keyords',
        label_visibility='hidden'
    )

    if selected_keywords:
        keyword_plot = temp_df_plot[selected_keywords]
        st.line_chart(keyword_plot)


    # progress_bar = st.sidebar.progress(0)
    # status_text = st.sidebar.empty()
    # last_rows = np.random.randn(1, 1)
    # chart = st.line_chart(last_rows)

    # for i in range(1, 101):
    #     new_rows = last_rows[-1, :] + np.random.randn(5, 1).cumsum(axis=0)
    #     status_text.text("%i%% Complete" % i)
    #     chart.add_rows(new_rows)
    #     progress_bar.progress(i)
    #     last_rows = new_rows
    #     time.sleep(0.05)

    # progress_bar.empty()

    # # Streamlit widgets automatically run the script from top to bottom. Since
    # # this button is not connected to any other logic, it just causes a plain
    # # rerun.
    # st.button("Re-run")

def doj_web_scraper():

    st.markdown(f'# doj web scraper')
   
    # FUNCTIONS
    def doj_base_url(start_date, end_date) -> str:
        base_url = f'https://www.justice.gov/news/press-releases?search_api_fulltext=+&start_date={start_date}&end_date={end_date}&sort_by=field_date'

        return base_url

    def doj_pagination_counter(base_url) -> int:
        
        website = requests.get(base_url)
        soup = BeautifulSoup(website.content, 'html.parser')
        
        pagination = soup.find('ul', {'class': 'usa-pagination__list js-pager__items'})
        pages = pagination.findChildren(recursive=False)
        
        max_page_num = 0
        # iterate <li> tags to get the max number of pages returned by date values
        for i, page in enumerate(pages):
            if i == len(pages) - 1:
                a = page.find('a')['href']
                idx = a.index('page=')
                max_page_num = a[idx:].replace('page=', '')
                max_page_num = int(max_page_num)

        return max_page_num

    def get_doj_press_releases(base_url, max_page_num) -> pd.DataFrame:
        
        # Add a placeholder
        latest_iteration = st.empty()
        bar = st.progress(0)

        # Intialize list
        feed_data = []

        # Iterate press release pages
        for i in range(max_page_num + 1):
            page_url = base_url + '&page=' + str(i)
            page = requests.get(page_url)

            # Instantiate bs4 object
            soup = BeautifulSoup(page.content, 'html.parser')
            
            # find summary tag(s)
            articles = soup.find('div', {'class': 'rows-wrapper'})
            articles = articles.findChildren(recursive=False)

            # Get article title and summaries
            for article in articles:

                title = article.find('a').text.strip()    
                summary = article.find('p')
                
                if summary:
                    summary = summary.text.strip()
                else:
                    summary = np.nan
                
                url = 'https://www.justice.gov/' + article.find('a')['href']
                date = pd.to_datetime(article.find('time')['datetime']).date()
        
                # Append feed data objects
                feed_data.append({
                    'article_title': title,
                    'article_summary': summary,
                    'article_url': url,
                    'date_published': date
                })

            latest_iteration.text(f'DOJ news pages {i + 1} of {max_page_num + 1}')
            bar.progress(int(100 * i / max_page_num))
            time.sleep(0.1)
        
        df = pd.DataFrame(feed_data)
        
        st.success('Success! Number of articles retrieved: ' + str(len(df)))
        
        return df

    def clean_text(doc):
    
        # normalize Text
        doc = doc.lower()

        # remove unnecessary whitespaces
        doc = re.sub('\s+', ' ', doc)
        doc = doc.strip()
        
        # remove html tags
        doc = re.sub('<.*?>', '', doc)
        
        # remove email addresses
        doc = re.sub(r'([a-z0-9+._-]+@[a-z0-9+._-]+\.[a-z0-9+_-]+)', '', doc)
        
        # remove url
        doc = re.sub(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', '', doc)
        
        # remove accented characters
        doc = unicodedata.normalize('NFKD', doc).encode('ascii', 'ignore').decode('utf-8', 'ignore')

        # remove special symbols/punctuation
        doc = re.sub(r'[^\w ]+', '', doc)

        # remove stopwords
        doc = ' '.join([word for word in doc.split() if word not in english_stopwords])

        # lemmatization
        text = []
        for tok in nlp(doc):
            text.append(tok.lemma_)    
        
        doc = ' '.join(text)

        return doc
    
    def model_infer(checkpoint, text_list) -> list:
    
        # Load pretrained tokenizer
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        
        # Load pretrained model
        finetuned_model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
        
        # List to hold classifed text label booleans and score probabilties 
        classifications = []
        
        # Iterate title/summary texts
        for text in text_list:
        
            inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
            
            outputs = finetuned_model(**inputs)
        
            predicted_class_id = outputs.logits.argmax().item()
            
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
            score = np.round(predictions[predicted_class_id].item(), 5)
            label = finetuned_model.config.id2label[predicted_class_id]
            
            classification = {'label': label, 'score': score}
            classifications.append(classification)
        
        # print(f'Number of classifications: {len(classifications)}')

        return classifications
    
    def make_classifiction_list(classifications) -> list:

        pos_class = 0
        neg_class = 0
        classification_list = []
        
        for classfication in classifications:
            
            label = classfication['label']
            if label == 'TRUE':
                pos_class += 1
            else:
                neg_class += 1
        
            classification_list.append(label)
            
        # print(f'Number of positive classes: {pos_class}')
        # print(f'Number of negative classes: {neg_class}')

        return classification_list

    ### DATABASE CONNECTION
    DB = 'compliance_db'
    USER = 'postgres'
    PW = 'admin'
    HOST = 'localhost'
    PORT = 5432
    
    date_today = date.today()

    conn = psycopg2.connect(dbname=DB, user=USER, password=PW, host=HOST, port=PORT)
    curs = conn.cursor()
    curs.execute('SELECT * FROM doj_scrape_data')

    cols = [desc[0] for desc in curs.description]
    df_doj_scraped = pd.DataFrame(curs.fetchall(), columns=cols)
    row_count = curs.rowcount
    
    curs.close()
    conn.close()

    # Check if database contains records
    if len(df_doj_scraped) > 0:
        last_doj_scrape = df_doj_scraped['date_published'].max()
    else:
        last_doj_scrape = datetime.strptime(date_today, '%m-%d-%Y').date()

    # DOJ data is more than a week old
    if last_doj_scrape + timedelta(days=7) < date_today:
        
        day_diff = date_today - last_doj_scrape
        st.write(f'## DOJ press release data is _' + str(day_diff.days) + '_ days old')

        st.write(f'''
                 - This page gets all Department of Justice press releases within the specified date range
                    - The retrieved data will be evaluated as being compliance related using a fine-tuned pre-trained distilbert model
                    - After the model predictions are completed the database will insert the new records <br>
                 
                 Data courtesy of the **Department of Justice** https://www.justice.gov.
                 '''
                 )

        if 'btn_get_more_data' not in st.session_state:
            st.session_state.btn_get_more_data = False
        
        def click_update_button():
            st.session_state.btn_get_more_data = not st.session_state.btn_get_more_data

        # Date picker
        col1, col2 = st.columns(2)
        with col1:
            st.write('### Start')
            min_val = last_doj_scrape + timedelta(days=1)
            start_date = st.date_input('Select Start', value=min_val, 
                                       min_value=min_val, 
                                       max_value=date_today,
                                       key='start_date',
                                       format='MM/DD/YYYY',
                                       label_visibility='hidden')

        with col2:
            st.write('### End')
            end_date = st.date_input('Select End', value="default_value_today", 
                                     min_value=last_doj_scrape + timedelta(days=1), 
                                     max_value=date_today,
                                     key='end_date',
                                     format="MM/DD/YYYY",
                                     label_visibility='hidden')
        
        # Submit button
        col1, col2, col3 = st.columns(3)
        with col1:
            pass        
        with col2:
            st.button('Get more DOJ data?', type='primary', on_click=click_update_button)
        with col3:
            pass 
        

        if st.session_state.btn_get_more_data:
            
            try:
                # Function calls to get DOJ press release scrape parameters
                base_url = doj_base_url(start_date + timedelta(days=-7), end_date)
                max_page_num = doj_pagination_counter(base_url)

                df_insert = get_doj_press_releases(base_url, max_page_num)
                # df_insert['model_pred'] = None

                # Drop duplicate rows by article title and url
                unique_cols = ['article_title', 'article_url']
                df_insert.drop_duplicates(subset=unique_cols, keep='last', inplace=True)

                # Combine title and summary
                df_insert['title_summary'] = df_insert['article_title'].astype(str) + \
                            " " + df_insert['article_summary'].astype(str)

                # Function call to clean itle and summary text
                df_insert['cleaned_title_summary'] = df_insert['title_summary'].apply(clean_text)
                df_insert.drop(columns=['title_summary'], inplace=True)
                
                # Reorder columns to match database schema
                col = df_insert.pop('cleaned_title_summary')
                df_insert.insert(2, col.name, col)

                st.text(f'Getting model predictions')
                # Set title/summary column to list
                titles_summaries = df_insert['cleaned_title_summary'].tolist()

                # Cast list items to strings
                texts = [str(summary) for summary in titles_summaries]
                
                # Set model reference
                finetuned_model = 'finetuned_distilbert_model/'

                # Get model predictions
                classifications = model_infer(checkpoint=finetuned_model, text_list=texts)
                classify_list = make_classifiction_list(classifications)
                # Add column to dataframe
                df_insert['model_pred'] = classify_list
                
                # Convert classifyifcations to bit values
                str_to_bit = {'TRUE': '1', 'FALSE': '0'}
                df_insert['model_pred'] = df_insert['model_pred'].map(str_to_bit)

                # Sort dataframe by published date
                df_insert.sort_values(by=['date_published'], inplace=True)
                df_insert['date_published'] = df_insert['date_published'].astype(str)

                st.text(f'Inserting records into database')
                
                row_ids = [i + 1 + row_count for i in range(len(df_insert))]
                df_insert['row_id'] = row_ids
                col = df_insert.pop('row_id')
                df_insert.insert(0, col.name, col)
                
                # create columsn (col1,col2,...)
                df_columns = list(df_insert)
                columns = ','.join(df_columns)

                # create VALUES('%s', '%s",...) one '%s' per column
                values = 'VALUES({})'.format(','.join(['%s' for _ in df_columns])) 

                #Create INSERT INTO statement
                sql_insert = 'INSERT INTO {} ({}) {} ON CONFLICT DO NOTHING'.format('doj_scrape_data', columns, values)
                                
                # Establish connection
                conn = psycopg2.connect(dbname=DB, user=USER, password=PW, host=HOST, port=PORT)
                curs = conn.cursor()

                # Execute SQL insert statment
                psycopg2.extras.execute_batch(curs, sql_insert, df_insert.values)
                conn.commit()

                # Select new row count
                curs.execute('SELECT * FROM doj_scrape_data')
                row_count = curs.rowcount - row_count
                curs.close()
                conn.close()
                
                # Check number of rows updated
                if row_count > 0:
                    st.success('Success! Inserted ' + str(row_count) + ' article into database.')
                else:
                    st.warning('No new database records were found.')
                
            except AttributeError:
                st.warning('Not enough data found to scrape')

def modify_data():

    st.markdown(f'# modifiy data')
    st.write(
        """
        This page provides the ability to update the model predictions, keywords, etc.
        """
    )

    def click_view_button():
        st.session_state.view_preds = not st.session_state.view_preds
    
    st.write('Click to view database table.')

    if 'view_preds' not in st.session_state:
        st.session_state.view_preds = False

    st.button('View Preds', on_click=click_view_button)

    if st.session_state.view_preds:
        
        conn = psycopg2.connect(dbname=DB, user=USER, password=PW, host=HOST, port=PORT)
        curs = conn.cursor()
        curs.execute('SELECT * FROM doj_scrape_data')
        
        # Initialize session state with dataframes
        # Include initialization of "edited" slots by copying originals
        if 'df1' not in st.session_state:
            cols = [desc[0] for desc in curs.description]
            
            # Set dataframe
            df = pd.DataFrame(curs.fetchall(), columns=cols)
            d = {False: 'false', True: 'true'}
            df['model_pred'] = df['model_pred'].map(d)
            
            # Sort dataframe by publish date
            df.sort_values(by=['row_id'], inplace=True)
            
            st.session_state.df1 = df
            st.session_state.edited_df1 = st.session_state.df1.copy()
        
        # Convenient shorthand notation
        df1 = st.session_state.df1

        curs.close()
        conn.close()

        if 'btn_disbale' not in st.session_state:
            st.session_state.btn_disbale = True
        
        def button_toggle():
            st.session_state.btn_disbale = False

        st.session_state.edited_df1 = st.data_editor(
            df1,
            column_config = {
                'row_id': None,
                'article_summary': None,
                'cleaned_title_summary': None,
                'article_url': None,
                'date_published': 'date',
                'model_pred': 'pred',

                'modify_pred': st.column_config.CheckboxColumn(
                    'reclass',
                    help="Select row to reclassify",
                    default=False,
                ),
            },
            hide_index=True,
            on_change=button_toggle
        )


        if 'btn_update_preds' not in st.session_state:
            st.session_state.btn_update_preds = False

        def click_update_button():
            st.session_state.btn_update_preds = not st.session_state.btn_update_preds
            st.session_state.btn_disbale = not st.session_state.btn_disbale

        st.button('Update data', disabled=st.session_state.btn_disbale, on_click=click_update_button)

        if st.session_state.btn_update_preds:

            df_update_preds = st.session_state.edited_df1

            conn = psycopg2.connect(dbname=DB, user=USER, password=PW, host=HOST, port=PORT)
            conn.autocommit = True
            
            sql = """
                UPDATE doj_scrape_data d
                SET modify_pred = t.modify_pred
                FROM (values %s) AS t(row_id, modify_pred)
                WHERE d.row_id = t.row_id;
            """
            
            rows_to_update = df_update_preds[['row_id', 'modify_pred']].copy()
            rows_to_update.dropna(inplace=True)
            rows_to_update = list(rows_to_update[['row_id', 'modify_pred']].itertuples(index=False, name=None))
            
            curs = conn.cursor()
            execute_values(curs, sql, rows_to_update)
            curs.close()
            conn.commit()
            conn.close()
            
            st.success('Record added Successfully')

            st.session_state.btn_update_preds = False            
            del st.session_state.edited_df1
            del st.session_state.df1
            

page_names_to_funcs = {
    '-': intro,
    'Keyword Plots': keyword_plots,
    'DOJ Web Scraper': doj_web_scraper,
    'Modify Data': modify_data
}

page_name = st.sidebar.selectbox('Choose a page', page_names_to_funcs.keys())
page_names_to_funcs[page_name]()