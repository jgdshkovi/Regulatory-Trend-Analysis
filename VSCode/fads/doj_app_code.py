
# IMPORTS
import streamlit as st
import pandas as pd
import numpy as np
import ast

from datetime import datetime

import psycopg2
from psycopg2.extras import execute_values


# FUNCTIONS
def get_date_range(df , col) -> str:
      # convert the date column to datetime format
    df[col] = pd.to_datetime(df[col])
    # change the datetime format
    df['date_formatted'] = df[col].dt.strftime('%m/%d/%Y')

    # date_val = df['date_formatted'].unique()[0]
    min_date = df['date_formatted'].min()
    max_date = df['date_formatted'].max()

    df.drop(['date_formatted'], axis=1)

    return (min_date, max_date)


def get_true_preds(df, col, val):
    mask = df[col] == val
    temp_df = df[mask].copy()
    
    temp_df.reset_index(drop=True, inplace=True)
    
    return temp_df


# CALL MAIN
if __name__ == "__main__":

    st.set_page_config(
        page_title='Complinace Terms App',
        layout='wide',
        initial_sidebar_state='collapsed',
        # menu_items={'Get Help': 'https://www.extremelycoolapp.com/help',
        #             'Report a bug': 'https://www.extremelycoolapp.com/bug',
        #             'About': '# This is a header. This is an *extremely* cool app!'}
    )

    # Dataframes
    df_preds = pd.read_csv('doj_preds.csv')
    # Get min and max dates formatted (mm/dd/yyyy)
    min_date, max_date = get_date_range(df_preds, 'date_published')


    df_matched_keywords = pd.read_csv('keywords_found.csv')
    # Convert string back to list (post saving dataframe as csv)
    df_matched_keywords['row_indexer'] = df_matched_keywords['row_indexer'].map(ast.literal_eval)
    terms = list(df_matched_keywords['compliance_terms'])
    counts = list(df_matched_keywords['counter'])
    df_vals = pd.DataFrame({'Compliance Words': terms, 
                            'Frequency': counts,})

    def click_view_button():
        st.session_state.view_preds = not st.session_state.view_preds
    
    with st.sidebar:
        st.write('Click to view database table.')

        if 'view_preds' not in st.session_state:
            st.session_state.view_preds = False

        st.button('View Preds', on_click=click_view_button)

    if st.session_state.view_preds:
        ### DATABASE CONNECTION
        DB = 'compliance_db'
        USER = 'postgres'
        PW = 'admin'
        HOST = 'localhost'
        PORT = 5432

        conn = psycopg2.connect(dbname=DB, user=USER, password=PW, host=HOST, port=PORT)
        curs = conn.cursor()
        curs.execute('SELECT * FROM mytable')
        test_df = pd.DataFrame(curs.fetchall(), columns=['name', 'pet', 'is_cool'])
        curs.close()
        conn.close()

        # Initialize session state with dataframes
        # Include initialization of "edited" slots by copying originals
        if 'df1' not in st.session_state:
            st.session_state.df1 = test_df

            st.session_state.edited_df1 = st.session_state.df1.copy()

        # Convenient shorthand notation
        df1 = st.session_state.df1

        # Page functions commit edits in real time to "editied" slots in session state
        def df_updates():
            st.session_state.edited_df1 = st.data_editor(
                df1,
                column_config = {
                    'is_cool': st.column_config.CheckboxColumn(
                        'is_cool',
                        help="Select row to reclassify",
                        default=False,
                    )
                },
                hide_index=True,
            )
            return st.session_state.edited_df1
        
        df_update_preds = df_updates()

        if 'btn_update_preds' not in st.session_state:
            st.session_state.btn_update_preds = False

        def click_update_button():
            st.session_state.btn_update_preds = not st.session_state.btn_update_preds

        st.button('Update data', on_click=click_update_button)

        if st.session_state.btn_update_preds:
            # The message and nested widget will remain on the page
            # st.write('Button is on!')

            conn = psycopg2.connect(dbname=DB, user=USER, password=PW, host=HOST, port=PORT)
            conn.autocommit = True
            
            sql = """
                UPDATE mytable m
                SET 
                    is_cool = CAST(t.is_cool AS BOOLEAN)
                FROM (values %s) AS t(name, pet, is_cool)
                WHERE m.name = t.name;
            """
            rows_to_update = list(df_update_preds.itertuples(index=False, name=None))

            curs = conn.cursor()
            execute_values(curs, sql, rows_to_update)
            curs.close()
            conn.close()
            
            st.success('Record added Successfully')
            
        # else:
        #     st.write('Button is off!')


    ### Bar Chart of matched keywords
    st.header('Compliance Words: ' + min_date + ' - ' + max_date)    
    color = st.color_picker('Color', '#FF0000')
    st.divider()
    st.bar_chart(df_vals, x='Compliance Words', y='Frequency', color=color, height=400)

    # ## Number of days bewteen two dates
    # d0 = date_object = datetime.strptime(min_date, '%m/%d/%Y').date()
    # d1 = date_object = datetime.strptime(max_date, '%m/%d/%Y').date()
    # delta = (d1 - d0)
    # days = delta.days

    df_vals = pd.DataFrame({'Compliance Words': terms, 
                            'Frequency': counts,})
    
    
    ### Line Chart of keyword frequencies
    df_preds = pd.read_csv('doj_preds.csv')

    filtered_df = get_true_preds(df_preds, 'bert_preds', True)

    counter = {}
    date_freq = {}
    date_keys = filtered_df['date_published'].unique()

    for k in reversed(date_keys):
        str_date = str(k)
        date_freq[str_date] = 0


    for i, df in df_matched_keywords.iterrows():
        
        row_list = df['row_indexer']
        term = df['compliance_terms']
        dates = date_freq.copy()
        counter[term] = dates
        
        for x in range(len(row_list)):

            ref_idx = int(row_list[x])

            d0 = filtered_df['date_published'][ref_idx]
            date_format = '%Y-%m-%d'
            date_obj = datetime.strptime(d0, date_format).date()
            str_date = str(date_obj)
            
            date_idx = list(date_freq.keys()).index(str_date)
            counter[term][str_date] += 1
         
    df_plot = pd.DataFrame(counter)
    df_plot = df_plot.loc[(df_plot!=0).any(axis=1)]

    st.header('Frequency over time: ' + min_date + ' - ' + max_date)   
    # st.line_chart(df_plot)
    keyword_option_select = st.multiselect('Select keyword(s) to view data', df_plot.columns)
    df_plot = df_plot[keyword_option_select]
    st.line_chart(df_plot)
