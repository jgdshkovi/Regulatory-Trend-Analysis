
# IMPORTS
import time
import requests
from bs4 import BeautifulSoup

import streamlit as st
import pandas as pd
import numpy as np
import ast

# from datetime import datetime
from datetime import datetime, timedelta, date

import psycopg2
from psycopg2.extras import execute_values


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

# CALL MAIN
if __name__ == "__main__":

    ### DATABASE CONNECTION
    DB = 'compliance_db'
    USER = 'postgres'
    PW = 'admin'
    HOST = 'localhost'
    PORT = 5432

    conn = psycopg2.connect(dbname=DB, user=USER, password=PW, host=HOST, port=PORT)
    curs = conn.cursor()
    curs.execute('SELECT * FROM test_table')

    cols = [desc[0] for desc in curs.description]
    df_doj_scraped = pd.DataFrame(curs.fetchall(), columns=cols)
    row_count = curs.rowcount

    curs.close()
    conn.close()

    INIT_DATE = '04-30-2024'
    DATE_FORMAT = '%m-%d-%Y'
    # Check if database contains records
    if len(df_doj_scraped) > 0:
        last_doj_scrape = df_doj_scraped['date_published'].max()
    else:
        last_doj_scrape = datetime.strptime(INIT_DATE, DATE_FORMAT).date()

    date_today = date.today()

    # DOJ data is more than a week old
    if last_doj_scrape + timedelta(days=7) < date_today:
        
        day_diff = date_today - last_doj_scrape
        st.write('# DOJ data is ' + str(day_diff.days) + ' old')

        if 'btn_get_more_data' not in st.session_state:
            st.session_state.btn_get_more_data = False
        
        def click_update_button():
            st.session_state.btn_get_more_data = not st.session_state.btn_get_more_data

        st.button('Get more DOJ data?', type='primary', on_click=click_update_button)

        if st.session_state.btn_get_more_data:
            # Declare run dates
            start_date = '05/01/2024'
            end_date = '05/15/2024'
            
            # Function calls to get DOJ press release scrape parameters
            base_url = doj_base_url(start_date, end_date)
            max_page_num = doj_pagination_counter(base_url)

            df_insert = get_doj_press_releases(base_url, max_page_num)
            df_insert['model_prediction'] = None


            # Drop duplicate rows by article title and url
            unique_cols = ['article_title', 'article_url']
            df_insert.drop_duplicates(subset=unique_cols, keep='last', inplace=True)
            
            st.text(f'Inserting records into database')
            # create columsn (col1,col2,...)
            df_columns = list(df_insert)
            columns = ','.join(df_columns)
            # create VALUES('%s', '%s",...) one '%s' per column
            values = 'VALUES({})'.format(','.join(['%s' for _ in df_columns])) 

            #Create INSERT INTO statement
            sql_insert = 'INSERT INTO {} ({}) {} ON CONFLICT DO NOTHING'.format('test_table', columns, values)

            # Establish connection
            conn = psycopg2.connect(dbname=DB, user=USER, password=PW, host=HOST, port=PORT)
            curs = conn.cursor()
            
            # Execute SQL insert statment
            psycopg2.extras.execute_batch(curs, sql_insert, df_insert.values)
            conn.commit()

            curs.execute('SELECT * FROM test_table')
            row_count = curs.rowcount - row_count
            curs.close()
            conn.close()
            if row_count > 0:
                st.success('Success! Inserted ' + str(row_count) + ' article into database.')
            else:
                st.warning('No new database records were found.')
        
    # print(date_today.strftime('%m-%d-%Y'))

    # st.date_input("When's your birthday", value="default_value_today", 
    #               min_value=last_doj_scrape + timedelta(days=1), 
    #               max_value=date_today, 
    #               key=None, help=None,
    #               on_change=None, args=None, kwargs=None, 
    #               format="YYYY/MM/DD", disabled=False, label_visibility="visible")

        # if 'btn_update_preds' not in st.session_state:
        #     st.session_state.btn_update_preds = False

        # def click_update_button():
        #     st.session_state.btn_update_preds = not st.session_state.btn_update_preds

        # st.button('Update data', on_click=click_update_button)

        # if st.session_state.btn_update_preds:
        #     # The message and nested widget will remain on the page
        #     # st.write('Button is on!')

        #     conn = psycopg2.connect(dbname=DB, user=USER, password=PW, host=HOST, port=PORT)
        #     conn.autocommit = True
            
        #     sql = """
        #         UPDATE mytable m
        #         SET 
        #             is_cool = CAST(t.is_cool AS BOOLEAN)
        #         FROM (values %s) AS t(name, pet, is_cool)
        #         WHERE m.name = t.name;
        #     """
        #     rows_to_update = list(df_update_preds.itertuples(index=False, name=None))

        #     curs = conn.cursor()
        #     execute_values(curs, sql, rows_to_update)
        #     curs.close()
        #     conn.close()
            
        #     st.success('Record added Successfully')
            
        # # else:
        # #     st.write('Button is off!')