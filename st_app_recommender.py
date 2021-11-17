import os
import requests

import numpy as np
import pandas as pd
import streamlit as st 
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
from matplotlib.figure import Figure

from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

def main():

        st.set_page_config(
                page_title='Book Recommender System', 
                page_icon= ':books:',
                layout='wide', 
                initial_sidebar_state='expanded'
            )

        main_df, book_info, user_info, user_rating_info = load_data()

        wide_df, wide_df_sparse = create_wide_df(main_df)

        model_knn = create_model(wide_df_sparse)

        selected_box = st.sidebar.selectbox(
                'Select Section',
                ('Main', 'Book-Crossing Data')
            )
        
        if selected_box == 'Main':

                book_recommendation(main_df, wide_df, model_knn)

        if selected_box == 'Book-Crossing Data': 

                analysis(main_df, book_info, user_info, user_rating_info) 

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def load_data(arg_df = 'explicit'):

        book_info = pd.read_csv(os.getcwd() + '/data/book_info_unique.csv')
        user_info = pd.read_csv(os.getcwd() + '/data/user_info.csv')
        rating_info = pd.read_csv(os.getcwd() + '/data/rating_info.csv')

        # user_rating df for visualisation purposes, keep only users with explicit ratings
        user_rating_info = pd.merge(rating_info, user_info, on=['user_id'])
        user_rating_info = user_rating_info[ (user_rating_info['book_rating'] != 0) ]

        # main df
        main_df = pd.merge(rating_info, book_info, on=['isbn'])
        main_df = pd.merge(main_df, user_info, on=['user_id'])

        if arg_df == 'explicit':

                main_df = main_df[ (main_df['book_rating'] != 0) ]
        
        if arg_df == 'implicit':

                main_df.loc[ main_df['book_rating'] >= 0, 'book_rating'] = 1
        
        # add average book ratings column
        average_ratings = main_df.groupby('isbn_unique')['book_rating'].mean().round(2).reset_index()
        average_ratings.rename(columns={'book_rating': 'average_rating'}, inplace=True)
        main_df = main_df.merge(average_ratings, on='isbn_unique')

        # add number of book ratings column
        n_ratings = main_df.groupby('isbn_unique')['book_rating'].count().reset_index()
        n_ratings.rename(columns={'book_rating': 'n_book_ratings'}, inplace=True)
        main_df = main_df.merge(n_ratings, on='isbn_unique')

        # only keep books that have been rated at least certain amount of times (top 25% - third quartile - 32 OR mean - 36)
        main_df = main_df[main_df['n_book_ratings'] >= 32]

        return main_df, book_info, user_info, user_rating_info

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def create_wide_df(main_df):
        wide_df = main_df.pivot_table(
                columns='user_id', 
                index='isbn_unique', 
                values='book_rating'
            )

        wide_df.fillna(
                0, 
                inplace=True
            )

        wide_sparse = csr_matrix(wide_df)

        return wide_df, wide_sparse

def create_model(wide_sparse):

        model = NearestNeighbors(
                algorithm='brute',
                metric = 'cosine'
            )

        model.fit(
                wide_sparse
            )
        
        return model

def book_recommendation(df, wide_df, model):

        ### Row 1
        row_11, _ = st.columns( (1, 0.001)  )
        
        row_11.title('Book Recommender System :books:')
        row_11.write('A Web App by [Julius Rabek](https://github.com/murtagh97)')
        
        ### Row s1
        row_s1, _ = st.columns( (2, 1) )
        
        row_s1.write('')

        ### Row 2
        row_21, _ = st.columns( (1, 0.001) )
        
        df_unique = df.drop_duplicates(
                subset = 'book_title', keep="first"
            )

        book_title = df_unique['book_title'].unique()
        index_lotr = np.where(book_title == 'The Return of the King')[0][0]

        option = row_21.selectbox(
                'Select a book you like:',
                book_title,
                index= int(index_lotr)
            )

        ### add helper expander
        book_author = df_unique['book_author'].unique()
        index_jkr = np.where(book_author == 'J. K. Rowling')[0][0]
        exp_help = row_21.expander('Search book by author:')
        author_option = exp_help.selectbox(
                '',
                book_author,
                index= int(index_jkr)
            )
        sub_df_author = df[ df['book_author'].str.lower() == author_option.lower() ].drop_duplicates(subset=['isbn_unique'], keep = 'first')
        exp_help.write(
            sub_df_author[['book_author', 'book_title']].reset_index(drop=True)
        )

        row_21.subheader('Book Info')

        ### Row 3
        row_31, _, row_33, _ = st.columns( (1, 0.75, 1.5, 1) )

        title = df_unique['book_title'].loc[ df_unique['book_title'] == option ].item()
        isbn = df_unique['isbn_unique'].loc[ df_unique['book_title'] == option ].item()
        author = df_unique['book_author'].loc[ df_unique['book_title'] == option ].item()
        year = df_unique['publication_year'].loc[ df_unique['book_title'] == option ].item()
        publisher = df_unique['publisher'].loc[ df_unique['book_title'] == option ].item()
        url = df_unique['Image-URL-L'].loc[ df_unique['book_title'] == option ].item()
        avg_rating = df_unique['average_rating'].loc[ df_unique['book_title'] == option ].item()
        n_rating = df_unique['n_book_ratings'].loc[ df_unique['book_title'] == option ].item()

        row_31.image(
                Image.open(requests.get(url, stream=True).raw), 
                use_column_width=True
            )
        
        row_33.markdown(
                f' ** Title:  ** {title} \n'
                f'\n  ** Author:  ** {author} \n'
                f'\n ** Publisher:  ** {publisher} \n'
                f'\n ** Publication Year:  ** {year} \n'
                f'\n  ** ISBN:  ** {isbn} \n'
                f'\n <div style="display: inline-block; margin: 0px auto;border-style: solid;border-color: #646464;border-width: 0.5px; background: #323232;border-radius: 4px;padding: 5px;">'
                f'&#11088; <span style="color: #fff">{avg_rating} |</span><span style="color: #f53666; display: inline-block; margin-left: 2px">{n_rating}x</span></div>',
                unsafe_allow_html=True
            )

        ### Row 4
        row_41, _ = st.columns( (1, 0.001) )
        row_41.subheader('Recommendations')

        best_match_index = wide_df.index.tolist().index(isbn)

        _, recommendations = model.kneighbors(
                wide_df.iloc[best_match_index, :].values.reshape(1,-1), 
                n_neighbors = 5 + 1
            )
        
        title_list = []
        author_list = []
        url_list = []
        avg_rating_list = []
        n_rating_list = []

        for i in range(1, 6):
                title_list.append(
                        df_unique['book_title'].loc[ df_unique['isbn_unique'] == wide_df.index[recommendations.flatten()[i]] ].item()
                    )
                author_list.append(
                        df_unique['book_author'].loc[ df_unique['isbn_unique'] == wide_df.index[recommendations.flatten()[i]] ].item()
                    )
                url_list.append(
                        df_unique['Image-URL-L'].loc[ df_unique['isbn_unique'] == wide_df.index[recommendations.flatten()[i]] ].item()
                    )
                avg_rating_list.append(
                        df_unique['average_rating'].loc[ df_unique['isbn_unique'] == wide_df.index[recommendations.flatten()[i]] ].item()
                    )
                n_rating_list.append(
                        df_unique['n_book_ratings'].loc[ df_unique['isbn_unique'] == wide_df.index[recommendations.flatten()[i]] ].item()
                    )
    
        ### Row 5
        row_51, row_52, row_53, row_54, row_55 = st.columns( (1, 1, 1, 1, 1) )
        
        row_51.image(
                Image.open(requests.get(url_list[0], stream=True).raw),
                use_column_width=True
            )
        row_51.markdown(
                f' _{title_list[0]}_, written by _{author_list[0]}_ \n'
                f'<br>'
                f'\n <div style="display: flex; align-items: center; justify-content: center;"><div style="margin: 0px auto;border-style: solid;border-color: #646464;border-width: 0.5px; background: #323232;border-radius: 4px;padding: 5px;">'
                f'&#11088; <span style="color: #fff">{avg_rating_list[0]} |</span> <span style="color: #f53666; display: inline-block; margin-left: 2px">{n_rating_list[0]}x</span></div>'
                f'</div>',
                unsafe_allow_html=True
            )
        
        row_52.image(
                Image.open(requests.get(url_list[1], stream=True).raw), 
                use_column_width=True
            )
        row_52.markdown(
                f' _{title_list[1]}_, written by _{author_list[1]}_ \n'
                f'<br /><br />'
                f'\n <div style="display: flex; align-items: center; justify-content: center;"><div style="margin: 0px auto;border-style: solid;border-color: #646464;border-width: 0.5px; background: #323232;border-radius: 4px;padding: 5px;">'
                f'&#11088; <span style="color: #fff">{avg_rating_list[1]} |</span> <span style="color: #f53666; display: inline-block; margin-left: 2px">{n_rating_list[1]}x</span></div>'
                f'</div>',
                unsafe_allow_html=True
            )

        row_53.image(
                Image.open(requests.get(url_list[2], stream=True).raw), 
                use_column_width=True
            )
        row_53.markdown(
                f' _{title_list[2]}_, written by _{author_list[2]}_ \n'
                f'<br> \n'
                f'\n <div style="display: flex; align-items: center; justify-content: center;"><div style="margin: 0px auto;border-style: solid;border-color: #646464;border-width: 0.5px; background: #323232;border-radius: 4px;padding: 5px;">'
                f'&#11088; <span style="color: #fff">{avg_rating_list[2]} |</span> <span style="color: #f53666; display: inline-block; margin-left: 2px">{n_rating_list[2]}x</span></div>'
                f'</div>',
                unsafe_allow_html=True
            )

        row_54.image(
                Image.open(requests.get(url_list[3], stream=True).raw), 
                use_column_width=True
            )
        row_54.markdown(
                f' _{title_list[3]}_, written by _{author_list[3]}_ \n'
                f'<br /><br />'
                f'\n <div style="display: flex; align-items: center; justify-content: center;"><div style="margin: 0px auto;border-style: solid;border-color: #646464;border-width: 0.5px; background: #323232;border-radius: 4px;padding: 5px;">'
                f'&#11088; <span style="color: #fff">{avg_rating_list[3]} |</span> <span style="color: #f53666; display: inline-block; margin-left: 2px">{n_rating_list[3]}x</span></div>'
                f'</div>',
                unsafe_allow_html=True
            )

        row_55.image(
                Image.open(requests.get(url_list[4], stream=True).raw), 
                use_column_width=True
            )
        row_55.markdown(
                f' _{title_list[4]}_, written by _{author_list[4]}_ \n'
                f'<br>'
                f'\n <div style="display: flex; align-items: center; justify-content: center;"><div style="margin: 0px auto;border-style: solid;border-color: #646464;border-width: 0.5px; background: #323232;border-radius: 4px;padding: 5px;">'
                f'&#11088; <span style="color: #fff">{avg_rating_list[4]} |</span> <span style="color: #f53666; display: inline-block; margin-left: 2px">{n_rating_list[4]}x</span></div>'
                f'</div>',
                unsafe_allow_html=True
            )

        ### Row s2
        row_s2, _ = st.columns( (1, 0.001) )
        row_s2.write('')
        row_s2.write('')
        
        exp_info = row_s2.expander('About/App Info')
        exp_info.markdown(
                """                
                This app offers a simple book recommendation engine built with the use of the <a href="http://www2.informatik.uni-freiburg.de/~cziegler/BX/" target="_blank">Book-Crossing Dataset</a>.

                Individual app sections allow user to: 
                * Select a book and create top 5 recommendations,
                * See data visualisation with an exploratory data analysis of the dataset in question.
                """,
                unsafe_allow_html=True
            )
        
def analysis(main_df, book_info, user_info, user_rating_info):
        
        row_11, _ = st.columns( (1, 0.001)  )
        row_11.title('Book-Crossing Data :books:')
        row_11.write('A Web App by [Julius Rabek](https://github.com/murtagh97)')
        
        #################
        ### User Part ###
        #################
        # analysing only the users that have rated books explicitly, i.e., the explicit part of the dataset

        row_u1, _, row_u3 = st.columns( (1, 0.1, 1) )

        row_u1.header('Analyzing User Info :bust_in_silhouette:')
        row_u3.header('')

        row_31, _, row_33 = st.columns( (1, 0.1, 1) )
        row_31.subheader("User Age Distribution")
        fig = Figure(figsize = (7.1,7))
        ax = fig.subplots()
        sns.histplot(
                data = user_rating_info['age'], color = '#0c5529', ax=ax, binwidth=2, kde=False
            )
        ax.set_xlabel('User Age')
        ax.set_ylabel('Count')
        ax.grid(zorder=0,alpha=.2)
        row_31.pyplot(fig)
        exp = row_31.expander('User Age')
        exp.markdown(
                f'The mean age of the users is _{user_rating_info.age.mean().round(2)}_, with the median age being _{user_rating_info.age.median()}_. The cutoff values for minimal and maximal age has been set at _{5}_ and _{115}_ years.',
                unsafe_allow_html=True
        )

        row_33.subheader("Most Active Users")
        fig = Figure(figsize = (7,7))
        ax = fig.subplots()
        # ds = main_df['user_id'].astype(str).value_counts().reset_index()
        ds = user_rating_info['user_id'].astype(str).value_counts().reset_index()
        ds.columns = ['value', 'counts']
        ds['value'] = 'U' + ds['value']
        sns.barplot(
                data = ds.head(25) , x = 'counts', y = 'value', ax=ax, palette='Reds_r'
            )
        ax.set_xlabel('Number of Books Rated')
        ax.set_ylabel('User ID')
        ax.grid(zorder=0,alpha=.2)
        row_33.pyplot(fig)
        exp = row_33.expander('User Activity')
        exp.markdown(
                f'In total, _{ds.counts.count()}_ users have rated at least one book. One user has rated _{ds.counts.mean().round(2)}_ books in average, while the most active user has reviewed _{ds.counts.max()}_ books!',
                unsafe_allow_html=True
        )

        row_41, _, row_43 = st.columns( (1, 0.1, 1) )
        row_41.subheader("Most Frequent User Cities")
        fig = Figure(figsize = (7,7))
        ax = fig.subplots()
        ds = user_rating_info['city'].value_counts().reset_index()
        ds.columns = ['value', 'counts']
        sns.barplot(
                data = ds.head(20), x = 'counts', y = 'value', ax=ax, palette='Greens_r'
            )
        ax.set_xlabel('Number of Books Rated')
        ax.set_ylabel('City')
        ax.grid(zorder=0,alpha=.2)
        row_41.pyplot(fig)
        desc_stats = ds['counts'].head(3)
        exp = row_41.expander('User Location: Cities')
        exp.markdown(
                f'The three most frequent cities are _{ds.value.iloc[0]}_ with _{ds.counts.iloc[0]}_ users, _{ds.value.iloc[1]}_ with _{ds.counts.iloc[1]}_ users and _{ds.value.iloc[2]}_ with _{ds.counts.iloc[1]}_ users.',
                unsafe_allow_html=True
        )

        row_43.subheader("Most Frequent User States")
        fig = Figure(figsize = (6.9,7))
        ax = fig.subplots()
        ds = user_rating_info['state'].value_counts().reset_index()
        ds.columns = ['value', 'counts']
        sns.barplot(
                data = ds.head(20), x = 'counts', y = 'value', ax=ax, palette='Reds_r'
            )
        ax.set_xlabel('Number of Books Rated')
        ax.set_ylabel('State')
        ax.grid(zorder=0,alpha=.2)
        row_43.pyplot(fig)
        exp = row_43.expander('User Location: States')
        exp.markdown(
                f'The most frequently listed states are _{ds.value.iloc[0]}_ with _{ds.counts.iloc[0]}_ users and _{ds.value.iloc[1]}_ with _{ds.counts.iloc[1]}_ users. On the other hand, around _{ds.counts.iloc[4] + ds.counts.iloc[11] }_ users have not further specified which state do they come from.',
                unsafe_allow_html=True
        )

        _, row_52, _ = st.columns( (0.4, 1, 0.4) )
        row_52.subheader("Most Frequent User Countries")
        fig = Figure(figsize = (7,7))
        ax = fig.subplots()
        ds = user_rating_info['country'].value_counts().reset_index()
        ds.columns = ['value', 'counts']
        sns.barplot(
                data = ds.head(15), x = 'counts', y = 'value', ax=ax, palette='Greens_r'
            )
        ax.set_xlabel('Number of Books Rated')
        ax.set_ylabel('Country')
        ax.grid(zorder=0,alpha=.2)
        row_52.pyplot(fig)
        num = ds.counts.head(3).sum() + ds.counts[5] + ds.counts[13]
        en_percentage = ( num / ds.counts.sum() ) * 100
        exp = row_52.expander('User Location: Countries')
        exp.markdown(
                f'Around _{en_percentage.round(2)}_% of the users come from the English speaking countries, i.e., _{ds.value[0]}_, _{ds.value[1]}_, _{ds.value[2]}_, _{ds.value[5]}_, and _{ds.value[13]}_. The remaining users mostly come from the mainland Europe, while _{ds.counts[6]}_ users have not further specified which country do they come from.',
                unsafe_allow_html=True
        )

        #################
        ### Book Part ###
        #################
        # analysing only the explicit part of the dataset
        # mean quantity analyses are already made on the filtered dataset

        row_61, _, row_63 = st.columns( (1, 0.1, 1) )

        row_61.header('Analyzing Book Info :open_book:')
        row_63.header('')

        row_71, _, row_73 = st.columns( (1, 0.1, 1) )
        row_71.subheader("Book Rating Distribution")
        fig = Figure(figsize = (7.1,7))
        ax = fig.subplots()
        ds = main_df['book_rating'].astype(str).value_counts().reset_index()
        ds.columns = ['value', 'count']
        sns.barplot(
                data = ds, x = 'count', y = 'value', ax=ax, palette='Greens_r'
            )
        ax.set_xlabel('Number of Books')
        ax.set_ylabel('Book Rating')
        ax.grid(zorder=0,alpha=.2)
        row_71.pyplot(fig)
        exp = row_71.expander('Book Ratings')
        exp.markdown(
                f'On the scale from _1_ to _10_, the mean and the median book ratings are _{main_df.book_rating.mean().round(2)}_ and _{main_df.book_rating.median()}_.',
                unsafe_allow_html=True
        )

        # ds = book_info.drop_duplicates(
        #         subset = 'isbn_unique', keep="first"
        #     )
        ds = book_info['publication_year'].astype(str).value_counts().reset_index()
        ds.columns = ['value', 'counts']
        row_73.subheader("Most Frequent Years of Publication")
        fig = Figure(figsize = (7,7))
        ax = fig.subplots()
        sns.barplot(
                data = ds.head(20), x = 'counts', y = 'value', ax=ax, palette='Reds_r'
            )
        ax.set_xlabel('Number of Books')
        ax.set_ylabel('Publication Year')
        ax.grid(zorder=0,alpha=.2)
        row_73.pyplot(fig)
        num = ds['counts'].head(20).sum()
        year_percentage = ( num / ds['counts'].sum() ) * 100
        exp = row_73.expander('Publication Years')
        exp.markdown(
                f'The oldest book in the database was published in _1806_, while the latest books were published in _2004_ . Around _{year_percentage.round(2)}_% of the books were published in _1985 or later_ .',
                unsafe_allow_html=True
        )

        row_81, _, row_83 = st.columns( (1, 0.1, 1) )
        row_81.subheader("Most Rated Books")
        fig = Figure(figsize = (4.6,7))
        ax = fig.subplots()
        ds = main_df.drop_duplicates(
                 subset = 'isbn_unique', keep="first"
            )
        ds = ds.sort_values(by = 'n_book_ratings', ascending = False)
        sns.barplot(
                data = ds.head(15), x = 'n_book_ratings', y = 'book_title', ax=ax, palette='Greens_r'
            )
        ax.set_xlabel('Number of Ratings')
        ax.set_ylabel('Book')
        ax.grid(zorder=0,alpha=.2)
        row_81.pyplot(fig)
        exp = row_81.expander('Books: Quantity')
        exp.markdown(
                f'The three most rated books are the following. _{ds.book_title.iloc[0]}_, written by _{ds.book_author.iloc[0]}_, with _{ds.n_book_ratings.iloc[0]}_ ratings. _{ds.book_title.iloc[1]}_, written by _{ds.book_author.iloc[1]}_, with _{ds.n_book_ratings.iloc[1]}_ ratings. _{ds.book_title.iloc[2]}_, written by _{ds.book_author.iloc[2]}_, with _{ds.n_book_ratings.iloc[2]}_ ratings. In average, one book is rated _{ds.n_book_ratings.mean().round(2)}_ times.',
                unsafe_allow_html=True
        )

        row_83.subheader("Best Rated Popular Books")
        fig = Figure(figsize = (5,7))
        ax = fig.subplots()
        ds = main_df.drop_duplicates(
                 subset = 'isbn_unique', keep="first"
            )
        ds = ds[ds['n_book_ratings'] >= 50]
        ds = ds.sort_values(by = 'average_rating', ascending = False)
        sns.barplot(
                data = ds.head(15), x = 'average_rating', y = 'book_title', ax=ax, palette='Reds_r'
            )
        ax.set_xlabel('Average Rating')
        ax.set_ylabel('Book')
        ax.set(xlim=(1, 10))
        ax.grid(zorder=0,alpha=.2)
        row_83.pyplot(fig)
        exp = row_83.expander('Books: Quality')
        exp.markdown(
                f'The best rated popular book is _{ds.book_title.iloc[0]}_, written by _{ds.book_author.iloc[0]}_, with an average rating of _{ds.average_rating.iloc[0]}_. On the other hand, the worst rated popular book is _{ds.book_title.iloc[-1]}_, written by _{ds.book_author.iloc[-1]}_, with an average rating of _{ds.average_rating.iloc[-1]}_. Popular books being the books with at least 50 ratings.',
                unsafe_allow_html=True
        )

        row_91, _, row_93 = st.columns( (1, 0.1, 1) )
        row_91.subheader("Most Rated Authors")
        fig = Figure(figsize = (6.5,7))
        ax = fig.subplots()
        ds = main_df['book_author'].value_counts().reset_index()
        ds.columns = ['value', 'n_ratings']
        sns.barplot(
                data = ds.head(15), x = 'n_ratings', y = 'value', ax=ax, palette='Greens_r'
            )
        ax.set_xlabel('Number of Ratings')
        ax.set_ylabel('Author')
        ax.grid(zorder=0,alpha=.2)
        row_91.pyplot(fig)
        exp = row_91.expander('Authors: Quantity')
        exp.markdown(
                f'The three most rated authors are _{ds.value.iloc[0]}_ with _{ds.n_ratings.iloc[0]}_ ratings, _{ds.value.iloc[1]}_ with _{ds.n_ratings.iloc[1]}_ ratings, and _{ds.value.iloc[2]}_ with _{ds.n_ratings.iloc[2]}_ ratings . In average, one author is rated _{ds.n_ratings.mean().round(2)}_ times.',
                unsafe_allow_html=True
        )

        row_93.subheader("Best Rated Popular Authors")
        fig = Figure(figsize = (6.25,7))
        ax = fig.subplots()
        ds = main_df['book_author'].value_counts().reset_index()
        ds.columns = ['book_author', 'author_evaluation_count']
        ds = pd.merge(main_df, ds, on='book_author')
        ds = ds[ds['author_evaluation_count'] > 100]
        ds = ds.groupby('book_author')['book_rating'].mean().reset_index().sort_values('book_rating', ascending = False)
        ds = ds[~ds['book_author'].str.contains("J.R.R.")]
        sns.barplot(
                data = ds.head(15), x = 'book_rating', y = 'book_author', ax=ax, palette='Reds_r'
            )
        ax.set_xlabel('Average Rating')
        ax.set_ylabel('Author')
        ax.grid(zorder=0,alpha=.2)
        row_93.pyplot(fig)
        exp = row_93.expander('Authors: Quality')
        exp.markdown(
                f'The best rated popular author is _{ds.book_author.iloc[0]}_ with an average rating of _{ds.book_rating.iloc[0].round(2)}_. On the other hand, the worst rated popular author is _{ds.book_author.iloc[-1]}_ with an average rating of _{ds.book_rating.iloc[-1].round(2)}_. Popular authors being the authors with at least 100 ratings.',
                unsafe_allow_html=True
        )

        row_101, _, row_103 = st.columns( (1, 0.1, 1) )
        row_101.subheader("Most Rated Publishers")
        fig = Figure(figsize = (6.5,7))
        ax = fig.subplots()
        ds = main_df['publisher'].value_counts().reset_index()
        ds.columns = ['value', 'n_ratings']
        sns.barplot(
                data = ds.head(15), x = 'n_ratings', y = 'value', ax=ax, palette='Greens_r'
            )
        ax.set_xlabel('Number of Ratings')
        ax.set_ylabel('Publisher')
        ax.grid(zorder=0,alpha=.2)
        row_101.pyplot(fig)
        exp = row_101.expander('Publishers: Quantity')
        exp.markdown(
                f'The three most rated publishers are _{ds.value.iloc[0]}_ with _{ds.n_ratings.iloc[0]}_ ratings, _{ds.value.iloc[1]}_ with _{ds.n_ratings.iloc[1]}_ ratings, and _{ds.value.iloc[2]}_ with _{ds.n_ratings.iloc[2]}_ ratings . In average, one publisher is rated _{ds.n_ratings.mean().round(2)}_ times.',
                unsafe_allow_html=True
        )

        row_103.subheader("Best Rated Popular Publishers")
        fig = Figure(figsize = (6.15,7))
        ax = fig.subplots()
        ds = main_df['publisher'].value_counts().reset_index()
        ds.columns = ['publisher', 'author_evaluation_count']
        ds = pd.merge(main_df, ds, on='publisher')
        ds = ds[ds['author_evaluation_count'] > 150]
        ds = ds.groupby('publisher')['book_rating'].mean().reset_index().sort_values('book_rating', ascending = False)
        ds = ds[~ds['publisher'].str.contains("J.R.R.")]
        sns.barplot(
                data = ds.head(15), x = 'book_rating', y = 'publisher', ax=ax, palette='Reds_r'
            )
        ax.set_xlabel('Average Rating')
        ax.set_ylabel('Publisher')
        ax.grid(zorder=0,alpha=.2)
        row_103.pyplot(fig)
        exp = row_103.expander('Publishers: Quality')
        exp.markdown(
                f'The best rated popular publisher is _{ds.publisher.iloc[0]}_ with an average rating of _{ds.book_rating.iloc[0].round(2)}_. On the other hand, the worst rated popular publisher is _{ds.publisher.iloc[-1]}_ with an average rating of _{ds.book_rating.iloc[-1].round(2)}_. Popular publishers being the publishers with at least 150 ratings.',
                unsafe_allow_html=True
        )

        _, row_112, _ = st.columns( (0.4, 1, 0.4) )
        row_112.subheader("Authors with Most Books Published")
        fig = Figure(figsize = (7.5,7))
        ax = fig.subplots()
        ds = book_info['book_author'].value_counts().reset_index()
        ds.columns = ['value', 'n_ratings']
        sns.barplot(
                data = ds.head(15), x = 'n_ratings', y = 'value', ax=ax, palette='Greens_r'
            )
        ax.set_xlabel('Number of Books')
        ax.set_ylabel('Author')
        ax.grid(zorder=0,alpha=.2)
        row_112.pyplot(fig)
        exp = row_112.expander('Author Productivity')
        exp.markdown(
                f'''Three authors with the most books published are the following. _{ds.value.iloc[0]}_ with _{ds.n_ratings.iloc[0]}_ books, _{ds.value.iloc[1]}_ with _{ds.n_ratings.iloc[1]}_ books, and _{ds.value.iloc[2]}_ with _{ds.n_ratings.iloc[2]}_ books . In average, one author has _{ds.n_ratings.mean().round(2)}_  books published.''',
                unsafe_allow_html=True
        )

if __name__ == "__main__":

        main()
