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

        main_df, book_info, user_info = load_data()

        wide_df, wide_df_sparse = create_wide_df(main_df)

        model_knn = create_model(wide_df_sparse)

        selected_box = st.sidebar.selectbox(
                'Select Section',
                ('Main', 'Book Crossing Data')
            )
        
        if selected_box == 'Main':

                book_recommendation(main_df, wide_df, model_knn)

        if selected_box == 'Book Crossing Data': 

                analysis(main_df, book_info, user_info) 

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def load_data(arg_df = 'explicit'):

        book_info = pd.read_csv(os.getcwd() + '/data/book_info_unique.csv')
        user_info = pd.read_csv(os.getcwd() + '/data/user_info.csv')
        rating_info = pd.read_csv(os.getcwd() + '/data/rating_info.csv')

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

        # only keep books that have been rated at least certain amount of times
        main_df = main_df[main_df['n_book_ratings'] >= 35]

        return main_df, book_info, user_info

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
        row_11, _, row_12, = st.beta_columns( (2, 1.5, 1) )
        
        row_11.title('Book Recommender System :books:')
        row_12.subheader('A Web App by [Julius Rabek](https://github.com/murtagh97)')
        
        ### Row s1
        row_s1, _ = st.beta_columns( (2, 1) )
        
        row_s1.write('')
        row_s1.write('')

        ### Row 2
        row_21, _ = st.beta_columns( (1, 0.001) )
        
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
        row_21.subheader('Book Info')

        ### Row 3
        row_31, _, row_33, _ = st.beta_columns( (1, 0.75, 1.5, 1) )

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
        row_41, _ = st.beta_columns( (1, 0.001) )
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
        row_51, row_52, row_53, row_54, row_55 = st.beta_columns( (1, 1, 1, 1, 1) )
        
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
        row_s2, _ = st.beta_columns( (1, 0.001) )
        row_s2.write('')
        row_s2.write('')
        
        exp_info = row_s2.beta_expander('About/App Info')
        exp_info.markdown(
                """
                by Július Rábek  \n
                <a href="https://github.com/murtagh97/segmentator_unet" target="_blank">GitHub</a> <a href="https://www.linkedin.com/in/julius-rabek/" target="_blank">LinkedIn</a>
                
                This app examines the use of <a href="https://arxiv.org/abs/1505.04597" target="_blank">UNet</a> model to segment the 
                lung fields from a set of front view chest X-rays given in the <a href="https://www.isi.uu.nl/Research/Databases/SCR/" target="_blank">SCR dataset</a>.

                Individual app sections allow user to: 
                * See the details of the final model and the underlying dataset,
                * Display the training procedure and the model results on the respective datasets,
                * Upload an image and try different data augmentation methods,
                * Upload an image and predict the resulting segmentation.
                
                Feel free to reach out if you have any feedback or suggestions!
                """,
                unsafe_allow_html=True
            )
        
def analysis(main_df, book_info, user_info):
        
        row_11, _, row_12, = st.beta_columns( (2, 1.5, 1) )
        
        row_11.title('Book Crossing Data :books:')
        row_12.subheader('A Web App by [Julius Rabek](https://github.com/murtagh97)')

        row_s1, _ = st.beta_columns( (2, 1) )
        row_s1.write('')

        ### User Part ###
        row_31, _, row_33 = st.beta_columns( (1, 0.1, 1) )

        row_31.header('Analyzing User Info :bust_in_silhouette:')
        row_33.header('')

        row_31.subheader("User Age Distribution")
        fig = Figure(figsize = (7.1,7))
        ax = fig.subplots()
        sns.histplot(
                data = user_info['age'], color = '#0c5529', ax=ax, binwidth=2, kde=False
            )
        ax.set_xlabel('User Age')
        ax.set_ylabel('Count')
        ax.grid(zorder=0,alpha=.2)
        row_31.pyplot(fig)

        row_33.subheader("Most Active Users")
        fig = Figure(figsize = (7,7))
        ax = fig.subplots()
        ds = main_df['user_id'].astype(str).value_counts().reset_index().head(25)
        ds.columns = ['value', 'count']
        ds['value'] = 'U' + ds['value']
        sns.barplot(
                data = ds , x = 'count', y = 'value', ax=ax, palette='Reds_r'
            )
        ax.set_xlabel('Number of Books Rated')
        ax.set_ylabel('User ID')
        ax.grid(zorder=0,alpha=.2)
        row_33.pyplot(fig)

        row_41, _, row_43 = st.beta_columns( (1, 0.1, 1) )
        row_41.subheader("Most Frequent User Cities")
        fig = Figure(figsize = (7.2,7))
        ax = fig.subplots()
        ds = user_info['city'].value_counts().reset_index().head(20)
        ds.columns = ['value', 'count']
        sns.barplot(
                data = ds, x = 'count', y = 'value', ax=ax, palette='Greens_r'
            )
        ax.set_xlabel('Number of Books Rated')
        ax.set_ylabel('City')
        ax.grid(zorder=0,alpha=.2)
        row_41.pyplot(fig)

        row_43.subheader("Most Frequent User States")
        fig = Figure(figsize = (7,7))
        ax = fig.subplots()
        ds = user_info['state'].value_counts().reset_index().head(15)
        ds.columns = ['value', 'count']
        sns.barplot(
                data = ds, x = 'count', y = 'value', ax=ax, palette='Reds_r'
            )
        ax.set_xlabel('Number of Books Rated')
        ax.set_ylabel('State')
        ax.grid(zorder=0,alpha=.2)
        row_43.pyplot(fig)

        _, row_52, _ = st.beta_columns( (0.4, 1, 0.4) )
        row_52.subheader("Most Frequent User Countries")
        fig = Figure(figsize = (7,7))
        ax = fig.subplots()
        ds = user_info['country'].value_counts().reset_index().head(15)
        ds.columns = ['value', 'count']
        sns.barplot(
                data = ds, x = 'count', y = 'value', ax=ax, palette='Greens_r'
            )
        ax.set_xlabel('Number of Books Rated')
        ax.set_ylabel('Country')
        ax.grid(zorder=0,alpha=.2)
        row_52.pyplot(fig)

        ### Book Data Part ###
        row_61, _, row_63 = st.beta_columns( (1, 0.1, 1) )

        row_61.header('Analyzing Book Info :open_book:')
        row_63.header('')

        row_71, _, row_73 = st.beta_columns( (1, 0.1, 1) )
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

        # ds = book_info.drop_duplicates(
        #         subset = 'isbn_unique', keep="first"
        #     )
        ds = book_info['publication_year'].astype(str).value_counts().reset_index().head(20)
        ds.columns = ['value', 'count']
        row_73.subheader("Most Frequent Years of Publication")
        fig = Figure(figsize = (7,7))
        ax = fig.subplots()
        sns.barplot(
                data = ds, x = 'count', y = 'value', ax=ax, palette='Reds_r'
            )
        ax.set_xlabel('Number of Books')
        ax.set_ylabel('Publication Year')
        ax.grid(zorder=0,alpha=.2)
        row_73.pyplot(fig)

        row_81, _, row_83 = st.beta_columns( (1, 0.1, 1) )
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

        row_91, _, row_93 = st.beta_columns( (1, 0.1, 1) )
        row_91.subheader("Most Rated Authors")
        fig = Figure(figsize = (6.5,7))
        ax = fig.subplots()
        ds = main_df['book_author'].value_counts().reset_index().head(15)
        ds.columns = ['value', 'count']
        sns.barplot(
                data = ds, x = 'count', y = 'value', ax=ax, palette='Greens_r'
            )
        ax.set_xlabel('Number of Ratings')
        ax.set_ylabel('Author')
        ax.grid(zorder=0,alpha=.2)
        row_91.pyplot(fig)

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

        row_101, _, row_103 = st.beta_columns( (1, 0.1, 1) )
        row_101.subheader("Most Rated Publishers")
        fig = Figure(figsize = (6.5,7))
        ax = fig.subplots()
        ds = main_df['publisher'].value_counts().reset_index().head(15)
        ds.columns = ['value', 'count']
        sns.barplot(
                data = ds, x = 'count', y = 'value', ax=ax, palette='Greens_r'
            )
        ax.set_xlabel('Number of Ratings')
        ax.set_ylabel('Publisher')
        ax.grid(zorder=0,alpha=.2)
        row_101.pyplot(fig)

        row_103.subheader("Authors with Most Books Published")
        fig = Figure(figsize = (6.8,7))
        ax = fig.subplots()
        ds = book_info['book_author'].value_counts().reset_index().head(15)
        ds.columns = ['value', 'count']
        sns.barplot(
                data = ds, x = 'count', y = 'value', ax=ax, palette='Reds_r'
            )
        ax.set_xlabel('Number of Books')
        ax.set_ylabel('Author')
        ax.grid(zorder=0,alpha=.2)
        row_103.pyplot(fig)

if __name__ == "__main__":

        main()
