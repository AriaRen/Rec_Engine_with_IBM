import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import project_tests as t
import pickle

%matplotlib inline

df = pd.read_csv('data/user-item-interactions.csv')
df_content = pd.read_csv('data/articles_community.csv')
del df['Unnamed: 0']
del df_content['Unnamed: 0']

#Delete Duplicate for df_content
dup_articles = df_content[df_content.duplicated(subset='article_id', keep=False)]
df_content = df_content.drop_duplicates(subset='article_id').reset_index(drop=True)


# Run this cell to map the user email to a user_id column and remove the email column
def email_mapper():
    coded_dict = dict()
    cter = 1
    email_encoded = []
    
    for val in df['email']:
        if val not in coded_dict:
            coded_dict[val] = cter
            cter+=1
        
        email_encoded.append(coded_dict[val])
    return email_encoded

email_encoded = email_mapper()
del df['email']
df['user_id'] = email_encoded

#Rank-Based Recommendations

def get_top_articles(n, df=df):
    '''
    INPUT:
    n - (int) the number of top articles to return
    df - (pandas dataframe) df as defined at the top of the notebook 
    
    OUTPUT:
    top_articles - (list) A list of the top 'n' article titles 
    
    '''
    top_articles = [str(i) for i in df.title.value_counts().index[:n].tolist()]
    
    return top_articles # Return the top article titles from df (not df_content)

def get_top_article_ids(n, df=df):
    '''
    INPUT:
    n - (int) the number of top articles to return
    df - (pandas dataframe) df as defined at the top of the notebook 
    
    OUTPUT:
    top_articles - (list) A list of the top 'n' article titles 
    
    '''
    top_articles = [str(i) for i in df.article_id.value_counts().index[:n].tolist()]
 
    return top_articles # Return the top article ids

#User-User Based Collaborative Filtering
def create_user_item_matrix(df):
    '''
    INPUT:
    df - pandas dataframe with article_id, title, user_id columns
    
    OUTPUT:
    user_item - user item matrix 
    
    Description:
    Return a matrix with user ids as rows and article ids on the columns with 1 values where a user interacted with 
    an article and a 0 otherwise
    '''

    user_item = df.drop_duplicates().groupby(by=['user_id', 'article_id']).count().reset_index().pivot(index='user_id', columns='article_id', values='title')
    user_item = user_item.fillna(0)
    
    return user_item # return the user_item matrix 

user_item = create_user_item_matrix(df)

def get_article_names(article_ids, df=df):
    '''
    INPUT:
    article_ids - (list) a list of article ids
    df - (pandas dataframe) df as defined at the top of the notebook
    
    OUTPUT:
    article_names - (list) a list of article names associated with the list of article ids 
                    (this is identified by the title column)
    '''
    
    article_names = [df[df['article_id']==float(article_id)].title.tolist()[0] for article_id in article_ids]
    
    return article_names # Return the article names associated with list of article ids

def get_user_articles(user_id, user_item=user_item):
    '''
    INPUT:
    user_id - (int) a user id
    user_item - (pandas dataframe) matrix of users by articles: 
                1's when a user has interacted with an article, 0 otherwise
    
    OUTPUT:
    article_ids - (list) a list of the article ids seen by the user
    article_names - (list) a list of article names associated with the list of article ids 
                    (this is identified by the doc_full_name column in df_content)
    
    Description:
    Provides a list of the article_ids and article titles that have been seen by a user
    '''
    
    article_ids = [str(i) for i in df[df['user_id']==user_id].article_id.unique()]
    article_names = [df[df['article_id'] == float(i)].title.tolist()[0] for i in article_ids]

    return article_ids, article_names # return the ids and names

def get_top_sorted_users(user_id, df=df, user_item=user_item):
    '''
    INPUT:
    user_id - (int)
    df - (pandas dataframe) df as defined at the top of the notebook 
    user_item - (pandas dataframe) matrix of users by articles: 
            1's when a user has interacted with an article, 0 otherwise
    
            
    OUTPUT:
    neighbors_df - (pandas dataframe) a dataframe with:
                    neighbor_id - is a neighbor user_id
                    similarity - measure of the similarity of each user to the provided user_id
                    num_interactions - the number of articles viewed by the user - if a u
                    
    Other Details - sort the neighbors_df by the similarity and then by number of interactions where 
                    highest of each is higher in the dataframe
     
    '''

    neighbor_id = [i for i in user_item.index.tolist() if i != user_id]
    similarity = [sum([x*y for x,y in zip(user_item.iloc[user_id-1,:], user_item.iloc[user-1,:])]) for user in \
                  user_item.index.tolist() if user != user_id]
    num_interactions = [df[df['user_id'] == user].shape[0] for user in user_item.index.tolist() if user != user_id]
    neighbors_df = pd.DataFrame({'neighbor_id': neighbor_id, 'similarity': similarity, 'num_interactions': num_interactions})
    neighbors_df = neighbors_df.sort_values(['similarity', 'num_interactions'], ascending = False).reset_index(drop=True)
    
    return neighbors_df # Return the dataframe specified in the doc_string

def user_user_recs_part2(user_id, m=10):
    '''
    INPUT:
    user_id - (int) a user id
    m - (int) the number of recommendations you want for the user
    
    OUTPUT:
    recs - (list) a list of recommendations for the user by article id
    rec_names - (list) a list of recommendations for the user by article title
    
    Description:
    Loops through the users based on closeness to the input user_id
    For each user - finds articles the user hasn't seen before and provides them as recs
    Does this until m recommendations are found
    
    Notes:
    * Choose the users that have the most total article interactions 
    before choosing those with fewer article interactions.

    * Choose articles with the articles with the most total interactions 
    before choosing those with fewer total interactions. 
   
    '''

    articles_seen = get_user_articles(user_id)[0]
    closest_neighbors = get_top_sorted_users(user_id).neighbor_id.tolist()
    # Keep the recommended articles here
    recs = np.array([])
    
    for neighbor in closest_neighbors:
        neighbs_likes = get_user_articles(neighbor)[0]
        
        #Obtain recommendations for each neighbor
        new_recs = np.setdiff1d(neighbs_likes, articles_seen, assume_unique=True)
        new_recs_dic = {rec:df[df['article_id'] == float(rec)].shape[0] for rec in new_recs}
        rec_list = [i[0] for i in sorted(new_recs_dic.items(), key=lambda x: x[1], reverse=True)] 
        
        # Update recs with new recs
        recs = np.unique(np.concatenate([rec_list, recs], axis=0))
        
        # If we have enough recommendations exit the loop
        if len(recs) > m-1:
            break
    recs = recs[:m]
    rec_names = get_article_names(recs, df=df)
    
    
    return recs, rec_names