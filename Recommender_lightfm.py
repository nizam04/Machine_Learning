'''This is a recommender system based on Lightfm model. Lightfm has it's own database of moview recomendation
but I created my own data to better understand the system. 

DOCS
Lightfm:
https://lyst.github.io/lightfm/docs/home.html#usage

Sparse matrices:
https://docs.scipy.org/doc/scipy/reference/sparse.html#sparse-matrix-classes

Youtube tutorial:
https://www.youtube.com/watch?v=ZspR5PZemcs
'''
import numpy as np
from numpy import array
from scipy import sparse
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

model = LightFM(no_components=2)

'''train = np.matrix('1 0 1 0 1; \
                   1 0 0 -1 0; \
                   0 0 1 0 1; \
                   0 1 0 -1 0'
                   )'''

#Creating sparse matrix in coordinated form to make it compatible with lighfm
row_coo = array([0, 0, 0, 1, 1, 2, 2, 3, 3])
col_coo = array([0, 2, 4, 0, 3, 2, 4, 1, 3])
Interactions = array([1, 1, 1, 1, -1, 1, 1, 1, 1])
train = sparse.coo_matrix((Interactions,(row_coo,col_coo)),shape=(4,5)).tocsr()

'''user_features = np.matrix('1 0; \
                           0 1; \
                           1 0; \
                           1 1'
                  )'''

#Creating sparse matrix for user features
#i.e is user A likes Action, Commedy, and etc. In here we considered two features only action and commedy
user_row = array([0, 1, 2, 3, 3])
user_col = array([0, 1, 0, 0, 1])
user_val = array([1, 1, 1, 1, 1])
user_features = sparse.coo_matrix((user_val,(user_row,user_col)),shape=(4,2)).tocsr()

'''item_features = np.matrix('1 0 
                              0 1 
                              1 0
                              0 1
                              1 1'
                              
                    )'''
# Creating sarse matrix for item (i.e movie) features
#i.e A movie  M1 has features like Action, Commedy, and etc. In here we considered two features only action and commedy
item_row = array([0, 1, 2, 3, 4, 4])
item_col = array([0, 1, 1, 1, 0, 1])
item_val = array([1, 1, 1, 1, 1, 1])
item_features = sparse.coo_matrix((item_val,(item_row,item_col)),shape=(5,2)).tocsr()

print("Training model!")

test_user_ids = [3]
test_item_ids = [3]

#This model is giving some unknown exception and I'm avoiding it for now
'''model.fit(train,
          user_features=user_features,
          item_features=item_features,
          epochs=2)

predictions = model.predict([1,2],
                            [1,2],
                            user_features=user_features,
                            item_features=item_features             
                            )'''


model.fit(train, epochs=100, num_threads=4)
predictions = model.predict(test_user_ids, test_item_ids, num_threads=4)

print("Printing predictions:")
print(predictions)