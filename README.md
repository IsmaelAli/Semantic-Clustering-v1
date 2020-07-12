# Semantic-Clustering-v1

#### Use TF/IDF + Kmeans to semantically cluster text documents.
#### Then evaluate the clustering results.

Output:

No of docs in cporpus:
8

 original input corpus: ################################# 

    The students went to their new school yesterday.
    Azad was the best basketball player in the city.
    Next week the basketball league will start in Zakho. 
    Today the school is full of happy students for starting their new school year. 
    Best team in zakho will play with the best team from Dohuk. 
    Teachers were so excited seeing their students happy.
    Hello dear students! Cheered the school principle.
    Students started to play basketball from day one.
    
{0: '    The students went to their new school yesterday.', 1: '    Azad was the best basketball player in the city.', 2: '    Next week the basketball league will start in Zakho. ', 3: '    Today the school is full of happy students for starting their new school year. ', 4: '    Best team in zakho will play with the best team from Dohuk. ', 5: '    Teachers were so excited seeing their students happy.', 6: '    Hello dear students! Cheered the school principle.', 7: '    Students started to play basketball from day one.'}

 Term/Feature Names:  ################################# 

['!', '.', 'azad', 'basketball', 'best', 'cheered', 'city', 'day', 'dear', 'dohuk', 'excited', 'full', 'happy', 'hello', 'league', 'new', 'next', 'one', 'play', 'player', 'principle', 'school', 'seeing', 'start', 'started', 'starting', 'student', 'teacher', 'team', 'today', 'wa', 'week', 'went', 'year', 'yesterday', 'zakho']

 File_IDSs:  [0, 1, 2, 3, 4, 5, 6, 7]
 
tf_idf_matrix: ################################# 
 
number of features/term:  36
number of docs:  8 
 
0 [0.0, 0.208, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.436, 0.0, 0.0, 0.0, 0.0, 0.0, 0.376, 0.0, 0.0, 0.0, 0.0, 0.292, 0.0, 0.0, 0.0, 0.0, 0.0, 0.52, 0.0, 0.52, 0.0]
1 [0.0, 0.172, 0.431, 0.312, 0.361, 0.0, 0.431, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.431, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.431, 0.0, 0.0, 0.0, 0.0, 0.0]
2 [0.0, 0.172, 0.0, 0.312, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.431, 0.0, 0.431, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.431, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.431, 0.0, 0.0, 0.0, 0.361]
3 [0.0, 0.141, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.354, 0.297, 0.0, 0.0, 0.297, 0.0, 0.0, 0.0, 0.0, 0.0, 0.512, 0.0, 0.0, 0.0, 0.354, 0.199, 0.0, 0.0, 0.354, 0.0, 0.0, 0.0, 0.354, 0.0, 0.0]
4 [0.0, 0.13, 0.0, 0.0, 0.547, 0.0, 0.0, 0.0, 0.0, 0.327, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.274, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.653, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.274]
5 [0.0, 0.195, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.489, 0.0, 0.41, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.489, 0.0, 0.0, 0.0, 0.275, 0.489, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
6 [0.408, 0.163, 0.0, 0.0, 0.0, 0.408, 0.0, 0.0, 0.408, 0.0, 0.0, 0.0, 0.0, 0.408, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.408, 0.295, 0.0, 0.0, 0.0, 0.0, 0.229, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
7 [0.0, 0.184, 0.0, 0.334, 0.0, 0.0, 0.0, 0.461, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.461, 0.387, 0.0, 0.0, 0.0, 0.0, 0.0, 0.461, 0.0, 0.259, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


 km.fit :  
 
  <bound method KMeans.fit of KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
    n_clusters=2, n_init=10, n_jobs=1, precompute_distances='auto',
    random_state=None, tol=0.0001, verbose=0)>
    
Results: ################################# 
                                                  text  category
0      The students went to their new school yest...         0
1      Azad was the best basketball player in the...         1
2      Next week the basketball league will start...         1
3      Today the school is full of happy students...         0
4      Best team in zakho will play with the best...         1
5      Teachers were so excited seeing their stud...         0
6      Hello dear students! Cheered the school pr...         0
7      Students started to play basketball from d...         1

Clustering Evaluation Results: ################################# 
 
precision: [ 0.    0.25]
recall: [ 0.   0.2]
fscore: [ 0.          0.22222222]
support: [3 5]

Process finished with exit code 0
