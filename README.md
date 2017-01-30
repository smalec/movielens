# MovieLens Recommender System

MovieLens (https://movielens.org/) shares with community several datasets containing user's ratings of some movies. Our goal is to bulid a recommender system that will recommend user some movies that he propably would like to see based on his already collected ratings of other movies. We will use 2 datasets for our purposes:

* 100k dataset (100,000 ratings and 1,300 tag applications applied to 9,000 movies by 700 users)
* 1M dataset (1 million ratings from 6000 users on 4000 movies).

Before we move on to the different approaches of implementing such systems, let us discuss about evaluating recommender systems. When one system is said to be *better* than another?

## Evaluating systems

Each recommender system can either offer user some movies that he doesn't yet see or predict a rating for a given movie. Thus, we will perform evaluation for both of those modes.

### RMSE

RMSE (Root Mean Square Error) measures how accurate rating predictions of given recommender system are. We divide dataset of ratings into train and test sets (train set consists of ratings of first 80% of users; remaining ratings will be included in the test set). For each user whose ratings belongs to test set we will perform 5-cross validation. 4 out of 5 parts of user's ratings recommender system will use in order to gain some informations about user preferences and for each rating\* from the remaining part of ratings we will compute RSE (Root Square Error):
<p align="center">
![equation](http://latex.codecogs.com/gif.latex?%28predicted%20-%20actual%29%5E2)

RMSE is the sum of all RSEs divided by the number of ratings for which RSE was computed. Of course: smaller RMSE value means that our system predicts ratings better.

\* *Actually recommender system can sometimes state that it has not enough information to predict rating for given movie and user. We will ignore such cases while computing RMSE.*

### MAP

MAP (Mean Average Precision) tries to measure if our recommendations are indeed relevant. We will use the same division of dataset into train and test sets as in RMSE computations. And we will also perform 5-cross validation among each user from test set, but this time we will try to measure how good our recommendations are. More precisely: the system will recommend top 5 movies based on 4 out of 5 parts of user's ratings and compute AP (Average Precision) for this recommendations (assuming that relevant recommendations are these which where rate with 3.0 or more in remaining part of user's rating). AP is computed as follows:

<p align="center">
![equation](http://latex.codecogs.com/gif.latex?%5Csum_%7Bi%3D1%7D%5E5%20%5Cfrac%7B1%7D%7Bi%7D%5CBig%28%5Cfrac%7B%5Csum_%7Bj%3D1%7D%5EiI_j%7D%7Bi%7D%5CBig%29*%20I_i)

where:

<p align="center">
![equation](http://latex.codecogs.com/gif.latex?I_i%20%3D%20%5Cbegin%7Bcases%7D%201%20%26%20i%5Ctext%7B-th%20recommendation%20is%20relevant%7D%5C%5C%200%20%26%20%5Ctext%7Botherwise%7D%20%5Cend%7Bcases%7D)

In particular: AP doesn't penalize for bad guesses, but we should care about order of our recommendations. MAP is the sum of all APs divided by the number of all APs (this is: size of test set multiplied by 5). Of course: bigger MAP value means that system gives more relevant recommendations.

## Content-based recommendation
In content-based recommender system we recommend movies that are similar to user's preferences.

Each movie in dataset is classified by some of 18 genres. We then represent movie type by 1-D vector of size 18 where *i*-th value of the vector is either 1 (if *i*-th genre is assigned to movie) or 0 (otherwise). Then we define *user profile* by weighted average of types of movies that he watched, where weights are user's ratings.

Our recommendations are movies that are closest (with cosine metric) to the *user profile*.

While predicting rating, content-based recommender system will find 5 closest movies, that user already watched, to the given movie. The predicted rating will be the average of ratings of this 5 movies.

Using this approach we achieve following results:

 | RMSE | MAP
--- | --- | ---
**100k dataset** | 0.991 | 0.011
**1M dataset** | 1.188 | 0.045

## Collaborative filtering

Collaborative filtering is about finding users that have similar preferences.

Firstly, we want to slightly redefine *user profile*. It still will be 1-D vector of size 18, but this time it would be weighted average over *signum(*2.75 - *rating) * movie_type* for each movie that was rated by user with weights *(*2.75 - *rating)<sup>2</sup>*. The intuition behind such definition is to promote movies which user rated highly (over 2.75) and to penalize movies that was rated under 2.75. Values in *user profile* are between -1 and 1 and they should tell us if user likes or not given genre.

In order to predict rating for a given movie and user, recommender system will find few (by default: 20 -- if there is not enough data, the system won't predict the rating) users who have watched the movie and have most similar profiles within cosine metric. Then the predicted rating will be the average of ratings of those users for this film.

While recommending movies the system will predict rating for all movies that user doesn't watch and choose top rated movies.

Results of computing RMSE and MAP for both datasets using collaborative filtering approach are presented in table:

 | RMSE | MAP
--- | --- | ---
**100k dataset** | 0.752 | 0.168
**1M dataset** | 0.993 | 0.155

Comparing this results to those obtained with contant-based approach we could claim with a good conscience that collaborative filtering gives better recommendations.

### Collaborative filtering with clustering

We could apply simple modification to above standard collaborative filtering approach. We could group users from training set using KMeans algorithm and then when predicting rating for given movie and user we would take under consideration only users from nearest group to the *user profile*. Our system will claim that it has not enough information to predict rating only if for given movie and user there are less than 5 users from the nearest group that rate the movie.

Trying different number of clusters that KMeans will use to group data we get following results for 100k dataset:

 | RMSE | MAP
--- | --- | ---
**clusters=5** | 0.757 | 0.227
**clusters=10** | *0.635* | *0.271*
**clusters=20** | 0.780 | 0.132

Using `clusters=10` we compute RMSE and MAP also for 1M dataset:

 | RMSE | MAP
--- | --- | ---
**100k dataset** | 0.635 | 0.271
**1M dataset** | 0.923 | 0.174

As we can see this slight modifaction improved our metrices a bit. It also results in a bit more expansive phase of training since we need to perform KMeans but the phase of recommending is a bit cheaper because we are limited to users only from the nearest cluster.
