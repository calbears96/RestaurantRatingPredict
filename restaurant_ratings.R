#restaurant and consumer data from UCI
#https://archive.ics.uci.edu/ml/datasets/Restaurant+%26+consumer+data
#rating_final.csv for collaborative filter with user, item, rating

#load libraries
library(caret)
library(caTools)
library(tidyverse)
library(data.table)
library(h2o)
library(recosystem)
library(recommenderlab)
library(ggplot2)
library(irlba)

#create RMSE function
RMSE = function(true_ratings, predicted_ratings){
        sqrt(mean((true_ratings - predicted_ratings)^2))
}

#read in the data
restaurant_cons = read.csv('~/Documents/edx_courses/harvard_capstone/RCdata/rating_final.csv')

#data transformation for userID, to drop the U and make integer
restaurant_cons$userID = substring(restaurant_cons$userID, 2,5)
restaurant_cons$userID = as.numeric(restaurant_cons$userID)

#will need to partition the data into a training set and validation (test) set
#split 80% training, 20% test/validation

set.seed(35)
sample = sample.split(restaurant_cons, SplitRatio = .8)
train_set = subset(restaurant_cons, sample==TRUE)
test_set = subset(restaurant_cons, sample==FALSE)

#will work with the train_set
#take a look at the train_set
glimpse(train_set)

#summary of train set
summary(train_set)

#take a look at the total rating (variable of interest)
train_set %>% ggplot(aes(x=rating)) +
        geom_histogram(binwidth = .2, fill='blue') +
        labs(x = 'Rating', y = 'Number of ratings')

top_restaurant = train_set %>%
        group_by(placeID) %>%
        summarize(count = n()) %>%
        top_n(20, count) %>%
        arrange(desc(count))

#graph of top restaurants
top_restaurant %>%
        ggplot(aes(x=reorder(placeID, count), y=count)) +
        geom_bar(stat='identity', fill='blue') +
        coord_flip(y=c(0,35)) +
        labs(x = "", y='Number of ratings')

#histogram of number ratings by placeID
train_set %>% count(placeID) %>%
        ggplot(aes(n)) +
        geom_histogram(bins=15, color='red', fill='blue') +
        labs(x = 'placeID',
             y = 'Number of ratings')

#histogram of ratings by userID
train_set %>% count(userID) %>%
        ggplot(aes(n)) +
        geom_histogram(bins=15, color='blue', fill='lightblue') +
        labs(x = 'userID',
             y = 'Number of ratings')


#simple modeling first, with restaurant effect first (placeID)

#calculate the mean rating first
mu = mean(train_set$rating)

#calculate b_i for training set
rest_avgs = train_set %>%
        group_by(placeID) %>%
        summarize(b_i = mean(rating - mu))

#predict ratings
predicted_ratings_bi = mu + test_set %>%
        left_join(rest_avgs, by='placeID') %>%
        .$b_i

#restaurant and user effect
user_avgs = train_set %>%
        left_join(rest_avgs, by='placeID') %>%
        group_by(userID) %>%
        summarize(b_u = mean(rating - mu - b_i))

#predicted ratings b_i, b_u
predicted_ratings_bu = test_set %>%
        left_join(rest_avgs, by='placeID') %>%
        left_join(user_avgs, by='userID') %>%
        mutate(pred = mu + b_i + b_u) %>%
        .$pred

#calculate RMSEs
rmse_model1 = RMSE(test_set$rating, predicted_ratings_bi)
rmse_model1

rmse_model2 = RMSE(test_set$rating, predicted_ratings_bu)
rmse_model2

#regularization model

#find tuning parameter for lamnda
lambdas = seq(0, 10, .25)

rmses = sapply(lambdas, function(l) {
        
        mu_reg = mean(train_set$rating)
        
        b_i_reg = train_set %>%
                group_by(placeID) %>%
                summarize(b_i_reg = sum(rating - mu_reg) / (n()+1))
        
        b_u_reg = train_set %>%
                left_join(b_i_reg, by='placeID') %>%
                group_by(userID) %>%
                summarize(b_u_reg = sum(rating - b_i_reg - mu_reg) / (n() + l))
        
        predicted_ratings_b_i_u = test_set %>%
                left_join(b_i_reg, by='placeID') %>%
                left_join(b_u_reg, by='userID') %>%
                mutate(pred = mu_reg + b_i_reg + b_u_reg) %>%
                .$pred
        
        return(RMSE(test_set$rating, predicted_ratings_b_i_u))
})

qplot(lambdas, rmses)

lambda = lambdas[which.min(rmses)]
lambda

rmse_model3 = min(rmses)
rmse_model3



#different approach
#create copy of set

train_set.copy = train_set %>%
        select(userID, placeID, rating)

#create new variable that counts number of restaraunts rated by each user
train_set.copy = train_set.copy %>%
        group_by(userID) %>%
        mutate(n.rest_user = n())

#create new variable that counts number user that rated each restaurant
train_set.copy = train_set.copy %>%
        group_by(placeID) %>%
        mutate(n.users_rest = n())

#recast userID and placeID as factors
train_set.copy$userID = as.factor(train_set.copy$userID)
train_set.copy$placeID = as.factor(train_set.copy$placeID)

#do same process for test_set

test_set.copy = test_set %>%
        select(userID, placeID, rating)

#create new variable that counts number of restaraunts rated by each user
test_set.copy = test_set.copy %>%
        group_by(userID) %>%
        mutate(n.rest_user = n())

#create new variable that counts number user that rated each restaurant
test_set.copy = test_set.copy %>%
        group_by(placeID) %>%
        mutate(n.users_rest = n())

#recast userID and placeID as factors
test_set.copy$userID = as.factor(test_set.copy$userID)
test_set.copy$placeID = as.factor(test_set.copy$placeID)

h2o.init(
        nthreads = 1,
        max_mem_size='5G'
)

h2o.removeAll()

#partition
splits = h2o.splitFrame(as.h2o(train_set.copy),
                        ratios =.7,
                        seed = 1)

train = splits[[1]]
test = splits[[2]]

invisible(gc())

#first model
gbdt_first = h2o.gbm(x = c('placeID', 'userID', 'n.rest_user', 'n.users_rest'),
                     y = 'rating',
                     training_frame = train,
                     nfolds = 3)

summary(gbdt_first)

#model using only place and user
gbdt_mod2 = h2o.gbm(x = c('placeID', 'userID'),
                    y = 'rating',
                    training_frame = train,
                    nfolds = 3,
                    seed = 1,
                    keep_cross_validation_predictions = TRUE,
                    fold_assignment = 'Random')

summary(gbdt_mod2)

#evaluate on the test set
h2o.performance(gbdt_mod2, test)

predicted_ratings_gbdt = h2o.predict(gbdt_mod2, as.h2o(test_set.copy))

rmse_gbdt = RMSE(predicted_ratings_gbdt, as.h2o(test_set.copy$rating))
rmse_gbdt

#look at random forest
invisible(gc())

#random forest model
rf_mod1 = h2o.randomForest(
        training_frame = train,
        x = c('placeID', 'userID',  'n.rest_user', 'n.users_rest'),
        y = 'rating',
        ntrees = 50,
        max_depth = 20
)

summary(rf_mod1)

#random forest model with just place and user
rf_mod2 = h2o.randomForest(
        training_frame = train,
        x = c('placeID', 'userID'),
        y = 'rating',
        nfolds = 3,
        seed = 1,
        keep_cross_validation_predictions = TRUE,
        fold_assignment = 'Random'
)

summary(rf_mod2)

#model 1 has lower RMSE, so let's use that--evaluate on test
h2o.performance(rf_mod1, test)

#predict the ratings on test_set.copy, evaluate RMSE
predicted_ratings_rf_mod1 = h2o.predict(rf_mod1, as.h2o(test_set.copy))

rmse_rf = RMSE(predicted_ratings_rf_mod1, as.h2o(test_set.copy$rating))
rmse_rf


