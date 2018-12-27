#Q1 ==========================================================================================================================================================

set.seed(42)
library(caret)
library(C50)

credit <- read.csv("credit_v2.csv")

#Data Cleaning
credit$default <- factor(credit$default)

#Shuffle dataset and split into train and test data - 90:10 split
s <- sample(1:nrow(credit))
credit <- credit[s,]
train_data <- credit[1:900,]
test_data <- credit[901:1000,]

help(trainControl)

# We aim to predict credit default using caret and C5.0 decision tree method. We train different models to get the best model for prediction.
buildCrossValidatedModel <- function(trainControl) { 
  # Trials specify the number of boosting iterations. A value of one indicates that a single model is used
  # We turn feature selection (winnow) to false
  tr_grid <- expand.grid(.trials = c(5:35), .model = c("tree"), .winnow = c(FALSE)) 
  model_cv <- train(default ~ ., data=train_data, method='C5.0', trControl=trainControl, tuneGrid = tr_grid)
  return(model_cv)
}

calculateAccuracy <- function(model){
  return(confusionMatrix(predict(model, test_data),test_data$default)$overall['Accuracy'])
}

createStratifedFolds <- function(y, folds){
  return(createFolds(train_data$default, folds, returnTrain = T))
}

# Function createFolds() creates stratified folds 
# 2-fold cross validation for model_cv1
folds = 2
tr_ctrl_cv1 <- trainControl(index = createStratifedFolds(train_data$default, folds), method = 'cv', number = folds )
model_cv1 <- buildCrossValidatedModel(tr_ctrl_cv1)
model_cv1
model_cv1_notstratified <- buildCrossValidatedModel(trainControl(method = 'cv', number = folds ))
model_cv1_notstratified
# 10-fold cross validation for model_cv2
folds = 10
tr_ctrl_cv2 <- trainControl(index = createStratifedFolds(train_data$default, folds), method = 'cv', number = folds )
model_cv2 <- buildCrossValidatedModel(tr_ctrl_cv2)
model_cv2
tr_ctrl_cv2 <- trainControl(method = 'cv', number = folds )
model_cv2_notstratified <- buildCrossValidatedModel(tr_ctrl_cv2)
model_cv2_notstratified
# 10-fold cross validation with repeats=5 for model_cv3
folds = 10
repeatCount = 5
tr_ctrl_cv3 <- trainControl(index = createStratifedFolds(train_data$default, folds), method = 'repeatedcv', number = folds, repeats = repeatCount )
model_cv3 <- buildCrossValidatedModel(tr_ctrl_cv3)
model_cv3
tr_ctrl_cv3 <- trainControl(method = 'repeatedcv', number = folds, repeats = repeatCount )
model_cv3_notstratified <- buildCrossValidatedModel(tr_ctrl_cv3)
model_cv3_notstratified
# 10-fold cross validation with selectionFunction = "oneSE" for model_cv4
# The simplest model within one standard error of the empirically optimal model, to avoid overfitting
folds = 10
tr_ctrl_cv4 <- trainControl(index = createStratifedFolds(train_data$default, folds), method = 'cv', number = folds, selectionFunction = "oneSE" )
model_cv4 <- buildCrossValidatedModel(tr_ctrl_cv4)
model_cv4
tr_ctrl_cv4 <- trainControl(method = 'cv', number = folds, selectionFunction = "oneSE" )
model_cv4_notstratified <- buildCrossValidatedModel(tr_ctrl_cv4)
model_cv4_notstratified
# 10-fold cross validation with selectionFunction = "tolerance" for model_cv5
# Tolerance takes the simplest model that is within a percent tolerance of the empirically optimal model. 
folds = 10
tr_ctrl_cv5 <- trainControl(index = createStratifedFolds(train_data$default, folds), method = 'cv', number = folds, selectionFunction = "tolerance" )
model_cv5 <- buildCrossValidatedModel(tr_ctrl_cv5)
model_cv5
tr_ctrl_cv5 <- trainControl(method = 'cv', number = folds, selectionFunction = "tolerance" )
model_cv5_notstratified <- buildCrossValidatedModel(tr_ctrl_cv5)
model_cv5_notstratified

#Optimal Trials of each model
model_cv1$finalModel$trials['Actual']
model_cv2$finalModel$trials['Actual']
model_cv3$finalModel$trials['Actual']
model_cv4$finalModel$trials['Actual']
model_cv5$finalModel$trials['Actual']
model_cv1_notstratified$finalModel$trials['Actual']
model_cv2_notstratified$finalModel$trials['Actual']
model_cv3_notstratified$finalModel$trials['Actual']
model_cv4_notstratified$finalModel$trials['Actual']
model_cv5_notstratified$finalModel$trials['Actual']

#Accuracy
print(calculateAccuracy(model_cv1))
print(calculateAccuracy(model_cv2))
print(calculateAccuracy(model_cv3))
print(calculateAccuracy(model_cv4))
print(calculateAccuracy(model_cv5))
print(calculateAccuracy(model_cv1_notstratified))
print(calculateAccuracy(model_cv2_notstratified))
print(calculateAccuracy(model_cv3_notstratified))
print(calculateAccuracy(model_cv4_notstratified))
print(calculateAccuracy(model_cv5_notstratified))

#Q2 ==========================================================================================================================================================
set.seed(42)
library(neuralnet)
library(onehot)
library(ROCR)
library(caret)

credit <- read.csv("credit_v2.csv")

#Data Cleaning
credit$default <- factor(credit$default)

#Shuffle dataset and split into train and test data 80:20 split
s <- sample(1:nrow(credit))
credit <- credit[s,]
train_data <- credit[1:800,]
test_data <- credit[801:1000,]

#Encode the categorical variables - neuralnet library doesnt like factors, nnet likes factors
#Standardize the predictors.
procValues <- preProcess(train_data, method = c("range"))
scaledTraindata <-  predict(procValues, train_data)
procValues <- preProcess(test_data, method = c("range"))
scaledTestdata <-  predict(procValues, test_data)
dummies <- dummyVars(~ ., data = scaledTraindata)
scaledTraindata_encoded <- predict(dummies, newdata = scaledTraindata)
dummies <- dummyVars(~ ., data = scaledTestdata)
scaledTestdata_encoded <- predict(dummies, newdata = scaledTestdata)

#Build the model
col_names <- colnames(scaledTraindata_encoded)
formula <- as.formula(paste('default.1+default.2 ~', paste(sprintf("`%s`", col_names[!col_names %in% c('default.1','default.2')]) , collapse = ' + ')))
model_nn <- neuralnet(formula, scaledTraindata_encoded, lifesign = 'full', act.fct = "logistic", linear.output = FALSE)
plot(model_nn)

#Predict using test data
results <- neuralnet::compute(model_nn, subset(scaledTestdata_encoded, select=-c(default.1,default.2)))

#AUC - with default as 1
pred <- ROCR::prediction(results$net.result[,1], scaledTestdata_encoded[,'default.1'])
performance(pred,"auc")@y.values

#AUC - with default as 2
pred <- ROCR::prediction(results$net.result[,2], scaledTestdata_encoded[,'default.2'])
performance(pred,"auc")@y.values

#Q3 ==========================================================================================================================================================
set.seed(42)
library(neuralnet)
library(onehot)
library(Metrics)
library(ggplot2)
library(caret)

#Get Data
diamond_train <- read.csv("diamonds_train.csv",stringsAsFactors = TRUE)
diamond_test <- read.csv("diamonds_test_no_label.csv",stringsAsFactors = TRUE)

#Feature Engineering
ggplot(as.data.frame(diamond_train$price), aes(x=diamond_train$price)) + geom_density() #Price data is right skewed
diamond_train$price <- log(diamond_train$price)
ggplot(as.data.frame(diamond_train$price), aes(x=diamond_train$price)) + geom_density() #Normally Distributed

#This function denormalizes data in the given range
denormalize <- function(x, min_x, max_x, a, b) { 
  return((((x-a) * (max_x - min_x))/(b-a)) + min_x)
}

#This function performs min-max normalization in the given range
normalize <- function(x, a, b){
  procValues <- preProcess(x, method = c("range"), rangeBounds = c(a, b))
  x_normalized <-  predict(procValues, x)
  return(x_normalized)
}

#This function encodes categorical variables using one-hot encoding (neuralnet library doesnt like factors)
encode <- function(x){
  dummies <- dummyVars(~ ., data = x)
  x_encoded <- predict(dummies, newdata = x)
  return(x_encoded)
}

#Split into training and validation set : 80:20 rule 
indexes <- sample(1:nrow(diamond_train),floor(0.8 * nrow(diamond_train)))
train_data <- diamond_train[indexes,]
validation_data <- diamond_train[-indexes,]

#Max-min scaling of numerical features and convert categorical variables into numerical variables
scaled_train_data <- normalize(train_data,-1,1)
scaled_train_data_encoded <- encode(scaled_train_data)
scaled_validation_data <- normalize(validation_data,-1,1)
scaled_validation_data_encoded <- encode(scaled_validation_data)
scaled_test_data <- normalize(diamond_test,-1,1)
scaled_test_data_encoded <- encode(scaled_test_data)

#clean up column names
colnames(scaled_train_data_encoded) = gsub(" ", "",colnames(scaled_train_data_encoded))
colnames(scaled_validation_data_encoded) = gsub(" ", "",colnames(scaled_validation_data_encoded))
colnames(scaled_test_data_encoded) = gsub(" ", "",colnames(scaled_test_data_encoded))

#Model building : hyperparameter tuning and model selection
col_names <- colnames(scaled_train_data_encoded)
tr_ctrl_cv <- trainControl(method = "cv", number = 3, selectionFunction = "oneSE")
formula <- as.formula(paste('price ~', paste(sprintf("`%s`", col_names[!col_names %in% c('price','X')]) , collapse = ' + ')))

#Tried combinations like 3,5,6,9,10,12,15,20 in each layer, when the node combination reaches 20,10 - starts to overfit where training RMSE is very low but validation RMSE is high
#Final 2 layer neural network model
model_cv_nn <- train(formula, data=scaled_train_data_encoded, method = "neuralnet", trControl = tr_ctrl_cv,
                     tuneGrid = expand.grid(layer1 = c(10), layer2 = c(10), layer3 = c(0)),
                     lifesign = 'full',
                     threshold = 0.02,
                     metric='RMSE',
                     linear.output = TRUE,
                     act.fct = 'tanh',
                     stepmax = 2e+05)

save(model_cv_nn, file = "model_cv_10_10_log.rda") #For future retrieval
load("model_cv_10_10_log.rda")
model_cv_nn

#Plot model
plot(model_cv_nn$finalModel)

getOrderedDataFrameFromNamedVector <- function(vector){
  df <- as.data.frame(vector)
  df$index <- as.numeric(row.names(df))
  df <- df[order(df$index), ]
}

#Calculate the Training RMSE value for the Neural Net Tanh Function
training_pred_scaled <- predict(model_cv_nn, subset(scaled_train_data_encoded, select=-c(price)))
training_pred_price <- exp(denormalize(training_pred_scaled, min(diamond_train$price), max(diamond_train$price),-1,1))
training_pred_price <- getOrderedDataFrameFromNamedVector(training_pred_price)
training_actual_price <-  exp(subset(train_data, select=c(price)))
training_actual_price <- getOrderedDataFrameFromNamedVector(training_actual_price)
plot(cbind(training_actual_price$price, training_pred_price$vector))
rmse <- rmse(training_actual_price, training_pred_price)
print(paste0('The Training RMSE value for the Neural Net Tanh Function is : ', round(rmse,3)))

#Predict and Calculate the Validation RMSE value for the Neural Net Tanh Function
validation_pred_scaled <- predict(model_cv_nn, subset(scaled_validation_data_encoded, select=-c(price)))
validation_pred_price <- exp(denormalize(validation_pred_scaled, min(diamond_train$price), max(diamond_train$price),-1,1))
validation_pred_price <- getOrderedDataFrameFromNamedVector(validation_pred_price)
validation_actual_price <-  subset(validation_data, select=c(price))
validation_actual_price <- getOrderedDataFrameFromNamedVector(validation_actual_price)
plot(cbind(validation_actual_price$price, validation_pred_price$vector))
rmse <- rmse(validation_actual_price, validation_pred_price)
print(paste0('The Validation RMSE value for the Neural Net Tanh Function is : ', round(rmse,3)))

#After determining the model , do the test prediction
test_pred_price_scaled <- predict(model_cv_nn, scaled_test_data_encoded)
test_pred_price <- exp(denormalize(test_pred_price_scaled, min(diamond_train$price), max(diamond_train$price),-1,1))

#Write output to file
write.csv(cbind(diamond_test$X, test_pred_price), file = 'A0074604J.csv', row.names = FALSE)

########################################END OF ASSIGNMENT#########################################################
#Further improvements for future
diamond_train$cut <- as.integer(diamond_train$cut, levels = c("Fair", "Good", "Very Good", "Premium", "Ideal"))
diamond_train$color <- as.integer(diamond_train$color, levels = c("J","I", "H", "G", "F", "E", "D"))
diamond_train$clarity <- as.integer(diamond_train$clarity, levels = c("I1", "SI2", "SI1","VS2", "VS1", "VVS2", "VVS1", "IF"))
diamond_test$cut <- as.integer(diamond_test$cut, levels = c("Fair", "Good", "Very Good", "Premium", "Ideal"))
diamond_test$color <- as.integer(diamond_test$color, levels = c("J","I", "H", "G", "F", "E", "D"))
diamond_test$clarity <- as.integer(diamond_test$clarity, levels = c("I1", "SI2", "SI1","VS2", "VS1", "VVS2", "VVS1", "IF"))
