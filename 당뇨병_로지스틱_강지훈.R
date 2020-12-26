library(nnet)
library(caret)
library(e1071)
df = read.csv("4194_2020fall/diabets_feautre5.csv")
df = df[,-c(1,2,4,5,6,7,9,16)]
str(df)
set.seed(1)
df$Outcome = factor(df$Outcome)

train.index = sample(c(1:dim(df)[1]), dim(df)[1]*0.6)
train.df = df[train.index,]
valid.df = df[-train.index,]


multi.result = multinom(Outcome~., data=train.df)
multi_train.predict = predict(multi.result, train.df)
table(multi_train.predict, train.df$Outcome)
confusionMatrix(multi_train.predict, train.df$Outcome)

multi_valid.predict = predict(multi.result, valid.df)
table(multi_valid.predict, valid.df$Outcome)
confusionMatrix(multi_valid.predict, valid.df$Outcome)
