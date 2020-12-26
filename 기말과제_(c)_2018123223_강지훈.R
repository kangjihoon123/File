# 1. 데이터 불러오기
df = read.csv("4194_2020fall/WA_Fn-UseC_-Telco-Customer-Churn.csv", na.strings ="", stringsAsFactors = TRUE)

#str(df)

# 2. 데이터 ID 열 제거 
df = df[,c(-1)]
#str(df)

# 3. 결측치 확인과 제거
sum(is.na(df))
df = na.omit(df)
sum(is.na(df))

# 4. 데이터 factor 화
df$SeniorCitizen<-factor(df$SeniorCitizen)

# 5. binning을 위한 점검
#hist(df$MonthlyCharges)
#hist(df$TotalCharges)
#hist(df$tenure)
quantile(df$MonthlyCharges, probs = seq(0, 1, 0.25))
quantile(df$TotalCharges, probs = seq(0, 1, 0.25))
quantile(df$tenure, probs = seq(0, 1, 0.25))


# 6. bining 후 factor 화와 split data
df$tenure = factor(round(df$tenure/10))
df$MonthlyCharges = factor(round(df$MonthlyCharges/10))
df$TotalCharges = factor(round(df$TotalCharges/1000))
#str(df)
#View(df)

# 
set.seed(2)
train.index = sample(c(1:dim(df)[1]), dim(df)[1]*0.6 )

train.df = df[train.index,]
valid.df = df[-train.index,]

dim(train.df)
dim(valid.df)
#str(train.df)

# 7. Navie Bayes 적용
library("e1071")
library("caret")
df.nb = naiveBayes(Churn~., data=train.df)

pred.train.class = predict(df.nb, newdata = train.df)
confusionMatrix(pred.train.class, train.df$Churn)

pred.valid.class = predict(df.nb, newdata = valid.df)
confusionMatrix(pred.valid.class, valid.df$Churn)

# 8. Decision Tree 적용

# 8.1 binning 된 기존 데이터 활용

library("rpart")
library("rpart.plot")
defalut.ct = rpart(Churn~., data=train.df, method="class")
prp(defalut.ct)
# str(train.df)

defalut.point.pred.train = predict(defalut.ct, train.df, type="class")
confusionMatrix(defalut.point.pred.train, train.df$Churn)

defalut.point.pred.valid = predict(defalut.ct, valid.df, type="class")
confusionMatrix(defalut.point.pred.valid, valid.df$Churn)

# 8.2 binning 전 Continuous 데이터 활용
before.df = read.csv("4194_2020fall/WA_Fn-UseC_-Telco-Customer-Churn.csv", na.strings ="", stringsAsFactors = TRUE)
#str(before.df)

# 8.2.1 데이터 전처리 및 split
before.df = before.df[,c(-1)]
sum(is.na(before.df))
before.df = na.omit(before.df)
sum(is.na(before.df))
before.df$SeniorCitizen<-factor(df$SeniorCitizen)
train.bf = before.df[train.index , ]
valid.bf = before.df[-train.index, ]

# 8.2.2 Decision Tree 적용
before.ct = rpart(Churn~., data=train.bf, method="class")
prp(before.ct)
#str(train.bf)

before.point.pred.train = predict(before.ct, train.bf, type="class")
confusionMatrix(before.point.pred.train, train.bf$Churn)

before.point.pred.valid = predict(before.ct, valid.bf, type="class")
confusionMatrix(before.point.pred.valid, valid.bf$Churn)

# 9.1 기존 데이터에 randomForest 적용
library(randomForest)
rf = randomForest(Churn~., data = train.df, ntree=500, importance = TRUE)

rf.pred.train = predict(rf, train.df)
confusionMatrix(rf.pred.train, train.df$Churn)

rf.pred.valid = predict(rf, valid.df)
confusionMatrix(rf.pred.valid, valid.df$Churn)

# 9.2 binning 전 Continuous 데이터 활용

rf = randomForest(Churn~., data = train.bf, ntree=500, importance = TRUE)
rf.pred.train = predict(rf, train.bf)
confusionMatrix(rf.pred.train, train.bf$Churn)

rf.pred.valid = predict(rf, valid.bf)
confusionMatrix(rf.pred.valid, valid.bf$Churn)

# 10. NN을 위한변수 7개 선택
varImpPlot(rf)

# MeanDecreaseAccuracy -> 선택
# Contract, tenure, MonthlyCharges, InternetService, OnlineSecurity
# TotalCharges, TechSupprot

# MeanDecreaseGINI
# tenure, MonthlyCharges, Contract, Paymentmethod, OnlineSecurity, InternetService
# Tech Support


# 11. MeanDecreaseAccuracy를 사용한 neuralnet을 위한 데이터 및 nnet 불러오기
library(nnet) 
library(neuralnet)

NeuralNet = read.csv("4194_2020fall/WA_Fn-UseC_-Telco-Customer-Churn.csv", na.strings ="", stringsAsFactors = TRUE)
#str(NeuralNet)
summary(NeuralNet)


# 12. 변수 선택, 결측치 제거
# Contract, tenure, MonthlyCharges, InternetService, OnlineSecurity
# TotalCharges, TechSupprot

NN = NeuralNet[,c(6,9,10,13,16,19,20,21)]
#str(NN)

sum(is.na(NN))
NN = na.omit(NN)
sum(is.na(NN))

# 13. 정규화
max(NN$TotalCharges)
NN$tenure = (NN$tenure-min(NN$tenure))/(max(NN$tenure)-min(NN$tenure))
NN$MonthlyCharges = (NN$MonthlyCharges-min(NN$MonthlyCharges))/(max(NN$MonthlyCharges)-min(NN$MonthlyCharges))
NN$TotalCharges = (NN$TotalCharges-min(NN$TotalCharges))/(max(NN$TotalCharges)-min(NN$TotalCharges))


#NN$tenure
#NN$MonthlyCharges
#NN$TotalCharges

#str(NN)

# 14. cbind와 paste를 이용한 더미화
NN.df1 = cbind(NN,
                 class.ind(NN$InternetService),
                 class.ind(NN$OnlineSecurity),  
                 class.ind(NN$TechSupport),
                 class.ind(NN$Contract),
                 class.ind(NN$Churn))
#NN.df1
vars = c("tenure", "InternetService", "OnlineSecurity", "TechSupport", "Contract"
         , "MonthlyCharges", "TotalCharges", "Churn")
names(NN.df1) = c(vars,
                    paste("Internet_", c(1, 2, 3), sep=""),
                    paste("Online_", c(1, 2, 3), sep=""),
                    paste("Tech_", c(1, 2, 3), sep=""),
                    paste("Contract_", c(1, 2, 3), sep=""),
                    paste("Churn_", c("no", "yes"), sep=""))

#NN.df1
colnames(NN.df1)

# 15. 더미화 된 변수 중 기존 변수 및 churn 제거
NN.df2 = NN.df1[, -c(2, 3, 4, 5, 8)]
colnames(NN.df2)

# 16. split data
set.seed(2)
train.index = sample(rownames(NN.df2), dim(NN.df2)[1]*0.6)
valid.index = setdiff(row.names(NN.df2), train.index)

train.NN = NN.df2[train.index,]
valid.NN = NN.df2[valid.index,]
dim(train.NN)
dim(valid.NN)
colnames(train.NN)

# 17. neuralnet 적용 및 정확도 확인
nn1 = neuralnet(Churn_no + Churn_yes~ 
                  tenure + MonthlyCharges + TotalCharges  +
                  Internet_1 + Internet_2 + Internet_3
                + Online_1 + Online_2 + Online_3 +
                  Tech_1 + Tech_2 + Tech_3 +
                Contract_1 +  Contract_2 + Contract_3
                  ,  data= train.NN, hidden = c(5,1), stepmax = 1e+06)
plot(nn1)

training.prediction = compute(nn1, train.NN[,-c(16:17)])
training.class = apply(training.prediction$net.result, 1, which.max) - 1
# training.class
NeuralNet$Churn = as.numeric(factor(NeuralNet$Churn)) - 1
confusionMatrix(factor(training.class), factor(NeuralNet[train.index,]$Churn))

validation.prediction = compute(nn1, valid.NN[,-c(16:17)])
validation.class = apply(validation.prediction$net.result, 1, which.max) -1
confusionMatrix(factor(validation.class), factor(NeuralNet[valid.index,]$Churn))

# 보너스: Accuracy 높이기 위한 전략

# 1. RandomForest에서 Ntree 값 조정

# 기존 데이터에 randomForest 적용 # Ntree 500 -> 400
library(randomForest)
rf = randomForest(Churn~., data = train.df, ntree=400, importance = TRUE)

# 0.9526(500) -> 0.9512(400)
rf.pred.train = predict(rf, train.df)
confusionMatrix(rf.pred.train, train.df$Churn)

# 0.8013(500) -> 0.8031 (400)
rf.pred.valid = predict(rf, valid.df)
confusionMatrix(rf.pred.valid, valid.df$Churn)

# binning 전 Continuous 데이터 활용 # Ntree 500 -> 400

rf = randomForest(Churn~., data = train.bf, ntree=400, importance = TRUE)

# 0.9839(500) -> 0.9834(400)
rf.pred.train = predict(rf, train.bf)
confusionMatrix(rf.pred.train, train.bf$Churn)

# 0.8091(500) -> 0.8041(400)
rf.pred.valid = predict(rf, valid.bf)
confusionMatrix(rf.pred.valid, valid.bf$Churn)

