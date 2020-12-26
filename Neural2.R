library(neuralnet)

accidents.df = read.csv("4194_2020fall/accidentsnn_ch11_size999.csv", na.strings = "")
dim(accidents.df)
head(accidents.df)
str(accidents.df)
summary(accidents.df)

# min max function써서 수치 바꿔야 할 것임.
# Profil도 factor 변수인데 숫자로 읽었으니깐, factor 변수로 읽었어도 변환시켜서 더미 변수 만들어서 들어가야 하니깐 괜찮을 것임.
# 레벨에 따라서 각각의 더미변수가 나와야 함. SUR도 컬럼이 5개 컬럼으로 나와야 함.
# 더미 변수를 만들어 줘야 하는 것임.

vars = c("ALCHL_I", "PROFIL_I_R", "VEH_INVL")

set.seed(99)

train.index = sample(rownames(accidents.df), dim(accidents.df)[1]*0.6)
valid.index = setdiff(row.names(accidents.df), train.index)
library(nnet)

# SUR_COND -> 5개의 class가 있음.
# 거기에 class.ind를 하니 콜룸이 5개가 생김
# factor의 class 가 3개면 3개로 나누고 하나에만 1을 적고 나머지 0으로 적게끔 만들어 줘야 함.
# cbind를 써서 모든 

head(accidents.df[train.index,]$SUR_COND)
head(class.ind(accidents.df[train.index,]$SUR_COND))

head(accidents.df[train.index,]$MAX_SEV_IR)
head(class.ind(accidents.df[train.index,]$MAX_SEV_IR))

trainData = cbind(accidents.df[train.index, c(vars)],
                  class.ind(accidents.df[train.index,]$SUR_COND),
                  class.ind(accidents.df[train.index,]$MAX_SEV_IR))


# 1, 2, 3, 4, 9의 값들을 가지고 
head(trainData)
names(trainData) = c(vars,
                    paste("SUR_COND_", c(1, 2, 3, 4, 9), sep=""),
                    paste("MAX_SEV_IR_", c(0, 1, 2), sep=""))
head(trainData)

validData = cbind(accidents.df[valid.index, c(vars)],
                  class.ind(accidents.df[valid.index,]$SUR_COND),
                  class.ind(accidents.df[valid.index,]$MAX_SEV_IR))
names(validData) = c(vars,
                     paste("SUR_COND_", c(1,2,3,4,9), sep=""),
                     paste("MAX_SEV_IR_", c(0,1,2), sep=""))
head(validData)

nn1 = neuralnet(MAX_SEV_IR_0 + MAX_SEV_IR_1 + MAX_SEV_IR_2 ~ ALCHL_I + PROFIL_I_R + VEH_INVL + SUR_COND_1 + SUR_COND_2
                + SUR_COND_3 + SUR_COND_4, data= trainData, hidden = 2)
plot(nn1)

nn2 = neuralnet(MAX_SEV_IR_0 + MAX_SEV_IR_1 + MAX_SEV_IR_2 ~
          ALCHL_I + PROFIL_I_R + VEH_INVL + SUR_COND_1 + SUR_COND_2 + SUR_COND_3 + SUR_COND_4, 
                data = trainData, hidden = c(2,2))
plot(nn2)

library(caret)
head(trainData)
training.prediction = compute(nn1, trainData[,-c(8:11)])
training.prediction$net.result

training.class = apply(training.prediction$net.result, 1, which.max)-1
head(training.class, n=20)
confusionMatrix(factor(training.class), factor(accidents.df[train.index,]$MAX_SEV_IR))

validation.prediction = compute(nn1, validData[,-c(8:11)])
validation.class = apply(validation.prediction$net.result,1,which.max)-1

