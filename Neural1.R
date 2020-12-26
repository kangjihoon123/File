tiny.df = data.frame(obs=c(1:6), Fat=c(0.2, 0.1, 0.2, 0.2,
                                       0.4, 0.3), Salt = c(0.9, 0.1, 0.4, 0.5, 0.5, 0.8),
                     Acceptance = c("like", "dislike", "dislike", "dislike", "like", "like"))
# install.packages("neuralnet")
library(neuralnet)

tiny.df$Like = tiny.df$Acceptance == "like"
tiny.df$Dislike = tiny.df$Acceptance == "dislike"

head(tiny.df)

# Fat, Salt를 predictor로 쓰고, Like와 Dislike를 아웃컴 변수로 사용 함.
tiny.nn = neuralnet(Like + Dislike ~ Salt + Fat, data = tiny.df,
                    linear.output = F, hidden = 3)

#c(2,3)으로 쓰면 첫 번째 히든 레이어 node 2개, 두번째 노드 3개)

plot(tiny.nn)

# weight이 어떤 의미를 갖는지 알 수 없다보니, 찾아서 의미를 부여를 할 수 없음.
tiny.nn$weights
library(caret)

# 다른 기법과는 다르게 predict대신 compute를 사용 함.
tiny.nn_predict = compute(tiny.nn, data.frame(tiny.df$Salt, tiny.df$Fat))

tiny.nn_predict
# net result는 어떤 식으로 돼 있냐면, 두 개의 노드가 있는데
# 첫번째 레코드에 마지막으로 예측한 값이 있음.
# 결과값을 깔끔하게 보려고 
# 1 record는 1에 가깝고, dislike는 0.07정도로 굉장히 멈
# 두 번째 record는 dislike에 가까움.
tiny.nn_predict$net.result
round(tiny.nn_predict$net.result, digits =5)


# which.max() 함수 사용 -> 어떤 위치에 있는 것이 가장 큰 것인가를 알려줌
# apply를 사용해서 표현

apply(tiny.nn_predict$net.result, 1#모든 레코드에 적용
      , which.max)

# -1을 해줬기에 0이면 like, 1이면 dislike임.
# confusionmatrix 할 때 factor 변환 잘 해줘야 함.
# 우리는 0 아니면 1이 되기를 원해 그래서 아래 식 사용

tiny.predicted.class = apply(tiny.nn_predict$net.result,1 ,which.max) -1
tiny.predicted.class

# 변환하기
ifelse(tiny.predicted.class=="1", "dislike", "like")
factor(ifelse(tiny.predicted.class== "1", "dislke", "like"))
tiny.df$Acceptance
str(tiny.df$Acceptance)

confusionMatrix(factor(ifelse(tiny.predicted.class =="1",
              "dislike", "like")), factor(tiny.df$Acceptance))

# Class가 3개, prediction.
