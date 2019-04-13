### ------ Mushroom Contest ------ ###
### ----- by Alican Tanaçan ------ ###
### ------- 12 April 2019 -------- ###

### ---- Libraries ----
if(require("pacman") == "FALSE"){
  install.packages("pacman")}
p_load(dplyr, ggplot2, plotly, naniar, tidyverse,
       devtools, corrplot, GGally, caret, e1071,
       kernlab, randomForest, gridExtra, caTools, 
       rpart, C50, MLmetrics)

### ---- Import Data ----
Mushroom <- readRDS("mushroom.rds")

### ---- Data Exploration ----
str(Mushroom)
summary(Mushroom)

## Change all data types to factor
Mushroom <- data.frame(lapply(Mushroom, as.factor))

### ---- Preprocessing ----
## Selecting variables that are discoverable by looking to a picture
Mushroom %>%  select(c("class", 
                       "cap.shape",
                       "cap.color", 
                       "bruises",
                       "stalk.color.above.ring",
                       "stalk.color.below.ring",
                       "population")) -> MushroomPic

MushroomPic$bruises <- as.logical(MushroomPic$bruises)

# Only take unique rows and remove duplicated rows
MushroomPic <- unique(MushroomPic)

### ---- Data Partition ----
## Creating Train Set for 75%
set.seed(456)
MushSample <- sample.split(MushroomPic, SplitRatio = 0.75)
mushtrain1 <- subset(MushroomPic, MushSample == T)
mushtest1 <- subset(MushroomPic, MushSample == F)

## Creating Train Set for 80%
set.seed(375)
trainIndex <- createDataPartition(MushroomPic$class, 
                                  p = .8,
                                  list = F,
                                  times = 1)
mushtrain2 <- MushroomPic[ trainIndex,]
mushtest2 <- MushroomPic[-trainIndex,]

### ---- Decision Tree Modelization ----
## Setting Train Control
DTtrctrl <- trainControl(method = "repeatedcv", 
                         number = 10, 
                         repeats = 2,
                         classProbs = T)

## Decision Tree on mushtrain1
set.seed(251)
DTmodel1 <- train(class ~ .,
                  data = mushtrain1,
                  method = "rpart",
                  trControl = DTtrctrl,
                  tuneLength = 5)

plot(DTmodel1)

varImp(DTmodel1)

predDTmodel1 <- predict(DTmodel1, mushtest1)

postResample(predDTmodel1, mushtest1$class) -> DTmodel1metrics

DTmodel1metrics
# Accuracy: 0.885
# Kappa: 0.754

confusionMatrix(predDTmodel1, mushtest1$class)
  
## Decision Tree on mushtrain2
set.seed(802)
DTmodel2 <- train(class ~ .,
                  data = mushtrain2,
                  method = "rpart",
                  parms = list(split = "gini"),
                  trControl = DTtrctrl,
                  tuneLength = 5)

plot(DTmodel2)

varImp(DTmodel2)

predDTmodel2 <- predict(DTmodel2, mushtest2)

postResample(predDTmodel2, mushtest2$class) -> DTmodel2metrics

DTmodel2metrics
# Accuracy: 0.935
# Kappa: 0.867

confusionMatrix(predDTmodel2, mushtest2$class)

### ---- C5.0 Modelization ----
## Setting Train Control and Grid
C50trctrl <- trainControl(method = "repeatedcv", 
                          number = 10, 
                          repeats = 2,
                          summaryFunction = multiClassSummary,
                          classProbs = TRUE)

C50Grid <- expand.grid(.model="tree",.trials = c(1:10),.winnow = FALSE)

## C50 on mushtrain1
C50model1 <- train(class~.,
                  data = mushtrain1,
                  method = "C5.0",
                  metric = "Sensitivity",
                  tuneGrid = C50Grid,
                  trControl = C50trctrl)

plot(C50model1)

varImp(C50model1)

predC50model1 <- predict(C50model1, mushtest1)

postResample(predC50model1, mushtest1$class) -> C50model1metrics

C50model1metrics
# Accuracy: 0.903
# Kappa: 0.795

confusionMatrix(predC50model1, mushtest1$class)

## C50 on mushtrain2
C50model2 <- train(class~.,
                   data = mushtrain2,
                   method = "C5.0",
                   metric = "Accuracy",
                   tuneGrid = C50Grid,
                   trControl = C50trctrl)

plot(C50model2)

varImp(C50model2)

predC50model2 <- predict(C50model2, mushtest2)

postResample(predC50model2, mushtest2$class) -> C50model2metrics

C50model2metrics
# Accuracy: 0.903
# Kappa: 0.795

confusionMatrix(predC50model2, mushtest2$class)

### ---- Random Forest Modelization ----
## Setting Train Control
RFtrctrl1 <- trainControl(method = "repeatedcv",
                         number = 10,
                         repeats = 2,
                         summaryFunction = multiClassSummary,
                         classProbs = T)

RFgrid1 <- expand.grid(mtry = c(1:5))

## Random Forest on mushtrain1
RFmodel1 <- train(class ~ ., 
                  data = mushtrain1,
                  method = "rf",
                  metric = "ROC",
                  trControl = RFtrctrl1,
                  tuneGrid = RFgrid1,
                  tuneLenght = 2)


plot(RFmodel1)
varImp(RFmodel1)
predRFmodel1 <- predict(RFmodel1, mushtest1)
postResample(predRFmodel1, mushtest1$class) -> RFmodel1metrics
RFmodel1metrics
confusionMatrix(predRFmodel1, mushtest1$class)

## Random Forest on mushtrain2
RFtrctrl2 <- trainControl(method = "repeatedcv",
                          number = 10,
                          repeats = 2,
                          summaryFunction = twoClassSummary,
                          classProbs = T)

RFgrid2 <- expand.grid(mtry = c(1:5))

RFmodel2 <- train(class ~ ., 
                  data = mushtrain2,
                  method = "rf",
                  metric = "ROC",
                  trControl = RFtrctrl2,
                  tuneGrid = RFgrid2,
                  tuneLenght = 2)

plot(RFmodel2)
varImp(RFmodel2)
predRFmodel2 <- predict(RFmodel2, mushtest2)
postResample(predRFmodel2, mushtest2$class) -> RFmodel2metrics
RFmodel2metrics
confusionMatrix(predRFmodel2, mushtest2$class)


### ---- Save the best model for validation ----
save(RFmodel1, file = "RFModel1.rda")
save(RFmodel2, file = "RFModel2.rda")
