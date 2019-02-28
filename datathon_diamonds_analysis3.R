#### Ubiqum Datathon Game
### Exploratory Data Analysis
## Alican Tanaçan

### Libraries ----
if (require(pacman) == FALSE) {
  install.packages('pacman')
}
pacman::p_load(tidyverse)

# import data 
diamondsdata <- read_csv2('train.csv')
validation <- read_csv2('validation.csv')

### Dataset information ----
?diamonds

### Data insights ----
# The diamonds with a bad cut are in average more expensive
diamondsdata %>% 
  ggplot(aes(cut, price)) + 
  geom_boxplot()

# The diamonds with a bad color are also more expensive
diamondsdata %>% 
  ggplot(aes(color, price)) + 
  geom_boxplot()

# And the diamonds with a bad clarity have a higer price
diamondsdata %>% 
  ggplot(aes(clarity, price)) +
  geom_boxplot()

### Your task ----

# Why the diamonds that have a fair cut, bad color and a bad clarity are,
# in median, more expensive? We would like to receive a model to predict 
# their price and to know which ones are the most relevant features to do that. 


### My Answers ----

# Libraries
library(ggplot2)
library(dplyr)
library(memisc)
library(lattice)
library(gridExtra)
library(grid)
library(GGally)
library(modelr)

# Data Exploration
summary(diamondsdata)
str(diamondsdata)
diamondsdata <- diamondsdata %>% mutate_at(c("cut",
                                             "color",
                                             "clarity"), as.factor)

# Removing Outliers
boxplot(diamondsdata$y)$out

boxplot(diamondsdata$z)$out

Y_Outliers <- boxplot(diamondsdata$y)$out
diamondsdata[which(diamondsdata$y %in% Y_Outliers), ]
diamondsdata1 <- diamondsdata[-which(diamondsdata$y %in% Y_Outliers), ]

Z_Outliers <- boxplot(diamondsdata1$z)$out
diamondsdata1[which(diamondsdata1$z %in% Z_Outliers), ]
diamondsdata2 <- diamondsdata1[-which(diamondsdata1$z %in% Z_Outliers), ]

# Data exploration
str(diamondsdata2)
plot(diamondsdata2$carat)
plot(diamondsdata2$price)
plot(diamondsdata2$cut)
plot(diamondsdata2$color)
plot(diamondsdata2$clarity)

ggplot(data=diamondsdata2) + geom_histogram(binwidth = 500, 
                                       aes(x=diamondsdata2$price)) + 
  ggtitle("Price Distribution") + 
  xlab ("Price") + 
  ylab("Frequency") + 
  theme_classic()

# Histogram for price and log10(price)
plot1 <- ggplot(diamondsdata2,aes(x=price))+
  geom_histogram(color='blue',fill = 'blue',binwidth=100)+
  scale_x_continuous(breaks=seq(300,19000,1000),limit=c(300,19000))+
  ggtitle('Price')
plot2 <- ggplot(diamondsdata2,aes(x=price))+
  geom_histogram(color='red',fill='red',binwidth=0.01)+
  scale_x_log10(breaks=seq(300,19000,1000),limit=c(300,19000))+
  ggtitle('Price(log10)')
grid.arrange(plot1,plot2,ncol=2)

# GGplot to see correlations
ggplot(diamondsdata2,aes(x=carat,y=price))+
  geom_point(color='blue',fill='blue')+
  ggtitle('Diamond price vs. carat')

ggpairs(diamondsdata2, params = c(shape=I('.'),outlier.shape=I('.')))

# Using the log function we are transforming the data from logarithmic to that of a 
# linear pattern which makes it easier to work with the data. 
diamondsdata3 <- diamondsdata2 %>%
  mutate(lprice = log2(price), lcarat = log2(carat))

ggplot(diamondsdata3, aes(lcarat, lprice)) + geom_hex(bins = 50) + 
  ggtitle("Log Transformation of Price versus Carat Weight") + 
  xlab("Log2 of Carat Weight") + 
  ylab("Log2 of Price")

ggcorr(diamondsdata3[,1:13])

# Linear model try
lmmodel1 <- lm(lprice ~ lcarat, data = diamondsdata3)
summary(lmmodel1)

# Generalized linear model try
glmmodel2 <- glm(lprice ~ lcarat, data = diamondsdata3)
summary(glmmodel2)

# Linear model grid and ggplot
grid1 <- diamondsdata3 %>%
  data_grid(carat = seq_range(carat, 20)) %>%
  mutate(lcarat = log2(carat)) %>%
  add_predictions(lmmodel1, "lprice") %>%
  mutate(price = 2 ^ lprice)

ggplot(diamondsdata3, aes(carat, price)) + 
  geom_hex(bins = 50) + 
  geom_line(data = grid1, color = "red", size = 1)

# GLM model grid and ggplot
grid2 <- diamondsdata3 %>%
  data_grid(carat = seq_range(carat, 20)) %>%
  mutate(lcarat = log2(carat)) %>%
  add_predictions(glmmodel2, "lprice") %>%
  mutate(price = 2 ^ lprice)

ggplot(diamondsdata3, aes(carat, price)) + 
  geom_hex(bins = 50) + 
  geom_line(data = grid2, color = "green", size = 1)

# Best linear models
lm1 <- lm(I(log2(price)) ~ I(carat^(1/3)), data = diamondsdata3)
lm2 <- update(lm1,~ . +carat)
lm3 <- update(lm2,~ . +cut)
lm4 <- update(lm3,~ . +color)
lm5 <- update(lm4,~ . +clarity)
mtable(lm1,lm2,lm3,lm4,lm5)

summary(lm5)

# Predict on validation
validpred <- predict(lm5, validation)
2^validpred

validation["PredictedPrice"] <- 2^validpred

validationrds <- readRDS("validation_price.rds")

validation["TruePrice"] <- validationrds$price

postResample(validation$PredictedPrice, validation$TruePrice)
