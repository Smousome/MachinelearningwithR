#Installation of necessary libraries

install.packages("Amelia")
install.packages("dplyr")
install.packages("ggplot2")
install.packages("ggpubr")
install.packages("corrplot")
install.packages("neuralnet")
install.packages("Metrics")

#Importing the Life Expectancy dataset

life <- read.csv('Life_Expectancy.csv', header = TRUE, sep = ",", na.strings = c(""))
print(head(life))
str(life)
summary(life)


#Find missing values

library("Amelia")
missmap(life, main = 'Missing Map', col = c('yellow','black'), legend = FALSE)

life_complete <- na.omit(life)
print(head(life_complete))
str(life_complete)
summary(life_complete)

#Keeping only numeric and integer variables for analysis

library(dplyr)
life_complete <- select(life_complete,-Country,-Year, -Status)
head(life_complete)
str(life_complete)
summary(life_complete)

#Find missing values after removing missing values and irrelevant variables

library("Amelia")
missmap(life_complete, main = 'Missing Map after data pre-processing', col = c('yellow','black'), legend = FALSE)

###Plots for understanding the data

library(ggplot2)
library(ggpubr)

fig1 <- ggplot(life_complete, aes(Life.expectancy)) + geom_histogram(bins = 20, alpha = 0.5, fill = 'blue')
fig2 <- ggplot(life_complete, aes(Adult.Mortality)) + geom_histogram(bins = 20, alpha = 0.5, fill = 'blue')
fig3 <- ggplot(life_complete, aes(infant.deaths)) + geom_histogram(bins = 20, alpha = 0.5, fill = 'blue')
fig4 <- ggplot(life_complete, aes(Alcohol)) + geom_histogram(bins = 20, alpha = 0.5, fill = 'blue')
fig5 <- ggplot(life_complete, aes(percentage.expenditure)) + geom_histogram(bins = 20, alpha = 0.5, fill = 'blue')
fig6 <- ggplot(life_complete, aes(Hepatitis.B)) + geom_histogram(bins = 20, alpha = 0.5, fill = 'blue')
fig7 <- ggplot(life_complete, aes(Measles)) + geom_histogram(bins = 20, alpha = 0.5, fill = 'blue')
fig8 <- ggplot(life_complete, aes(BMI)) + geom_histogram(bins = 20, alpha = 0.5, fill = 'blue')
fig9 <- ggplot(life_complete, aes(under.five.deaths)) + geom_histogram(bins = 20, alpha = 0.5, fill = 'blue')
fig10 <- ggplot(life_complete, aes(Polio)) + geom_histogram(bins = 20, alpha = 0.5, fill = 'blue')
fig11 <- ggplot(life_complete, aes(Total.expenditure)) + geom_histogram(bins = 20, alpha = 0.5, fill = 'blue')
fig12 <- ggplot(life_complete, aes(Diphtheria)) + geom_histogram(bins = 20, alpha = 0.5, fill = 'blue')
fig13 <- ggplot(life_complete, aes(HIV.AIDS)) + geom_histogram(bins = 20, alpha = 0.5, fill = 'blue')
fig14 <- ggplot(life_complete, aes(GDP)) + geom_histogram(bins = 20, alpha = 0.5, fill = 'blue')
fig15 <- ggplot(life_complete, aes(Population)) + geom_histogram(bins = 20, alpha = 0.5, fill = 'blue')
fig16 <- ggplot(life_complete, aes(thinness..1.19.years)) + geom_histogram(bins = 20, alpha = 0.5, fill = 'blue')
fig17 <- ggplot(life_complete, aes(thinness.5.9.years )) + geom_histogram(bins = 20, alpha = 0.5, fill = 'blue')
fig18 <- ggplot(life_complete, aes(Income.composition.of.resources)) + geom_histogram(bins = 20, alpha = 0.5, fill = 'blue')
fig19 <- ggplot(life_complete, aes(Schooling)) + geom_histogram(bins = 20, alpha = 0.5, fill = 'blue')

ggarrange(fig1,fig2,fig3,fig4,fig5,fig6,fig7,fig8,fig9, fig10, fig11, fig12, 
          fig13, fig14, fig15, fig16, fig17, fig18, fig19, ncol = 4, nrow = 5, 
          align=("v"))

###Examine correlation matrix

library("corrplot")

corrplot(cor(life_complete), method = "color",
         col = colorRampPalette(c("red","white","blue"))(10),
         tl.cex = 0.5)

#Applying normalization to the dataset using min-max method
max_data <- apply(life_complete, 2, max)
min_data <- apply(life_complete, 2, min)
data_scaled <- scale(life_complete,center = min_data, scale = max_data - min_data)

#Creating training and test data
life_train <- as.data.frame(data_scaled[1:1154, ])
life_test <- as.data.frame(data_scaled[1155:1649, ])

##FFNN model 1: train the naive FFNN model with 1 Hidden Layer and 1 Node
library(neuralnet)

set.seed(12345) # in order to reproduce results
life_model1 <- neuralnet(formula = Life.expectancy ~ ., data = life_train, 
                        linear.output = T)
# visualize the network topology
plot(life_model1, information = T, show.weights = F, rep = "best")

#Predict output variable in test set
predict_life1 <- predict(life_model1,life_test[,2:19])

#re-transform the predict data (scaled) to the original level
predict_life_start1 <- predict_life1*(max_data[19]-min_data[19])+min_data[19]

#re-transform the scaled output variable to the original level
test_start1 <- as.data.frame((life_test$Life.expectancy)*(max_data[19]-min_data[19])+min_data[19])

#Evaluate the model with 1 hidden layer and 1 node

library(Metrics)
MSE.model_results1 <- sum((test_start1 - predict_life_start1)^2)/nrow(test_start1)
MSE.model_results1

cor.life_model1 = cor(predict_life_start1,test_start1 )
cor.life_model1

##FFNN model 2: train the FFNN model with 1 Hidden Layer and 10 Nodes

set.seed(12345) # in order to reproduce results
life_model2 <- neuralnet(formula = Life.expectancy ~ ., data = life_train, 
                        hidden = 10, linear.output = T)
# visualize the network topology
plot(life_model2, information = T, show.weights = F, rep = "best")

#Predict output variable in test set
predict_life2 <- predict(life_model2,life_test[,2:19])

#re-transform the predict data (scaled) to the original level
predict_life_start2 <- predict_life2*(max_data[19]-min_data[19])+min_data[19]

#re-transform the scaled output variable to the original level
test_start2 <- as.data.frame((life_test$Life.expectancy)*(max_data[19]-min_data[19])+min_data[19])

#Evaluate the model with 1 hidden layer and 10 nodes

library(Metrics)
MSE.model_results2 <- sum((test_start2 - predict_life_start2)^2)/nrow(test_start2)
MSE.model_results2

cor.life_model2 = cor(predict_life_start2,test_start2 )
cor.life_model2

##FFNN model 3: train the FFNN model with 2 Hidden Layer. The first layer has 10 nodes and the second layer has 5 nodes.

set.seed(12345) # in order to reproduce results
life_model3 <- neuralnet(formula = Life.expectancy ~ ., data = life_train, 
                         hidden = c(10,5), linear.output = T)
# visualize the network topology
plot(life_model3, information = T, show.weights = F, rep = "best")

#Predict output variable in test set
predict_life3 <- predict(life_model3,life_test[,2:19])

#re-transform the predict data (scaled) to the original level
predict_life_start3 <- predict_life3*(max_data[19]-min_data[19])+min_data[19]

#re-transform the scaled output variable to the original level
test_start3 <- as.data.frame((life_test$Life.expectancy)*(max_data[19]-min_data[19])+min_data[19])

#Evaluate the model with 2 hidden layers

library(Metrics)
MSE.model_results3 <- sum((test_start3 - predict_life_start3)^2)/nrow(test_start3)
MSE.model_results3

cor.life_model3 = cor(predict_life_start3,test_start3 )
cor.life_model3

##FFNN model 4: train the neuralnet model with 3 Hidden Layer. The first layer has 10 nodes, the second layer has 10 nodes, and the third layer has 5 nodes.

set.seed(12345) # in order to reproduce results
life_model4 <- neuralnet(formula = Life.expectancy ~ ., data = life_train, 
                         hidden = c(10,10,5), linear.output = T)

# visualize the network topology
plot(life_model4, information = T, show.weights = F, arrow.length = 0.15, 
     rep = "best" )

#Predict output variable in test set
predict_life4 <- predict(life_model4,life_test[,2:19])

#re-transform the predict data (scaled) to the original level
predict_life_start4 <- predict_life4*(max_data[19]-min_data[19])+min_data[19]

#re-transform the scaled output variable to the original level
test_start4 <- as.data.frame((life_test$Life.expectancy)*(max_data[19]-min_data[19])+min_data[19])

#Evaluate the model with 3 hidden layer

library(Metrics)
MSE.model_results4 <- sum((test_start4 - predict_life_start4)^2)/nrow(test_start4)
MSE.model_results4

cor.life_model4 = cor(predict_life_start4,test_start4 )
cor.life_model4

## Model 5: Creating a Multiple Linear Regression model for comparison
#Creating training and test data for regression analysis (avoiding feature-scaled data)
lm_train <- life_complete[1:1154, ]
lm_test <- life_complete[1155:1649, ]
Regression_Model <- lm(Life.expectancy~., data = lm_train)
summary(Regression_Model)

#Evaluating the Regression Model
predict_lm <- predict(Regression_Model,lm_test[2:19])
MSE.lm <- sum((predict_lm - lm_test$Life.expectancy)^2)/nrow(lm_test)
MSE.lm

cor.lm = cor(predict_lm,lm_test$Life.expectancy)
cor.lm

#Overview of all the MSE for comparison
MSE.model_results1; MSE.model_results2; MSE.model_results3; MSE.model_results4; MSE.lm

#Overview of all Correlation values of predicted results and test set results
cor.life_model1;cor.life_model2;cor.life_model3;cor.life_model4;cor.lm
