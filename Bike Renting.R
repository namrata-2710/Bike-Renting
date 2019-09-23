library(ggplot2)
library(corrgram)
library(sampling)
library(rpart)
library(caret)
library(randomForest)
library(LinearRegressionMDE)
library(FNN)
data = read.csv("day.csv")
#Missing Value Analysis
missing_per = data.frame(apply(data,2,function(x){sum(is.na(x))}))#No missing value
#Outlier Analysis
#numerical columns
num_index = sapply(data,is.numeric)
num_data = data[,num_index]
cname = colnames(num_data)
for(i in 1:length(cname)){
plt <- assign(paste0("gn",i),ggplot(data = num_data, aes(y= cname[i], x = "cnt")))
plt + geom_boxplot(outlier.colour = 'red',outlier.fill = 'grey', outlier.shape =18 ) + theme_minimal() + labs(x = "cnt", y = cname[i])
}
  
gridExtra::grid.arrange(gn1,gn5,gn2, ncol=3)
gridExtra::grid.arrange(gn6,gn7,ncol=2)
gridExtra::grid.arrange(gn8,gn9,ncol=2)


#Feature Selection
corrgram(data[,num_index],order = F, upper.panel=panel.pie, text.panel=panel.txt, main = "correlation plot")

#temp and atemp are hoghly correlated, dropping atemp and we are removing date column as some columns are already extracted from it
data_del = subset(data, select = -c(atemp,dteday))

#Feature Scaling
# Casual and registered are unscaled
#Normality check
qqnorm(data_del$casual)
hist(data_del$casual)#left or positive skewed
#Applying Normalization
data_del[,"casual"]= (data_del[,"casual"]-min(data_del[,"casual"]))/(max(data_del[,"casual"])-min(data_del[,"casual"]))

qqnorm(data_del$registered)
hist(data_del$registered)# Normally Distributed
data_del[,"registered"]=(data_del[,"registered"]-mean(data_del[,"registered"]))/sd(data_del[,"registered"])


#Sampling
#Stratified Sampling Technique
set.seed(1234)
train.index = createDataPartition(data_del$holiday,p=0.70,list = FALSE)
train= data_del[train.index,]
test = data_del[-train.index,]

#Data Modelling
#Decision Tree
fit = rpart(cnt~.,data = train, method = "anova")
summary(fit)
predict_DT = predict(fit,test[,-14])
#MAPE
mape = function(x, xpred){
  mean(abs(x-xpred)/x)
}
mape(test[,14],predict_DT)#Error Rate 0.12%

#Random Forest
RF = randomForest(cnt~.,train,ntree = 500)
predict_RF = predict(RF,test[,-14])
mape(test[,14],predict_RF)#Error Rate 0.06%

#Linear Regression
LM = lm(cnt~.,data = train)
summary(LM)

predict_LM = predict(LM,test[,-14])
mape(test[,14],predict_LM)#Error Rate 1.23%

predict_knn <- knn.reg(train = train, test = test,train$cnt, k=10)
mape(test[,14],predict_knn)#Error Rate 0.06%

