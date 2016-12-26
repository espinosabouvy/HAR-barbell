limpia <- function(){
     setwd("~/DataScience/Proyectos/coursera/MachineLearning")
     
     library(data.table)
     library(caret)
     library(Amelia)
     library(dplyr)
     library(lubridate)
     
     setAs("character","fullDate", function(from) dmy_hm(from))
     setClass('fullDate')
     
     col.type.train <- c(user_name = "factor",
                      classe = "factor",
                      cvtd_timestamp = "fullDate",
                      new_window = "factor")
     raw.training <<- read.csv("pml-training.csv", colClasses = col.type.train, header = T, 
                           stringsAsFactors = F)

     col.type.test <- c(user_name = "factor",
                         cvtd_timestamp = "fullDate",
                         new_window = "factor")
     raw.testing <<- read.csv("pml-testing.csv", colClasses = col.type.test, header = T, 
                              stringsAsFactors = F)
     
     #cleaning
     par(mar = c(2,1,3,1))
     #missmap(raw.training, x.cex = 0.5, main = "Missing data in columns")
     par(mar=c(5.1,4.1,4.1,2.1))
         
     raw.training <- raw.training[colSums(!is.na(raw.training))/nrow(raw.training)==1]
     raw.testing <- raw.testing[colSums(!is.na(raw.testing))/nrow(raw.testing)==1]
     #eliminate kurtosis, skewness, max_yaw, min_yaw and amplitud_yaw
     raw.training <- raw.training[, sapply(raw.training, class) != "character"]
     raw.testing <- raw.testing[, sapply(raw.testing, class) != "character"]
     #remove first  7 cols (characteristics)
     raw.training <- raw.training[,8:ncol(raw.training)]
     raw.testing <- raw.testing[,8:ncol(raw.testing)]
     
     #create train and testing set
     inTrain <- createDataPartition(raw.training$classe, p = 0.75, list = F)
     dat.train <<- raw.training[inTrain,]
     dat.test <<- raw.training[-inTrain,]
}

explore <- function(){
     
     #valores con alta frecuencia o pocos valores únicos - la funcion + 
     #descarta las columnas que tienen estas 2 condiciones
     nearZeroVar(dat.train)
     
     #alta correlacion entre las variables también causa inestabilidad
     # y la selección de cual variable usar se vuelve aleatoria
     cormat <- cor(dat.train[,-ncol(dat.train)])
     highCorr <- findCorrelation(cormat, 0.9)
     dat.train <<- dat.train[,-highCorr]
     dat.test <<- dat.test[,-highCorr]
     
     #preprocess
     params <- preProcess(dat.train, method = "pca")
     proc.dat.train <<- predict(params, dat.train)
     proc.dat.test <<- predict(params, dat.test)
     
     #only to see pca
     pca <<- prcomp(dat.train[,-ncol(dat.train)])
     print(summary(pca))
     plot(pca, type = "l")
}

fittings <- function(){
     meth <- c("gbm","rf","lm")
     
     
     set.seed(88)

     #default params
     fit1 <<- train(classe~., method = "gbm", data = dat.train)
     pred1 <<- predict(fit1, dat.test)
     errors1 <<- confusionMatrix(pred1, dat.test$classe)
     
     #better
     bootControl <- trainControl(number = 5)
     gbmgrid <- expand.grid(interaction.depth = c(4,5), n.trees = c(150,200), 
                            shrinkage = 0.1, n.minobsinnode = c(1))
     fit2 <<- train(classe~., method = "gbm", data = dat.train ,
                   tuneGrid = gbmgrid,
                   trControl = bootControl)
     pred2 <<- predict(fit2, dat.test)
     errors2 <<- confusionMatrix(pred2, dat.test$classe)
     
     

}
