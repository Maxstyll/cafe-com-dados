# 01-Estudo-Caso - Análise de Dados em Operadoras de Cartão de Crédito 

# Define pasta de trabalho
setwd("/mnt/sources/Estudos/R/cafe-com-dados")

# Instalar e carregar os pacotes
install.packages("mlbench")
install.packages("caret")
install.packages("e1071")

library(mlbench)
library(caret)
library(e1071)

# Carregar o dataset
dados <- read.csv("dados/db-cartao-credito.csv")
View(dados)

# Verifica o balanceamento das variáveis
skewness(dados$LIMIT_BAL)
histogram(dados$LIMIT_BAL)

# Sumariza os dados
summary(dados)
str(dados)


# 1 - Dados com vies
preProcessamentoParams <- preProcess(dados, method = c("BoxCox"))
print(preProcessamentoParams)

# Transformar o dataset usando os pârametros
transformed <- predict(preProcessamentoParams, dados)
myDadosTrans <- transformed


# Verifica o balanceamento das variáveis 1- respondido
str(myDadosTrans)
skewness(dados$LIMIT_BAL)
skewness(myDadosTrans$LIMIT_BAL)
histogram(myDadosTrans$LIMIT_BAL)


# 2 - Identificar as variaveis categoricas
myDadosTrans$default.payment.next.month <- factor(myDadosTrans$default.payment.next.month)
myDadosTrans$SEX <- factor(myDadosTrans$SEX)
myDadosTrans$EDUCATION <- factor(myDadosTrans$EDUCATION)
myDadosTrans$MARRIAGE <- factor(myDadosTrans$MARRIAGE)

myDadosTrans = na.omit(myDadosTrans)

str(myDadosTrans)


# Usa Random Forest para encontrar as variaveis mais relevantes
install.packages("randomForest")
library(randomForest)
rfModel = randomForest(myDadosTrans$default.payment.next.month ~ ., data = myDadosTrans, ntree=500)
varImpPlot(rfModel)


# Dividir os dados em treino e teste
row <- nrow(myDadosTrans)
row

set.seed(12345)
trainindex <- sample(row, 0.7*row, replace = FALSE)
training <- myDadosTrans[trainindex,]
validation <- myDadosTrans[-trainindex,]


# Preparar os datasets com as melhores variaveis preditoras
x_training <- training[, c(2,3,4,5)]
y_training <- training[, 24]


x_validation <- validation[, c(2,3,4,5)]
y_validation <- validation[, 24]


# Modelo KNN
knnModel = train(x=x_training, 
                 y=y_training, 
                 method = "knn", 
                 preProc=c("center", "scale"),
                 tuneLength = 10)

knnModel

# Plot da acurácia
plot(knnModel$results$k,
     knnModel$results$Accuracy,
     type="o",
     xlab="Numero de Vizinhos mais Próximos (K)",
     ylab="Acyrácia",
     main="Modelo KNN para Previsão de Concessão de Cartão de Crédito")


knnPred = predict(knnModel, newdata = x_validation)
knnPred
y_validation
