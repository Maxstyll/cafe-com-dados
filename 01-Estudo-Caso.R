# 01-Estudo-Caso - Análise de Dados em Operadoras de Cartão de Crédito 

# Define pasta de trabalho
setwd("/mnt/sources/Estudos/R/cafe-com-dados")

# Instalar e carregar os pacotes
#install.packages("mlbench")
#install.packages("caret")
#install.packages("e1071")
#install.packages("randomForest")

library(mlbench)
library(caret)
library(e1071)
library(randomForest)

# Carregar o dadaset
dados <- read.csv("dados/db-cartao-credito.csv")
View(dados)


# Verificar o balanceamento das variáveis
skewness(dados$LIMIT_BAL)
histogram(dados$LIMIT_BAL)


summary(dados)
str(dados)

# Identificar as variáveis categoricas
dados$default.payment.next.month <- factor(dados$default.payment.next.month)
dados$SEX <- factor(dados$SEX)
dados$EDUCATION <- factor(dados$EDUCATION)
dados$MARRIAGE <- factor(dados$MARRIAGE)
dados = na.omit(dados)


# Tratar variáveis com vies
preProcessParams <- preProcess(dados, method = c("BoxCox"))
print(preProcessParams)

# Transformacão do dataset usando os parametros
myDados <- predict(preProcessParams, dados)

# Verificar o balanceamento das variáveis
skewness(myDados$LIMIT_BAL)
histogram(myDados$LIMIT_BAL)


# Usa Random Foreste para identificar as variaveis mais relevantes
rfModel = randomForest(myDados$default.payment.next.month ~ ., data = myDados, ntree=500)
varImpPlot(rfModel)

# Dividir os dados em Treino e Test
row <- nrow(myDados)
row

set.seed(12345)
trainindex <- sample(row, 0.7*row, replace = FALSE)
training <- myDados[trainindex, ]
validation <- myDados[-trainindex,]

# Prepara ois dataser com as melhores variáveis preditoras
x_train <- training[, c(1,4,5,11,12,13,14,17)]
y_train <- training[, 24]


x_valid <- validation[, c(1,4,5,11,12,13,14,17)]
y_valid <- validation[, 24]

# Modelo KNN
knnModel = train(x=x_train,
                 y=y_train,
                 method="knn",
                 preProc=c("center", "scale"),
                 tuneLength = 10)

knnModel

#Plot da acurácia
plot(knnModel$results$k,
     knnModel$results$Accuracy,
     type="o",
     xlab="Nímero de Vizinhos mais próximos (K)",
     ylab="Acurácia",
     main="Modelo KNN para previsão de concessão de cartão de crédito")

knnPred = predict(knnModel, newdata = x_valid)
confusionMatrix(knnPred, y_valid)

  
  
  