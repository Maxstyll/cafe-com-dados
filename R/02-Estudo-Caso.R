# 02-Estudo-Caso - Construindo Um Modelo de Decisão Para Risco de Crédito

# Define pasta de trabalho
setwd("/mnt/sources/Estudos/cafe-com-dados/R")


# Calculando a entropi de duas calsses
-0.60 * log2(0.60) - 0.40 * log2(0.40)


# Gerando a cuva de Entropia
curve(-x * log2(x) - (1-x) * log2(1 - x), col= "red", xlab="x", ylab="Entropy", lwd =4)


# Carregar o dadaset
dados <- read.csv("../dados/db-banco-credito.csv")
str(dados)
View(dados)

# Verificar 2 atributos do cliente
table(dados$checking_balance)
table(dados$savings_balance)


# Verificar as caracteristicas do crédito
summary(dados$months_loan_duration)
summary(dados$amount)

# Variavel target
table(dados$default)


set.seed(123)
train_sample <- sample(1000, 900)
train_sample

training <- dados[train_sample, ]
validation <- dados[-train_sample, ]



# Verificando a proporcão da variável target
prop.table(table(training$default))
prop.table(table(validation$default))



# Construindo um modelo
install.packages("C50")
library(C50)
?C5.0

# Criando e visualizando o modelo
credit_model <- C5.0(training[-17], training$default)
credit_model


# Informacões detalhadas do modelo
summary(credit_model)


# Avaliandop a performace dp modelo
credit_pred <- predict(credit_model, validation)


# Confusion Matrix para comparar valores observados e valores previstos
install.packages("gmodels")
library(gmodels)

# Criandoa Confusion Matrix
?CrossTable
CrossTable(validation$default,
           credit_pred,
           prop.chisq = FALSE,
           prop.c = FALSE,
           prop.r = FALSE,
           dnn = c("Observado", "Previsto"))

# Melhorar a performace do modelo

# Aumentar a precisão com 10 tentativas
credit_model_10 <- C5.0(training[-17], training$default, trials = 10)
credit_model_10
summary(credit_model_10)


credit_pred10 <- predict(credit_model_10, validation)
CrossTable(validation$default,
           credit_pred10,
           prop.chisq = FALSE,
           prop.c = FALSE,
           prop.r = FALSE,
           dnn = c("Observado", "Previsto"))

# Dando pesos aos error

# Criand uma matriz de dimensões de custo
matrix_dimensions <- list(c("no", "yes"), c("no", "yes"))
names(matrix_dimensions) <- c("Previsto", "Observado")
matrix_dimensions

# Construindo a matriz
error_cost <- matrix(c(0, 1, 4, 0), nrow = 2, dimnames = matrix_dimensions)
error_cost


# Aplicar as pesos nos error
?C5.0
credit_cost_model <- C5.0(training[-17], training$default, costs = error_cost)
credit_pred10_cost <- predict(credit_cost_model, validation)

CrossTable(validation$default,
           credit_pred10_cost,
           prop.chisq = FALSE,
           prop.c = FALSE,
           prop.r = FALSE,
           dnn = c("Observado", "Previsto"))

CrossTable(validation$default,
           credit_pred,
           prop.chisq = FALSE,
           prop.c = FALSE,
           prop.r = FALSE,
           dnn = c("Observado", "Previsto"))


# Gerar o grafivo da arvore
install.packages("rpart.plot")
library(rpart)
library(rpart.plot)

credit_tree <- rpart(default ~ .,
                     data = training,
                     method = "class",
                     parms = list(split = "information"),
                     control = rpart.control(minsplit = 1)
                     )

?prp
prp(credit_tree, type = 0, extra = 1, under = TRUE, compress = TRUE)