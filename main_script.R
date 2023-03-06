library(dplyr)
library(janitor)
library(ggplot2)

xdat <- read.csv('./Grp Project/bank-additional/bank-additional/bank-additional-full.csv', sep=';')
head(xdat)

xdat1 <- xdat %>% 
  mutate(response = ifelse(y=='no',0,1)) %>% 
  select(-c(y,duration))



model <- glm(response ~ .,family=binomial(link='logit'),data=xdat1)
summary(model)

xtmp <- xdat1 %>% select(-response)

fitted.results <- predict(model,xtmp,type='response')
fitted.results <- ifelse(fitted.results > 0.5,1,0)
misClasificError <- mean(fitted.results != xdat1$response)
print(paste('Accuracy',1-misClasificError))

table(xdat1$response, fitted.results)

## Old customers -------------------------------------------

xdat_old <- xdat1 %>% 
  filter(previous != 0) %>% 
  select(-c(previous, pdays))

head(xdat_old)

model <- glm(response ~ .,family=binomial(link='logit'),data=xdat_old)
summary(model)

xtmp <- xdat_old %>% select(-response)

fitted.results <- predict(model,xtmp,type='response')
fitted.results <- ifelse(fitted.results > 0.5,1,0)
misClasificError <- mean(fitted.results != xdat_old$response)
print(paste('Accuracy',1-misClasificError))

table(xdat_old$response, fitted.results)

### Feature selection -----------

model1 <- glm(response ~ age + contact + month +
                campaign + poutcome + `emp.var.rate`+
                `cons.price.idx` + `cons.conf.idx` + `nr.employed`,
              family=binomial(link='logit'),data=xdat_old)
summary(model1)

deviance(model1)

### ROC curves ----------------
library(pROC)
# Compute ROC curve and AUC
roc_obj <- roc(xdat_old$response, fitted.results)
auc <- auc(roc_obj)

# Plot ROC curve
plot(roc_obj, main = paste("ROC Curve (AUC =", auc, ")"))



xtmp <- xdat_old %>% select(-response)

fitted.results <- predict(model1,xtmp,type='response')
fitted.results <- ifelse(fitted.results > 0.5,1,0)
misClasificError <- mean(fitted.results != xdat_old$response)
print(paste('Accuracy',1-misClasificError))

table(xdat_old$response, fitted.results)

library(aod)
wald.test(Sigma = vcov(model1), b = coef(model1), Terms = 4:7)
wald.test(Sigma = vcov(model1), b = coef(model1), Terms = 2:3)
wald.test(Sigma = vcov(model), b = coef(model), Terms = 9:12)
wald.test(Sigma = vcov(model), b = coef(model), Terms = 13:14)



model2 <- glm(response ~ campaign + poutcome + `emp.var.rate`+`cons.price.idx` ,
              family=binomial(link='logit'),data=xdat_old)
summary(model2)

library(AICcmodavg)

models <- list(model, model1, model2)
mod.names <- c('All', 'Significant only','Significant 2')
aictab(cand.set = models, modnames = mod.names)

### Sigmoid --------------------

glm_emp <- glm(response ~ `emp.var.rate`, data=xdat_old, family=binomial(link='logit'))
summary(glm_emp)

emp <-seq (-92, 94, 0.01)

emp_rate <- predict(glm_emp, list(`emp.var.rate`=emp),type="response")

plot(emp, emp_rate, pch = 16, xlab = "Employment rate", ylab = "Conversion", main='Rate of change')

lines(emp, emp_rate, col = "red", lwd = 2)

coef <- glm_emp$coefficients
int <- coef[1]
slope <- coef[-1]

x = -2.5
est_prob = exp(int + slope * x)/(1 + exp(int + slope * x))
est_prob

ic_prob = slope * est_prob * (1 - est_prob)

## CPI
glm_cpi <- glm(response ~ `cons.price.idx`, data=xdat_old, family=binomial(link='logit'))
summary(glm_cpi)

emp <-seq (80, 120, 0.01)

emp_rate <- predict(glm_cpi, list(`cons.price.idx`=emp),type="response")

plot(emp, emp_rate, pch = 16, xlab = "CPI", ylab = "Conversion", main='Rate of change')

lines(emp, emp_rate, col = "red", lwd = 2)

coef <- glm_cpi$coefficients
int <- coef[1]
slope <- coef[-1]
x = 93
est_prob = exp(int + slope * x)/(1 + exp(int + slope * x))
est_prob

ic_prob = slope * est_prob * (1 - est_prob)
ic_prob


## New to brand -------------------------------------------------

xdat_new <- xdat1 %>% 
  filter(previous == 0) %>% 
  select(-c(previous,poutcome, pdays))
head(xdat_new)

model_new <- glm(response ~ .,family=binomial(link='logit'),data=xdat_new)
summary(model_new)


### SMOTE -------------------------
library(DMwR)
library(caret)

xdat_new <- xdat_new %>% mutate_if(is.integer, as.numeric) %>% 
  mutate_if(is.character, as.factor)
xdat_new$response <- as.factor(xdat_new$response)
xdat_smote <- SMOTE(response ~ ., xdat_new, perc.over = 200, k = 5)
table(xdat_smote$response)

model_new1 <- glm(response ~ .,family=binomial(link='logit'),data=xdat_smote)
summary(model_new1)

deviance(model_new1)

xtmp <- xdat_smote %>% select(-response)

fitted.results <- predict(model_new1,xtmp,type='response')
fitted.results <- ifelse(fitted.results > 0.5,1,0)
misClasificError <- mean(fitted.results != xdat_smote$response)
print(paste('Accuracy',1-misClasificError))

table(xdat_smote$response, fitted.results)
confusionMatrix(table(fitted.results, xdat_smote$response), positive = "1")$byClass[c("Precision", "Recall")]




