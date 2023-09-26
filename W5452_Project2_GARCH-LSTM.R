.rs.restartR() # Start new session

reticulate::use_python("~/miniforge3/envs/rstudio-tf-2.8/bin/python")
reticulate::use_condaenv("rstudio-tf-2.8", conda="~/miniforge3/bin/conda", required = TRUE)

library(tensorflow)
library(keras)
tf_config() 
tf_version() 
tf$config$list_logical_devices()

#Loading the environmental time series dataset
amzn <- read.csv("AMZN.csv", header = TRUE, sep = ",", na.strings = c(""))

###
amzn$Date <- as.Date(amzn$Date)
amzn1 <- amzn[amzn$Date < "2020-01-01" & amzn$Date > "2015-01-01" ,]
amzn <- amzn1
###

#Displaying summary of the dataset

head(amzn)
str(amzn)
summary(amzn)

library(keras)
library(tensorflow)
library(longmemo)
library(ggplot2)
suppressMessages(devtools::source_gist(filename = "helper_funcs.R",
                                       id = "https://gist.github.com/Letmode/8c842b722ab31210ad6df64ef786a8c2"))

set_random_seed(10000)

price = amzn$Close[amzn$Close > 0]   
ret = diff(log(price))
ret = ret - mean(ret)
ret2 = ret^2
nret = length(ret)
Year = seq(2015, 2019 + 12 / 12, length.out = nret)

p = 1
q = 1 

n_te = 250
n_tr = nret - 250
ntrain = floor(0.7*nret) # size of training set
nndata = dataprep_nn_garch(ret, ntrain, p.max = p, q.max = q, fixed = TRUE)
nndata$m_garch@fit$matcoef

X_tr = nndata$Xtrain 
X_te = nndata$Xtest
Y_tr = nndata$Ytrain
Y_te = nndata$Ytest
Y0_te = nndata$Y0test # original test data (ret2)

dim(X_tr) = c(dim(X_tr), 1) # transform to array with one dimension

pq = p + q

model <- keras_model_sequential() %>%
  layer_lstm(units = 10,
             batch_input_shape = c(1, pq, 1),
             stateful = TRUE) %>%
  layer_dense(units = 1, activation = "relu")

#model <- keras_model_sequential() %>%
#  layer_lstm(units = 20,
#             batch_input_shape = c(1, pq, 1),
#             stateful = TRUE) %>%
#  layer_dense(units = 1, activation = "relu")

model %>%
  compile(loss = 'mse', optimizer = 'adam', metrics = 'mae')

summary(model)

model %>% fit(
  x = X_tr,
  y = Y_tr,
  batch_size = 1,
  epochs = 50,
  verbose = 1,
  shuffle = FALSE
)

X_pred = array(X_te, dim = c(dim(X_te), 1))
pred_out <- model %>% 
  predict(X_pred, batch_size = 1)

model %>% evaluate(X_tr, Y_tr, batch_size = 1)
model %>% evaluate(X_pred, Y_te, batch_size = 1) 

pred_out = pred_out * nndata$Ydiff + nndata$Ymin

ASE_LSTM = ASE_calc(Y0_te, pred_out)
ASE_GARCH = ASE_calc(Y0_te, nndata$Ytest_garch)
AAE_LSTM = AAE_calc(Y0_te, pred_out)
AAE_GARCH = AAE_calc(Y0_te, nndata$Ytest_garch)
CCP_LSTM = cor(Y0_te, pred_out)
CCP_GARCH = cor(Y0_te, nndata$Ytest_garch)

eval_crits_GARCH = data.frame("ASE" = c(ASE_LSTM, ASE_GARCH), "AAE" = c(AAE_LSTM, AAE_GARCH), 
                              "CCP" = c(CCP_LSTM, CCP_GARCH))
rownames(eval_crits_GARCH) = c("LSTM", "ARMA")
eval_crits_GARCH

Year.og = Year
Year.te = Year[(ntrain + 1):nret]
df_og <- as.data.frame(cbind(Year.og, ret2))
df_fcs <- as.data.frame(cbind(Year.te, pred_out, Y0_te, nndata$Ytest_garch))
colnames(df_fcs) <- c("Year", "pred.te", "real_data", "pred.arma")

plot_data = ggplot(df_og) +
  geom_line(aes(x = Year.og, y = ret2), color = "black") +
  labs(title = "AMZN squared return series", y = "", x = "Year") +
  theme(legend.text = element_text(size = 12), plot.title = element_text(size = 14),
        axis.title = element_text(size = 12))

plot_fcs = ggplot(df_fcs) +
  geom_line(aes(x = Year, y = real_data, color = "og")) +
  geom_line(aes(x = Year, y = pred.te, color = "lstm")) +
  geom_line(aes(x = Year, y = pred.arma, color = "arma")) +
  labs(title = "LSTM-predictions", y = "") +
  scale_color_manual("", values = c("og" = "black", "lstm" = "red", "arma" = "blue"),
                     labels = c("og" = "Test data", "lstm" = "LSTM-preds", "arma" = "GARCH-preds")) +
  theme(legend.text = element_text(size = 12), plot.title = element_text(size = 14),
        axis.title = element_text(size = 12), legend.position = c(0.85, 0.85))

ggpubr::ggarrange(plot_data, plot_fcs, ncol = 1)


