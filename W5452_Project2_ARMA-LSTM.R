.rs.restartR() # Start new session

reticulate::use_python("~/miniforge3/envs/rstudio-tf-2.8/bin/python")
reticulate::use_condaenv("rstudio-tf-2.8", conda="~/miniforge3/bin/conda", required = TRUE)

library(tensorflow)
library(keras)
tf_config() 
tf_version() 
tf$config$list_logical_devices()

#Loading the environmental time series dataset
elec <- read.csv("elec_cap.csv", header = TRUE, sep = ",", na.strings = c(""))

#Displaying summary of the dataset

head(elec)
str(elec)
summary(elec)

elec$DATE <- as.Date(elec$DATE)
str(elec)
summary(elec)
#Find missing values in the dataset

library("Amelia")
missmap(elec, main = 'Missing Map', col = c('yellow','black'), legend = FALSE)

# loading packages and helper functions
library(keras)
library(tensorflow)
library(longmemo)
library(ggplot2)

suppressMessages(devtools::source_gist(filename = "helper_funcs.R",
                                       id = "https://gist.github.com/Letmode/8c842b722ab31210ad6df64ef786a8c2"))


set_random_seed(10000) # seed function from the tensorflow package

Y0 = elec$CAPUTLG2211S
n0 = length(Y0)
n_te = 204
n_tr = n0 - n_te
ntrain = floor(0.7 * n0) # size of training set

Year = seq(2010, 2020 + 9 / 12, length.out = n0)

### Modelling ARMA-LSTM

p = 1
q = 1 

nndata = dataprep_nn_arma(Y0, ntrain, p.max = p , q.max = q , fixed = TRUE)
nndata$m_arma

pq = p + q # 2 ar coefficients + 1 ma coefficient
X_tr = nndata$Xtrain # training data input features
X_te = nndata$Xtest # test data input features
Y_tr = nndata$Ytrain # training data target
Y_te = nndata$Ytest # test data target
Y0_te = nndata$Y0test # unscaled test data
# tensorflow accepts tensors which are arrays with certain dimensions
# in our case we have an array with 1 dimension
dim(X_tr) = c(dim(X_tr), 1)

model <- keras_model_sequential() %>%
  layer_lstm(units = 10,
             batch_input_shape = c(1, pq, 1),
             stateful = TRUE) %>%
  layer_dense(units = 1, activation = "relu")


#model <- keras_model_sequential() %>%
#  layer_lstm(units = 8,
#            batch_input_shape = c(1, pq, 1),
#            stateful = TRUE,
#            return_sequences = TRUE) %>%
# layer_lstm(units = 4,
#            stateful = TRUE) %>%
# layer_dense(units = 1, activation = "relu")

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
pred_out = pred_out * nndata$Ydiff + nndata$Ymin

model %>% evaluate(X_tr, Y_tr, batch_size = 1)
model %>% evaluate(X_pred, Y_te, batch_size = 1) 

ASE_lstm = ASE_calc(Y0_te, pred_out)
ASE_arma = ASE_calc(Y0_te, nndata$Ytest_arma)
AAE_lstm = AAE_calc(Y0_te, pred_out)
AAE_arma = AAE_calc(Y0_te, nndata$Ytest_arma)
CCP_lstm = cor(Y0_te, pred_out)
CCP_arma = cor(Y0_te, nndata$Ytest_arma)
eval_crits = data.frame("ASE" = c(ASE_lstm, ASE_arma), "AAE" = c(AAE_lstm, AAE_arma), 
                        "CCP" = c(CCP_lstm, CCP_arma))
rownames(eval_crits) = c("LSTM", "ARMA")
eval_crits

Year.og = Year
Year.te = Year[(ntrain + 1):n0]
df_og <- as.data.frame(cbind(Year.og, Y0))
df_fcs <- as.data.frame(cbind(Year.te, pred_out, Y0_te, nndata$Ytest_arma))
colnames(df_fcs) <- c("Year", "pred.te", "real_data", "pred.arma")

plot_data = ggplot(df_og) + 
  geom_line(aes(x = Year.og, y = Y0), color = "black") +
  labs(title = "Capacity Utilization for Electric Power Generation, Transmission and Distribution", y = "", x = "Year") +
  theme(legend.text = element_text(size = 12), plot.title = element_text(size = 14),
        axis.title = element_text(size = 12))

plot_fcs = ggplot(df_fcs) +
  geom_line(aes(x = Year, y = real_data, color = "og")) +
  geom_line(aes(x = Year, y = pred.te, color = "lstm")) +
  geom_line(aes(x = Year, y = pred.arma, color = "arma")) +
  labs(title = "LSTM-predictions", y = "") +
  scale_color_manual("", values = c("og" = "black", "lstm" = "red", "arma" = "blue"),
                     labels = c("og" = "Test data", "lstm" = "LSTM-preds", "arma" = "ARMA-preds")) +
  theme(legend.text = element_text(size = 12), plot.title = element_text(size = 14),
        axis.title = element_text(size = 12), legend.position = c(0.75, 0.15))

ggpubr::ggarrange(plot_data, plot_fcs, ncol = 1)

