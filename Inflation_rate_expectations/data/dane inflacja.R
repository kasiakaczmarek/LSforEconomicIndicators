#install.packages('forecast', dependencies = TRUE)
library(forecast)
library(readxl)


#read data
#my_data <- read_excel("Interactive_VAR_data_2022_with_tone.xlsx")
my_data <- read_excel("dane_2022.xlsx")

country_list <-unique(my_data$country)
for(i in 1:length(country_list)){
#  i=2
  arma11_prediction<-c()
  arma11_spread<-c()
  autoarima_prediction<-c()
  autoarima_spread<-c()
  s_arma11_prediction<-c()
  s_arma11_spread<-c()
  s_autoarima_prediction<-c()
  s_autoarima_spread<-c()
 
  time_series<-my_data[my_data$country==country_list[i],]
  
  for(t in 12:nrow(time_series)){
 # t =194
     print(t)
  #last year
  series1<-time_series$inf_rate[(t-12):t]
  print(series1)
  arma11 <- tryCatch({arima(series1, order = c(1,0,1))},error = function(e) {auto.arima(series1, max.p = 2, max.q = 2)})
  arma11_p<-tryCatch({predict(arma11)},error = function(e) {forecast(arma11,h=1)[4]})
  arma11_prediction<-append(arma11_prediction, arma11_p$pred)
  arma11_spread<-append(arma11_spread, arma11_p$se)
  #arima<-auto.arima(series1, max.p = 2, max.q = 2)
  #autoarima_p<-predict(arima)
  #autoarima_prediction<-append(autoarima_prediction, autoarima_p$pred)
  #autoarima_spread<-append(autoarima_prediction, autoarima_p$se)
  
  #all last data
  series2<-time_series$inf_rate[1:t]
  print(series2)
  
  s_arma11 <- tryCatch({arima(series2, order = c(1,0,1))},error = function(e) {auto.arima(series1, max.p = 2, max.q = 2)})
  s_arma11_p<-tryCatch({predict(s_arma11)},error = function(e) {forecast(s_arma11,h=1)[4]})
  s_arma11_prediction<-tryCatch({append(s_arma11_prediction, s_arma11_p$pred)},error = function(e){append(s_arma11_prediction,s_arma11_p$mean)})
  s_arma11_spread<-tryCatch({append(s_arma11_spread, s_arma11_p$se)},error = function(e){append(s_arma11_spread,(forecast(s_arma11,h=1)$mean - forecast(s_arma11,h=1)$lower[1]))})
  #s_arima<-auto.arima(series2, max.p = 2, max.q = 2)
  #s_autoarima_p<-predict(arima)
  #s_autoarima_prediction<-append(s_autoarima_prediction, s_autoarima_p$pred)
  #s_autoarima_spread<-append(s_autoarima_prediction, s_autoarima_p$se)
  
  } 
  #results<-cbind(arma11_prediction,arma11_spread,autoarima_prediction,autoarima_spread,s_arma11_prediction,s_arma11_spread,s_autoarima_prediction,s_autoarima_spread)
  results<-cbind(arma11_prediction,arma11_spread,s_arma11_prediction,s_arma11_spread)
  
    #save to country file
  write.table(results, paste(country_list[i],".csv"), dec=".")
}