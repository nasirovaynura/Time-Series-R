library(tidyverse)
library(tidymodels)
library(inspectdf)
library(data.table)
library(modeltime)
library(skimr)
library(timetk)
library(highcharter)
library(h2o)
library(forecast)
library(prophet)
library(xgboost)


df <- read_csv("AirPassengers.csv"); view(df)

glimpse(df)
inspect_na(df)

names(df) <- names(df) %>% gsub('#', '', .); names(df)

df$Month <- as.Date(paste(df$Month, "-01", sep=""), format="%Y-%m-%d")


df %>% 
  plot_time_series(
    Month, Passengers, 
    .interactive = T,
    .plotly_slider = T,
    .smooth = F)


# Seasonality plot

df %>%
  plot_seasonal_diagnostics(
    Month, Passengers, .interactive = T)


# 1. Use arima_boost(), exp_smoothing(), prophet_reg() models ----

splits <- initial_time_split(df, prop = 0.8)


model_fit_arima_boosted <- arima_boost(
  min_n = 2,
  learn_rate = 0.015) %>% 
  set_engine(engine = "auto_arima_xgboost") %>%
  fit(Passengers ~ Month, data = training(splits))


model_fit_ets <- exp_smoothing() %>%
  set_engine(engine = "ets") %>%
  fit(Passengers ~ Month, data = training(splits))


model_fit_prophet <- prophet_reg(seasonality_weekly = T) %>%
  set_engine(engine = "prophet") %>%
  fit(Passengers ~ Month, data = training(splits))


# 2. Compare RMSE scores on test set ----

arima_boost_forecast <- predict(model_fit_arima_boosted, new_data = testing(splits))
ets_forecast <- predict(model_fit_ets, new_data = testing(splits))
prophet_forecast <- predict(model_fit_prophet, new_data = testing(splits))


# RMSE scores

arima_boost_rmse <- sqrt(mean((testing(splits)$Passengers - arima_boost_forecast$.pred)^2))
ets_rmse <- sqrt(mean((testing(splits)$Passengers - ets_forecast$.pred)^2))
prophet_rmse <- sqrt(mean((testing(splits)$Passengers - prophet_forecast$.pred)^2))

tibble(cat("ARIMA-Boost RMSE:", round(arima_boost_rmse, 2), "\n"), 
       cat("ETS RMSE:", round(ets_rmse, 2), "\n"),
       cat("Prophet RMSE:", round(prophet_rmse, 2), "\n"))

best_model <- which.min(c(arima_boost_rmse, ets_rmse, prophet_rmse))


# 3. Make forecast on lowest RMSE score model ----

if (best_model == 1) {
  best_forecast <- arima_boost_forecast$.pred
  best_model_name <- "ARIMA-Boost"
} else if (best_model == 2) {
  best_forecast <- ets_forecast$.pred
  best_model_name <- "ETS"
} else {
  best_forecast <- prophet_forecast$.pred
  best_model_name <- "Prophet"
}


# 4. Visualize past data and forecast values on one plot; make separation with two different colors.

forecast_data <- data.frame(
  Month = testing(splits)$Month,
  Passengers = testing(splits)$Passengers,
  Forecast = best_forecast)


ggplot(forecast_data, aes(x = Month)) +
  geom_line(aes(y = Passengers, color = "Actual"), size = 1) +
  geom_line(aes(y = Forecast, color = "Forecast"), size = 1) +
  labs(title = paste("Best Model:", best_model_name),
       x = "Month",
       y = "Passengers") +
  scale_color_manual(values = c("Actual" = "darkgreen", "Forecast" = "darkred")) +
  theme_minimal()

