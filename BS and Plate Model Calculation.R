library(tidyverse)

### Section zero: Global parameters
r_m <- 0.08  # Market return, for example, 8%
r_f <- 0.03  # Risk-free rate, for example, 3%
sigma_M <- 0.2  # Example value for the volatility of the RETURNS, this is a relative sigma. For absolute volatility the shorthand sd is used instead
S_0 <- 1 # this is a term from Black-Scholes; the price at t=0, which is normalized to 1 everywhere here

### Section one: functions to calculate Black-Scholes Options 

calculate_BS_Call <- function(beta, K, sigma_idiosyncratic, t){
  sigma <- sqrt(beta^2*sigma_M^2 + sigma_idiosyncratic^2)
  d1 <- (log(1/K) + (r_f + 0.5*sigma^2)*t)/(sigma*sqrt(t))
  d2 <- d1 - sigma*sqrt(t)
  price_call <- pnorm(d1) - K * exp(-r_f*t) * pnorm(d2)
  return(price_call)
}

calculate_BS_Put <- function(beta, K, sigma_idiosyncratic, t){
  sigma <- sqrt(beta^2*sigma_M^2 + sigma_idiosyncratic^2)
  d1 <- (log(1/K) + (r_f + 0.5*sigma^2)*t)/(sigma*sqrt(t))
  d2 <- d1 - sigma*sqrt(t)
  price_put <- K * exp(-r_f*t) * pnorm(-d2) - pnorm(-d1) 
  return(price_put)
}

### Section two: functions to calculate Plate Options 

calculate_Plate_Call <- function(beta, K, sigma_idiosyncratic, t){
  # general terms for the underlying
  sigma <- sqrt(beta^2*sigma_M^2 + sigma_idiosyncratic^2)
  mu <- r_f + (r_m - r_f) * beta
  #d terms
  dc1 <- (log(1/K) + (mu + sigma^2) * t) / (sigma * sqrt(t))
  dc2 <- (log(1/K) + mu * t) / (sigma * sqrt(t))
  dc3 <- (log(1/K) + (mu + 2 * sigma^2) * t) / (sigma * sqrt(t))
  # parts of the equation; S_0 is normalized to one so it drops out of most places
  F_C <- exp((mu + 0.5*sigma^2)*t)*pnorm(dc1) - K * pnorm(dc2)
  sigma_C <- sqrt(
    F_C^2 - 2 * (F_C + K) * exp((mu + 0.5*sigma^2)*t) * pnorm(dc1) + K * (K + 2*F_C) * pnorm(dc2) + exp((mu + sigma^2)*2*t) * pnorm(dc3)
  )
  #parts of the correlation formula
  first <- -(K + F_C + exp((mu + 0.5*sigma^2)*t)) * pnorm(dc1)
  second <- (K + F_C) * pnorm(dc2) + exp((mu + sigma^2)*t) * pnorm(dc3)
  third <- F_C * (pnorm(dc1) - pnorm(dc2))
  rho_CU <- 1/(sigma_C * sqrt(exp(t*sigma^2)-1)) * (first + second + third)
  
  p_M <- (exp((mu + 0.5*sigma^2)*t) - exp(r_f * t)) / sqrt(exp((2*mu + sigma^2)*t) * (exp(t*sigma^2)-1))
  
  price_call <- (F_C - rho_CU * sigma_C * p_M) / exp(r_f*t)
  return(price_call)
}

calculate_Plate_Put <- function(beta, K, sigma_idiosyncratic, t){
  # general terms for the underlying
  sigma <- sqrt(beta^2*sigma_M^2 + sigma_idiosyncratic^2)
  mu <- r_f + (r_m - r_f) * beta
  #d terms
  dp1 <- (log(1/K) - (mu + sigma^2) * t) / (sigma * sqrt(t))
  dp2 <- (log(1/K) - mu * t) / (sigma * sqrt(t))
  dp3 <- (log(1/K) - (mu + 2 * sigma^2) * t) / (sigma * sqrt(t))
  # parts of the equation; S_0 is normalized to one so it drops out of most places
  F_P <- K * pnorm(dp2) - exp((mu + 0.5*sigma^2)*t)*pnorm(dp1)
  sigma_P <- sqrt(
    exp((mu + sigma^2)*2*t) * pnorm(dp3) + 2 * (F_P - K) * exp((mu + 0.5*sigma^2)*t) * pnorm(dp1) + (F_P - K)^2 * pnorm(dp2) + F_P^2 * pnorm(-dp2)
  )
  #parts of the correlation formula
  first <- (F_P - K) * pnorm(dp2) * exp((mu + 0.5*sigma^2)*t)
  second <- (exp((mu + 0.5*sigma^2)*t) - F_P + K) * exp((mu + 0.5*sigma^2)*t) * pnorm(dp1)
  third <- - exp((mu + sigma^2)*t) * pnorm(dp3) + F_P * exp((mu + 0.5*sigma^2)*t) * (pnorm(dp1) - pnorm(dp2))
  rho_PU <- 1/(sigma_P * sqrt(exp(t*sigma^2)-1)) * (first + second + third)
  
  p_M <- (exp((mu + 0.5*sigma^2)*t) - exp(r_f * t)) / sqrt(exp((2*mu + sigma^2)*t) * (exp(t*sigma^2)-1))
  
  price_put <- (F_P - rho_PU * sigma_P * p_M) / exp(r_f*t)
  return(price_put)
}

### Generate table
generate_table <- function(t, sigma_idiosyncratic, option_function, beta_U_min, beta_U_max, beta_U_steps, K_min, K_max, K_steps){
  if (!option_function %in% c("calculate_BS_Call", "calculate_BS_Put", "calculate_Plate_Call", "calculate_Plate_Put")) {
    stop("Invalid option function name!")
  }
  # Create sequences for beta_U and K
  beta_U_seq <- seq(beta_U_min, beta_U_max, length.out = beta_U_steps)
  K_seq <- seq(K_min, K_max, length.out = K_steps)
  
  results_df <- data.frame(beta_U = numeric(0), K = numeric(0), OptionPrice = numeric(0))
  for (beta in beta_U_seq) {
    for (k in K_seq) {
      # Call the function using match.fun
      option_price <- match.fun(option_function)(beta, k, sigma_idiosyncratic, t)
      results_df <- rbind(results_df, data.frame(beta_U = beta, K = k, OptionPrice = option_price))
    }
  }
  return(results_df)
}

results_table <- generate_table(1, 0.2, "calculate_Plate_Call", -2, 2, 11, 0, 2, 9)
print(results_table)
reshaped_table <- results_table %>%
  spread(key = K, value = OptionPrice)
print(reshaped_table)







