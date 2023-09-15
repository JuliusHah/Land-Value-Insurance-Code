library(MASS)
library(tidyverse)

### Section zero: Global parameters
r_m <- 0.05  # Market return, for example, 8%
r_f <- 0.03  # Risk-free rate, for example, 3%
sigma_M <- 0.2  # Example value for the volatility of the RETURNS, this is a relative sigma. For absolute volatility the shorthand sd is used instead
S_0 <- 1 # this is a term from Black-Scholes; the price at t=0, which is normalized to 1 everywhere here

### Section one: Function to simulate log-normals

simulate_lognormals <- function(beta_U, n, t, sigma_idiosyncratic) {
  # For the assets distribution
  r_i <- r_f + beta_U * (r_m - r_f)
  mu_i <- (r_i + 1)^t
  sd_i_abs <- sqrt(t) * sqrt(beta_U^2 * sigma_M^2 + sigma_idiosyncratic^2)
  sigma2_i <- log((sd_i_abs^2/mu_i^2) + 1)
  meanlog_corrected_i <- log(mu_i) - 0.5 * sigma2_i
 
  # For the market
  mu_m <- (r_m + 1)^t
  sd_m <- sigma_M
  sigma2_m <- log((sd_m^2/mu_m^2) + 1)
  meanlog_corrected_m <- log(mu_m) - 0.5 * sigma2_m
  
  # Calculate Correlation
  rho <- beta_U * (sd_m/sd_i_abs)
  # Calculate the covariance
  cov_val <- rho * sqrt(sigma2_i) * sqrt(sigma2_m)
  sigma_matrix <- matrix(c(sigma2_i, cov_val, cov_val, sigma2_m), 2, 2)
  
  #Simulate normals
  normals <- mvrnorm(n = n, mu = c(meanlog_corrected_i, meanlog_corrected_m), Sigma = sigma_matrix)
  # Transform to lognormals
  prices_i <- exp(normals[,1])
  prices_m <- exp(normals[,2])
  return(list(prices_i = prices_i, prices_m = prices_m))
}

correlation <- function(first, second){
  if (length(first) != length(second)) {
    stop("The two lists/vectors must be of the same length.")
  }
  return(cor(first, second))
}

### Section two: Functions to generated the (realized) payout of the samples
calculate_Call_payout <- function(K, realization) {
  payout <- max(0, (realization - K))
  return(payout)
}

calculate_Put_payout <- function(K,realization) {
  payout <- max(0, (K - realization))
  return(payout)
}

### Section three: simulate the correct results
simulate_actual_price <- function(beta_U, n, t, sigma_idiosyncratic, K, PorC) {
  simulation_results <- simulate_lognormals(beta_U, n, t, sigma_idiosyncratic)
  # Extract the first and second lists
  list_i <- simulation_results$prices_i
  list_m <- simulation_results$prices_m
  # Calculate options payout based on Put or Call
  if (PorC == "C") {
    list_i_payout <- sapply(list_i, function(realization) {
      calculate_Call_payout(K, realization)
    })
  } else if (PorC == "P") {
    list_i_payout <- sapply(list_i, function(realization) {
      calculate_Put_payout(K, realization)
    })
  } else {
    stop("Invalid value for PorC. Please use either 'C' for Call or 'P' for Put.")
  }
  # Calculate sd of the option payout and the correlation to the market to be able to get the options beta
  cor <- correlation(list_i_payout,list_m)
  sd_option <- sd(list_i_payout)
  beta_option <- cor*(sd_option/sigma_M)
  # The fiar price of the option is the average payout, discounted by the interest rate implied by its beta
  avg_payout <- mean(list_i_payout)
  discount_rate <- r_f + (r_m-r_f)*beta_option
  price <- avg_payout / (1 + discount_rate)
  return(price)
} 

generate_table <- function(n, t, sigma_idiosyncratic, PorC, beta_U_min, beta_U_max, beta_U_steps, K_min, K_max, K_steps){
  # Create sequences for beta_U and K
  beta_U_seq <- seq(beta_U_min, beta_U_max, length.out = beta_U_steps)
  K_seq <- seq(K_min, K_max, length.out = K_steps)
  results_df <- data.frame(beta_U = numeric(0), K = numeric(0), OptionPrice = numeric(0))
  for (beta in beta_U_seq) {
    for (k in K_seq) {
      option_price <- simulate_actual_price(beta, n, t, sigma_idiosyncratic, k, PorC)
      results_df <- rbind(results_df, data.frame(beta_U = beta, K = k, OptionPrice = option_price))
    }
  }
  return(results_df)
}

results_table <- generate_table(10000, 1, 0.2, "C", -2, 2, 11, 0, 2, 9)
reshaped_table <- results_table %>%
  spread(key = K, value = OptionPrice)
print(reshaped_table)
write.csv(reshaped_table, "Table_Monte_Carlo.csv", row.names=FALSE)

