import sympy as sp
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import time

start_time = time.time()

##### Section 1: Define symbols and parameters #######
# Define symbols
L, W, I, σ_H, σ_L, ρ_IL, µ_L, µ_I, x, a, β_L, σ_M, σ_e, r_f, r_m, β_I, s, W_1, µ_W, σ_W, y, I_real, L_real = sp.symbols(
    'L W I σ_H σ_L ρ_IL µ_L µ_I x a β_L σ_M σ_e r_f r_m, β_I s W_1 µ_W σ_W y I_real L_real')

# Independent Parameter dictionary 
var_values = {
     'L': 100, #  H Wealth (H+I=W) - can be any positive value 
     'I': 100, #  I Wealth (H+I=W) - can be any positive value 
     'r_f': 20/100, # Risk free rate - any positive value (think of as over 10 year period)
     'r_m': 0.6, # market rate - any positive value > r_f (think of as over 10 year period)
     'β_L': 0.5, # Beta of the fixed H Investment - any value between -1 and 1
     'σ_M': 1.0, # Market volatility - >=0
     'σ_e': 1.5, # H's independent volatility - between 0 and 5 
     'a': 1/1000 # risk aversion parameter
}

####### Section 2: Define underlying model #######
#Functions still based on return 
µ_I = 1 + r_f + (r_m - r_f) * β_I
µ_L = 1 + r_f + (r_m - r_f) * β_L
σ_I = β_I * σ_M 
σ_L = sp.sqrt(β_L**2 * σ_M**2 + σ_e**2) 
ρ_IL = β_L*σ_M/sp.sqrt(β_L**2*σ_M**2 + σ_e**2)
W = L + I
#scaling with W to have them be based on W
μ_W = ((µ_I) * (I / W) + µ_L * (L / W))*W
σ_W = (sp.sqrt((L/W)**2 * σ_L**2 + (I/W)**2 * σ_I**2 + 2 * (L/W) * (I/W) * σ_I * σ_L * ρ_IL))*W
W_1 = (1 / (σ_W * sp.sqrt(2 * sp.pi))) * sp.exp(-0.5*(((x-µ_W)/σ_W)**2))

#CARA Utility function
U = (1 - sp.exp(-a * x))
#Quadratic Utility function (switch between CARA a quadratic utility)
#U = x - (1/3200) * x**2
#analytic solution given the CARA * N(μ,σ)
EU_W = 1 - sp.exp(-a*µ_W + 0.5 * a**2 * σ_W **2)

##### Section 3: Definiting maximization, integration #######
def optimize_ana_EU(EU, β_I):
    diff_EU = sp.diff(EU, β_I)
    solutions = sp.solve(diff_EU, β_I)
    EU_func = sp.lambdify(β_I, EU, "sympy")
    EU_at_solutions = [round(EU_func(sol),5) for sol in solutions]
    solutions_rounded = [round(sol,5) for sol in solutions]
    return list(zip(solutions_rounded, EU_at_solutions))

def optimize_num_EU(β_I_min, β_I_max, EU, β_I):
    # Define the function to optimize as the negative of EU
    EU_func = sp.lambdify(β_I, EU, "sympy")
    def neg_EU(β_I_value):
        result = -EU_func(β_I_value[0])  # Index into the array here
        if result == sp.nan:
            raise ValueError(f"EU_func returned NaN for β_I_value={β_I_value}")
        return result
    # Use scipy's minimize function to find the minimum of the negative of EU
    result = minimize(neg_EU, np.array([(β_I_max + β_I_min) / 2]), bounds=[(β_I_min, β_I_max)])

    if result.success:
        # If the optimization was successful, return the maximum of EU and its value
        β_I_sol = float(result.x[0])  # Convert NumPy array to float
        EU_max = -result.fun
        return round(β_I_sol,5), round(EU_max,5)
    else:
        # If the optimization was not successful, raise an exception
        raise ValueError("Optimization was not successful: " + result.message)

def symbolic_trapezoidal_integration(W, var, lower_integration_bound, upper_integration_bound, steps):
    # Compute step size
    h = (upper_integration_bound - lower_integration_bound) / steps

    # Compute the x-values at which to evaluate the function
    x_values = np.linspace(lower_integration_bound, upper_integration_bound, steps + 1)

    # Compute the y-values for these x-values as symbolic expressions
    y_values = [W.subs(var, val) for val in x_values]

    # Print the table of x and y values
    #print("Table of x and y-values:")
    #print("-"*40)
    #for i, (x, y) in enumerate(zip(var_values, y_values)):
    #    print(f"x_{i}: {x}, y_{i}: {y}")
    #print("-"*40)
    
    # Implement the trapezoidal rule symbolically
    integral = h * (0.5*y_values[0] + 0.5*y_values[-1] + sum(y_values[1:-1]))
    return integral

##### Section 4: Results without Insurance and with short as insurance #######
### Results derived analytically  
print("### Solutions for the no-insurance case and the Short-Option cases derived analytically ###")
# No Insurance
solution_norm = optimize_ana_EU(EU_W.subs(var_values), β_I)
print("[No Insurance - analytically] β_I_opt and EU at β_I_opt are: ", solution_norm)

# Short - only idiosyncratic H risk
var_values_short_idio = var_values.copy()  # create a copy of the original dictionary
var_values_short_idio['σ_e'] = 0  
EU_short_idio_val = EU_W.subs(var_values_short_idio)
solution_short_idio =  optimize_ana_EU(EU_short_idio_val, β_I)
print("[Short, idiosyncratic H Risk - analytically] β_I_opt and EU at β_I_opt are: ", solution_short_idio)

#Short - total H risk
var_values_short_total = var_values.copy()  # create a copy of the original dictionary
var_values_short_total['σ_e'] = 0  
var_values_short_total['β_L'] = 0  
EU_short_total_val = EU_W.subs(var_values_short_total)
solution_short_total =  optimize_ana_EU(EU_short_total_val, β_I)
print("[Short, total H Risk - analytically] β_I_opt and EU at β_I_opt are: ", solution_short_total)

### Results derived numerically 
print()
print("### Solutions for the no-insurance case and the Short-Option cases derived numerically ###")
# No Insurance
U_W = (U * W_1).subs(var_values)
EU_num = symbolic_trapezoidal_integration(U_W, x, -2000, 5000, 200)
solution_norm_num = optimize_num_EU(-10, 20, EU_num, β_I)
print("[No Insurance - numerically] β_I_opt and EU at β_I_opt are: ", solution_norm_num)

# Short - only idiosyncratic H risk
U_W_short_idio = (U * W_1).subs(var_values_short_idio)
EU_short_idio_num = symbolic_trapezoidal_integration(U_W_short_idio, x, -2000, 5000, 200)
solution_short_idio_num = optimize_num_EU(-10, 20, EU_short_idio_num, β_I)
print("[Short, idiosyncratic H Risk - numerically] β_I_opt and EU at β_I_opt are: ", solution_short_idio_num)

# Short - total H risk
U_W_short_total = (U * W_1).subs(var_values_short_total)
EU_short_total_num = symbolic_trapezoidal_integration(U_W_short_total, x, -2000, 5000, 200)
solution_short_total_num = optimize_num_EU(-10, 20, EU_short_total_num, β_I)
print("[Short, total H Risk - numerically] β_I_opt and EU at β_I_opt are: ", solution_short_total_num)

##### Section 5: setting up mthe model and functions f the Put-Option results #######
µ_I_val = (µ_I * I).subs(var_values)
µ_L_val = (µ_L * L).subs(var_values)
σ_I_val = (σ_I * I).subs(var_values)
σ_L_val = (σ_L * L).subs(var_values)
ρ_IL_val = ρ_IL.subs(var_values)
k_val = µ_L_val # for this case only

def generate_sample(µ_1, µ_2, σ_1, σ_2, correlation, n):
    µ_1 = float(µ_1)
    µ_2 = float(µ_2)
    σ_1 = float(σ_1)
    σ_2 = float(σ_2)
    correlation = float(correlation)
    n = int(n)
    mean = [µ_1, µ_2]
    covariance = [[σ_1**2, correlation*σ_1*σ_2], [correlation*σ_1*σ_2, σ_2**2]]
    samples = np.random.multivariate_normal(mean, covariance, n)
    samples_1 = samples[:, 0]
    samples_2 = samples[:, 1]
    return samples_1, samples_2

def feed_sample_into_function(expr, utility, sample_I, sample_L):
    expr = expr.subs(var_values)
    expr_lambda = sp.lambdify((I_real, L_real), expr, "sympy")
    Total_list = []
    U_2 = utility.subs(var_values)
    for i, h in zip(sample_I, sample_L):
        # Substitute the samples into the lambda function
        expr_filled = expr_lambda(i, h)
        # Then substitute this new expression into U
        Total = U_2.subs(x, expr_filled)
        Total_list.append(Total)
    num_elements = len(Total_list)
    EU = sum(Total_list) / num_elements
    return(EU)

# Wealth function for total H risk put 
def W_put_total():
    eta_L = (1+r_f) / ((r_m-r_f)*β_L + 2 * norm.pdf(0) * σ_L)
    put_total_payout = sp.Piecewise((-(L_real - µ_L_val), L_real <= µ_L_val),(0, L_real > µ_L_val))
    put_total_cost = (1 + r_f) * (norm.pdf(0) * σ_L_val) / (1 + r_f + (r_m - r_f) * (-eta_L * β_L))
    put_total_cost = put_total_cost.subs(var_values)
    W_put_total = I_real + L_real + put_total_payout - put_total_cost
    return W_put_total

# Wealth function for idiosyncratic H risk put 
def W_put_idio():
    eta_L = (1+r_f) / ((r_m-r_f)*β_L + 2 * norm.pdf(0) * σ_L)
    eta_M = (1+r_f) / ((r_m-r_f) + 2 * norm.pdf(0) * σ_M)
    I_r = (I_real - µ_I_val) / β_I
    L_r = (µ_L_val - L_real)
    put_idio_payout = sp.Piecewise((L_r, L_r >= 0),(0, L_r < 0)) + sp.Piecewise((I_r, I_r >= 0),(0, I_r < 0))
    put_idio_cost = (1 + r_f) * (norm.pdf(0) * σ_L_val) / (1 + r_f + (r_m - r_f) * (-eta_L * β_L))  + (1 + r_f) * (norm.pdf(0) * (σ_M*L)) / (1 + r_f + (r_m - r_f) * (eta_M))
    W_put_idio = (I_real + L_real + put_idio_payout - put_idio_cost)
    W_put_idio = W_put_idio.subs(var_values)
    return W_put_idio

### Monte Carlo optimization function
def maximize_monte_carlo(expr, utility, start, end, step_size, rolls):
    
    num_steps = int(np.floor((end - start) / step_size))
    results = np.zeros((num_steps, 2))  # Initialize an array to store the results
    current_step = start
    i = 0
    
    while i < num_steps:
        # Swap in the current value for β_I in µ_I_val and σ_I_val
        current_µ_I_val = µ_I_val.subs(β_I, current_step)
        current_σ_I_val = σ_I_val.subs(β_I, current_step)
        current_expr = expr.subs(β_I, current_step)
        #print(current_expr)
        # Generate symbolic samples
        sample_I, sample_L = generate_sample(current_µ_I_val, µ_L_val, current_σ_I_val, σ_L_val, ρ_IL.subs(var_values), rolls)
        # Feed samples into function
        outcome_value = feed_sample_into_function(current_expr, utility, sample_I, sample_L)

        # Store current step and its corresponding outcome value
        results[i] = [current_step, outcome_value]

        # Increase step for the next iteration and increment i
        current_step += step_size
        i += 1
    # Fit a quadratic function to the results
    coefficients = np.polyfit(results[:, 0], results[:, 1], 2)
    
    # Calculate the maximum of the quadratic function
    a, b, c = coefficients
    max_x = -b / (2 * a)
    max_y = a * max_x**2 + b * max_x + c

    # Return the maximum x and the corresponding y
    return round(max_x,5), round(max_y,5)

###### Section 6: applying the Monte Carlo optimization for the Put cases #######
print()
print("### Solutions for the Put-Option cases derived numerically (Monte Carlo) ###")
#Solution for total H risk Put 
solution_put_total = maximize_monte_carlo(W_put_total(), U, 2, 6, 0.2, 10000)
print("[Put, total H Risk - monte carlo] β_I_opt and EU at β_I_opt are: ", solution_put_total)

#Solution for idiosyncratic H risk Put 
solution_put_idio = maximize_monte_carlo(W_put_idio(), U, 2, 6, 0.2, 10000)
print("[Put, idiosyncratic H Risk - monte carlo] β_I_opt and EU at β_I_opt are: ", solution_put_idio)

### Test for normal 
def W_norm():
    W_norm = L_real + I_real
    return W_norm

#print()
print("### Refernce Solution of the no-insurance case derived numerically, also using Monte Carlo - the difference between this and the first is (roughly) the error of the Monte Carlo model at this sample size ###")
solution_reference = maximize_monte_carlo(W_norm(), U, 2, 6, 0.2, 10000)
print("[normal reference value - monte carlo] β_I_opt and EU at β_I_opt are: ", solution_reference)

end_time = time.time()
print("This took ", end_time - start_time," seconds to run")



