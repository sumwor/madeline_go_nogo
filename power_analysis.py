import statsmodels.api as sm
from statsmodels.stats.power import FTestAnovaPower, FTestPower
import numpy as np
from scipy.stats import wilcoxon
from pyPlotHW import StartPlots
# Define parameters for the power analysis

"""one-way anova"""
effect_size = 0.3  # Effect size (Cohen's f)
alpha = 0.05      # Significance level (Type I error rate)
n_groups_factor1 = 3  # Number of groups in factor 1
n_groups_factor2 = 2  # Number of groups in factor 2
n_obs_per_group =15  # Number of observations per group

# Create an object for power analysis
power_analysis = FTestAnovaPower()
powerlist = []
# Calculate the power for the given effect size, alpha, and sample size
for n in n_obs_per_group:
    power = power_analysis.solve_power(effect_size, nobs=n, alpha=alpha)
    powerlist.append(power)

powerfig1 = StartPlots()
powerfig1.ax.plot(n_obs_per_group, powerlist)
powerfig1.ax.set_xlabel('Sample size')
powerfig1.ax.set_ylabel('Power')
print(f"Statistical power: {power:.4f}")

"""mann-whitney test"""
# Parameters
n_pairs = 30
effect_size = 0.1
n_simulations = 1000
powerlist = []

# Simulate paired data
for n in n_pairs:
    power = 0
    for _ in range(n_simulations):
        before = np.random.normal(loc=0, scale=1, size=n)
        after = np.random.normal(loc=effect_size, scale=1, size=n)

        # Perform the Wilcoxon Signed-Rank Test
        _, p_value = wilcoxon(before, after)

        # Check if the test rejects the null hypothesis (p-value < 0.05, for example)
        if p_value < 0.05:
            power += 1

    estimated_power = power / n_simulations
    powerlist.append(estimated_power)

powerfig1 = StartPlots()
powerfig1.ax.plot(n_pairs, powerlist)
powerfig1.ax.set_xlabel('Sample size')
powerfig1.ax.set_ylabel('Power')

# f test for equal variance
from scipy.stats import f

# Define parameters
alpha = 0.05  # Significance level
effect_size = 2# Desired effect size (ratio of variances)
df1 = 1  # Degrees of freedom for the numerator (larger variance)
 # Degrees of freedom for the denominator (smaller variance)
 # Numerator degrees of freedom (groups - 1)
 # Denominator degrees of freedom (sample size - groups)
sample_size = np.arange(5, 60, 5)  # Total sample size
#
# # Create an FTestAnovaPower object
power_analysis = FTestPower()
powerlist = []
num_simulations = 1000
# # Calculate the power for the specified parameters
for n in sample_size:

    significant_tests = 0
    df2 = (n-1)*2
    # Perform power analysis through simulation
    for _ in range(num_simulations):
        group1 = np.random.normal(0, 1, int(n / 2))
        group2 = np.random.normal(0, np.sqrt(effect_size), int(n / 2))

        # Perform the F-test for comparing variances
        statistic = np.var(group2, ddof=1) / np.var(group1, ddof=1)
        p_value = 1 - f.cdf(statistic, df2, df1)

        if p_value < alpha:
            significant_tests += 1

    estimated_power = significant_tests / num_simulations
    powerlist.append(estimated_power)
"""correlation"""
import numpy as np
from scipy.stats import pearsonr

# Parameters
correlation_effect_size = 0.15  # Desired correlation effect size
sample_size = 30  # Sample size
alpha = 0.05  # Significance level
n_simulations = 1000  # Number of simulations
powerlist = []

# Simulate data and perform correlation tests
for n in sample_size:
    power= 0
    for _ in range(n_simulations):
        x = np.random.normal(0, 1, n)  # Generate random data
        y = x * correlation_effect_size + np.random.normal(0, 1, n)  # Introduce the desired correlation

        r, p_value = pearsonr(x, y)  # Perform correlation test

        if p_value < alpha:
            power += 1

    estimated_power = power / n_simulations
    powerlist.append(estimated_power)

powerfig1 = StartPlots()
powerfig1.ax.plot(sample_size, powerlist)
powerfig1.ax.set_xlabel('Sample size')
powerfig1.ax.set_ylabel('Power')

print(f"Estimated Power: {estimated_power:.3f}")
print(f"Required Sample Size: {sample_size:.0f}")