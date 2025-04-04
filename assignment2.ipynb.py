'''task 1A'''
import random

def coin_toss_simulation(n=10000):
    heads = 0
    tails = 0

    for _ in range(n):
        toss = random.choice(['Heads', 'Tails'])
        if toss == 'Heads':
            heads += 1
        else:
            tails += 1

    prob_heads = heads / n
    prob_tails = tails / n

    print(f"Total Tosses: {n}")
    print(f"Heads: {heads} ({prob_heads:.4f})")
    print(f"Tails: {tails} ({prob_tails:.4f})")

coin_toss_simulation()

'''task 1B'''
import random

def simulate_dice_sum(n=10000):
    count_sum_7 = 0

    for _ in range(n):
        die1 = random.randint(1, 6)
        die2 = random.randint(1, 6)
        if die1 + die2 == 7:
            count_sum_7 += 1

    probability = count_sum_7 / n
    print(f"Total Rolls: {n}")
    print(f"Number of times sum is 7: {count_sum_7}")
    print(f"Estimated Probability of sum 7: {probability:.4f}")

simulate_dice_sum()

'''task 2'''
import random

def probability_at_least_one_six(trials=10000):
    success = 0

    for _ in range(trials):
        rolls = [random.randint(1, 6) for _ in range(10)]
        if 6 in rolls:
            success += 1

    probability = success / trials
    print(f"Total Trials: {trials}")
    print(f"Trials with at least one 6: {success}")
    print(f"Estimated Probability: {probability:.4f}")

probability_at_least_one_six()

'''task 3'''
#P = (Number of times Red follows Blue) / (Number of times Blue was drawn)
import random

def simulate_conditional_probability(trials=1000):
    colors = ['Red'] * 5 + ['Green'] * 7 + ['Blue'] * 8
    draws = [random.choice(colors) for _ in range(trials)]

    blue_count = 0
    red_after_blue_count = 0

    for i in range(1, trials):
        if draws[i - 1] == 'Blue':
            blue_count += 1
            if draws[i] == 'Red':
                red_after_blue_count += 1

    if blue_count == 0:
        probability = 0
    else:
        probability = red_after_blue_count / blue_count

    print(f"Total Blue followed draws: {blue_count}")
    print(f"Red followed after Blue: {red_after_blue_count}")
    print(f"Estimated P(Red | Previous was Blue): {probability:.4f}")

simulate_conditional_probability()
#P(Red | Blue_before) = [P(Blue_before | Red) * P(Red)] / P(Blue_before)

'''task 4'''
import random
import statistics

def simulate_discrete_variable(n=1000):
    values = [1, 2, 3]
    probabilities = [0.25, 0.35, 0.40]

    samples = random.choices(values, weights=probabilities, k=n)

    mean = statistics.mean(samples)
    variance = statistics.variance(samples)
    std_dev = statistics.stdev(samples)

    print(f"Generated {n} samples")
    print(f"Mean: {mean:.4f}")
    print(f"Variance: {variance:.4f}")
    print(f"Standard Deviation: {std_dev:.4f}")

simulate_discrete_variable()

'''task 5'''
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def simulate_exponential_distribution(n=2000, mean=5):
    rate = 1 / mean
    samples = np.random.exponential(scale=mean, size=n)

    # Plot histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(samples, bins=30, stat='density', color='skyblue', edgecolor='black', label='Sample Histogram')

    # Overlay PDF
    x = np.linspace(0, max(samples), 1000)
    pdf = rate * np.exp(-rate * x)
    plt.plot(x, pdf, 'r-', lw=2, label='Theoretical PDF')

    plt.title('Exponential Distribution (mean = 5)')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.show()

simulate_exponential_distribution()

'''task 6'''
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def clt_simulation():
    # Step 1: Generate 10,000 values from Uniform(0, 10)
    population = np.random.uniform(0, 10, 10000)
    
    # Step 2 & 3: Take 1000 samples of size 30 and compute means
    sample_means = []
    for _ in range(1000):
        sample = np.random.choice(population, size=30)
        sample_means.append(np.mean(sample))
    
    # Step 4: Plot distribution of sample means
    plt.figure(figsize=(10, 6))
    sns.histplot(sample_means, bins=30, kde=True, color='mediumseagreen', edgecolor='black')
    plt.title('Distribution of Sample Means (n=30) from Uniform(0, 10)')
    plt.xlabel('Sample Mean')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

clt_simulation()
