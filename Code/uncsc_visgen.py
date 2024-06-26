import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import logging
from scipy.stats import f_oneway
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Number of nodes to simulate
num_nodes = 1000

# Define weights for each factor
weights = {
    'reliability': 0.30,
    'security': 0.25,
    'participation': 0.20,
    'endorsements': 0.15,
    'protocol_adherence': 0.10
}

logging.info("Generating synthetic performance data for nodes...")
# Generate synthetic performance data for nodes
performance_data = {
    'reliability': np.random.rand(num_nodes) * 100,
    'security': np.random.rand(num_nodes) * 100,
    'participation': np.random.rand(num_nodes) * 100,
    'endorsements': np.random.rand(num_nodes) * 100,
    'protocol_adherence': np.random.rand(num_nodes) * 100
}

# Convert to DataFrame for easier manipulation
df = pd.DataFrame(performance_data)

# Function to normalize scores
def normalize(scores):
    return (scores - scores.min()) / (scores.max() - scores.min()) * 100

logging.info("Normalizing performance data...")
# Normalize performance data
for factor in performance_data:
    df[f'norm_{factor}'] = normalize(df[factor])
    logging.debug(f'Normalized {factor} scores: {df[f"norm_{factor}"].head()}')

logging.info("Calculating confidence scores...")
# Calculate confidence scores
df['confidence_score'] = sum(weights[factor] * df[f'norm_{factor}'] for factor in weights)
logging.debug(f'Confidence scores: {df["confidence_score"].head()}')

logging.info("Simulating raw test scores...")
# Simulate raw test scores
df['raw_test_score'] = np.random.rand(num_nodes) * 100
logging.debug(f'Raw test scores: {df["raw_test_score"].head()}')

logging.info("Scaling raw test scores...")
# Scale raw test scores (assuming scale factor of 1 for simplicity)
df['scaled_test_score'] = df['raw_test_score']
logging.debug(f'Scaled test scores: {df["scaled_test_score"].head()}')

logging.info("Calculating adjusted evaluation scores...")
# Adjusted evaluation scores
for factor in performance_data:
    df[f'adjusted_{factor}_score'] = weights[factor] * df[f'norm_{factor}'] + weights[factor] * df['scaled_test_score']
    logging.debug(f'Adjusted {factor} scores: {df[f"adjusted_{factor}_score"].head()}')

# Calculate final adjusted evaluation score
df['final_adjusted_evaluation_score'] = df[[f'adjusted_{factor}_score' for factor in performance_data]].sum(axis=1)
logging.debug(f'Final adjusted evaluation scores: {df["final_adjusted_evaluation_score"].head()}')

logging.info("Calculating final performance scores...")
# Calculate final performance score
df['final_performance_score'] = (df['final_adjusted_evaluation_score'] + df['scaled_test_score']) / 2
logging.debug(f'Final performance scores: {df["final_performance_score"].head()}')

# Log initial performance score distribution
logging.info(f"Initial performance score distribution: \n{df['final_performance_score'].describe()}")

# Define performance bands
performance_bands = {
    'high': df[df['final_performance_score'] > df['final_performance_score'].quantile(0.75)],
    'medium': df[(df['final_performance_score'] > df['final_performance_score'].quantile(0.25)) & (df['final_performance_score'] <= df['final_performance_score'].quantile(0.75))],
    'low': df[df['final_performance_score'] <= df['final_performance_score'].quantile(0.25)]
}

logging.info("Assigning nodes to zones using hybrid strategy...")
# Assign nodes to zones using performance band randomization
num_zones = 10
for band in performance_bands:
    for i in range(num_zones):
        sampled_indices = performance_bands[band].sample(frac=1/num_zones).index
        df.loc[sampled_indices, 'zone'] = i

# Log the nodes in each zone after the hybrid assignment
for i in range(num_zones):
    logging.info(f"Nodes in zone {i}: {df[df['zone'] == i].index.tolist()}")

logging.info("Ranking nodes within zones...")
# Rank nodes within zones
df['zone_rank'] = df.groupby('zone')['final_performance_score'].rank(method="dense", ascending=False)
logging.debug(f'Zone ranks: {df["zone_rank"].head()}')

logging.info("Normalizing and adjusting scores within zones...")
# Normalize and adjust within zones
for factor in performance_data:
    df[f'zone_norm_{factor}'] = df.groupby('zone')[f'norm_{factor}'].transform(normalize)
    df[f'zone_adjusted_{factor}_score'] = weights[factor] * df[f'zone_norm_{factor}'] + weights[factor] * df['scaled_test_score']
    logging.debug(f'Zone adjusted {factor} scores: {df[f"zone_adjusted_{factor}_score"].head()}')

df['zone_final_adjusted_evaluation_score'] = df[[f'zone_adjusted_{factor}_score' for factor in performance_data]].sum(axis=1)
df['zone_final_performance_score'] = (df['zone_final_adjusted_evaluation_score'] + df['scaled_test_score']) / 2
logging.debug(f'Zone final performance scores: {df["zone_final_performance_score"].head()}')

logging.info("Calculating global ranks and scaling...")
# Global ranking and scaling
df['global_rank'] = df['zone_final_performance_score'].rank(method="dense", ascending=False)
logging.debug(f'Global ranks: {df["global_rank"].head()}')

logging.info("Calculating scaled scores and aggregate scores...")
# Calculate scaled scores
for factor in performance_data:
    df[f'scaled_{factor}_score'] = df['final_performance_score'] * weights[factor]
    logging.debug(f'Scaled {factor} scores: {df[f"scaled_{factor}_score"].head()}')

# Aggregate score
df['aggregate_score'] = df[[f'scaled_{factor}_score' for factor in performance_data]].sum(axis=1)
logging.debug(f'Aggregate scores: {df["aggregate_score"].head()}')

logging.info("Calculating percentile ranks and global comparative scores...")
# Percentile rank
df['percentile_rank'] = df['aggregate_score'].rank(pct=True) * 100
logging.debug(f'Percentile ranks: {df["percentile_rank"].head()}')

# Global comparative score
df['global_comparative_score'] = df['percentile_rank']
logging.debug(f'Global comparative scores: {df["global_comparative_score"].head()}')

logging.info("Predicting potential top performer nodes...")
potential_top_performers = df.nlargest(5, 'global_comparative_score')
logging.info(f"Potential top performer nodes: {potential_top_performers.index.tolist()}")
logging.info(f"Details of potential top performer nodes:\n{potential_top_performers[['confidence_score', 'raw_test_score', 'scaled_test_score', 'final_adjusted_evaluation_score', 'final_performance_score', 'zone', 'zone_rank', 'global_rank', 'aggregate_score', 'percentile_rank', 'global_comparative_score']]}")

logging.info("Selecting the top performer node...")
top_performer = potential_top_performers.iloc[0]

# Handle ties for top performer selection
if len(potential_top_performers) > 1:
    top_performer = potential_top_performers.loc[
        potential_top_performers['final_performance_score'].idxmax()]

# If still tied, consider dual top performers
dual_top_performers = potential_top_performers[potential_top_performers['final_performance_score'] == top_performer['final_performance_score']]
if len(dual_top_performers) > 1:
    logging.info("Dual top performers detected.")
    top_performer = dual_top_performers.iloc[0]  # Select the first one for simplicity

logging.info(f"Top performer node: {top_performer.name} with confidence score: {top_performer['confidence_score']}")
logging.info(f"Top performer node details: \n{top_performer}")

# Distribution of final performance scores across zones
zone_performance_summary = df.groupby('zone')['final_performance_score'].describe()
logging.info(f"Distribution of final performance scores across zones:\n{zone_performance_summary}")

# Additional statistics on final performance scores by zone
for i in range(num_zones):
    logging.info(f"Zone {i} - Mean final performance score: {zone_performance_summary.loc[i, 'mean']}, Standard Deviation: {zone_performance_summary.loc[i, 'std']}")

# Statistical analysis to test the significance of differences between zones
zone_scores = [df[df['zone'] == i]['final_performance_score'].values for i in range(num_zones)]
f_statistic, p_value = f_oneway(*zone_scores)
logging.info(f"ANOVA test results: F-statistic = {f_statistic}, p-value = {p_value}")
if p_value < 0.05:
    logging.info("The differences in final performance scores across zones are statistically significant.")
else:
    logging.info("The differences in final performance scores across zones are not statistically significant.")


# Data for visualization
data = {
    'zone': np.tile(np.arange(10), 100),
    'final_performance_score': np.random.normal(loc=75, scale=30, size=1000)
}

df = pd.DataFrame(data)
df['final_performance_score'] = df['final_performance_score'].clip(17, 141)  # Clipping to simulate realistic scores

# Add global_rank based on final performance score
df['global_rank'] = df['final_performance_score'].rank(ascending=False, method='dense')

# Add global_comparative_score as a percentile rank
df['global_comparative_score'] = df['final_performance_score'].rank(pct=True) * 100

# Mean and standard deviation per zone
zone_stats = df.groupby('zone')['final_performance_score'].agg(['mean', 'std']).reset_index()

# Extracting the top 10 nodes by final performance score
top_10_nodes = df.nlargest(10, 'final_performance_score')

# Bar plot of top 10 nodes by final performance score
plt.figure(figsize=(12, 6))
sns.barplot(x=top_10_nodes.index, y='final_performance_score', data=top_10_nodes, palette='viridis')
plt.title('Top 10 Nodes by Final Performance Score')
plt.xlabel('Node Index')
plt.ylabel('Final Performance Score')
plt.xticks(rotation=45)
plt.grid(True)
plt.savefig('top_10_nodes.png')
plt.show()

# Scatter plot of global rankings
plt.figure(figsize=(12, 6))
sns.scatterplot(x='global_rank', y='final_performance_score', hue='zone', palette='tab10', data=df)
plt.title('Global Rankings of Nodes by Final Performance Score')
plt.xlabel('Global Rank')
plt.ylabel('Final Performance Score')
plt.grid(True)
plt.savefig('global_rankings.png')
plt.show()

# Creating a pivot table for heatmap
ranking_pivot = df.pivot_table(values='final_performance_score', index='zone', columns='global_rank', aggfunc=np.mean)

# Box plot of node rankings across zones
plt.figure(figsize=(12, 6))
sns.boxplot(x='zone', y='global_rank', data=df, palette='Set3')
plt.title('Box Plot of Node Rankings Across Zones')
plt.xlabel('Zone')
plt.ylabel('Global Rank')
plt.grid(True)
plt.savefig('boxplot_node_rankings.png')
plt.show()

# KDE plot of global comparative scores
plt.figure(figsize=(12, 6))
sns.kdeplot(df['global_comparative_score'], shade=True, color='green')
plt.title('KDE Plot of Global Comparative Scores')
plt.xlabel('Global Comparative Score')
plt.ylabel('Density')
plt.grid(True)
plt.savefig('kde_global_comparative_scores.png')
plt.show()

# Bar chart of mean final performance scores with standard deviation
plt.figure(figsize=(12, 6))
plt.bar(zone_stats['zone'], zone_stats['mean'], yerr=zone_stats['std'], capsize=5, color='skyblue')
plt.title('Mean Final Performance Scores with Standard Deviation Across Zones')
plt.xlabel('Zone')
plt.ylabel('Mean Final Performance Score')
plt.grid(True)
plt.savefig('bar_mean_final_scores.png')
plt.show()

# Histogram of final performance scores
plt.figure(figsize=(12, 6))
plt.hist(df['final_performance_score'], bins=30, color='lightgreen', edgecolor='black')
plt.title('Histogram of Final Performance Scores')
plt.xlabel('Final Performance Score')
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig('histogram_final_scores.png')
plt.show()

# Scatter plot of final performance scores by zone
plt.figure(figsize=(12, 6))
for i in range(10):
    plt.scatter(df[df['zone'] == i]['zone'], df[df['zone'] == i]['final_performance_score'], label=f'Zone {i}')
plt.title('Scatter Plot of Final Performance Scores by Zone')
plt.xlabel('Zone')
plt.ylabel('Final Performance Score')
plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1))
plt.grid(True)
plt.savefig('scatter_scores_by_zone.png')
plt.show()

# Violin plot of final performance scores across zones
plt.figure(figsize=(12, 6))
sns.violinplot(x='zone', y='final_performance_score', data=df, palette='muted')
plt.title('Violin Plot of Final Performance Scores Across Zones')
plt.xlabel('Zone')
plt.ylabel('Final Performance Score')
plt.grid(True)
plt.savefig('violin_scores_across_zones.png')
plt.show()

# Simulate some additional columns for a comprehensive pairplot
df['raw_test_score'] = np.random.normal(loc=75, scale=20, size=1000).clip(10, 100)
df['confidence_score'] = np.random.uniform(low=50, high=100, size=1000)

# Pairplot of the dataset
sns.pairplot(df[['final_performance_score', 'raw_test_score', 'confidence_score', 'zone']], hue='zone', palette='tab10')
plt.suptitle('Pairplot of Final Performance Score, Raw Test Score, Confidence Score by Zone', y=1.02)
plt.savefig('pairplot_dataset.png')
plt.show()

# KDE plot of final performance scores
plt.figure(figsize=(12, 6))
sns.kdeplot(df['final_performance_score'], shade=True, color='blue')
plt.title('KDE Plot of Final Performance Scores')
plt.xlabel('Final Performance Score')
plt.ylabel('Density')
plt.grid(True)
plt.savefig('kde_final_scores.png')
plt.show()