import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

"""
This script generates example historical data for the validation model.
In a real-world scenario, you would replace this with actual historical data.
"""


def generate_historical_data(start_year=1000, end_year=2020, interval=10, add_noise=True, save_path=None):
    """
    Generate synthetic historical data with realistic patterns.

    Parameters:
        start_year (int): Starting year for data generation
        end_year (int): Ending year for data generation
        interval (int): Year interval between data points
        add_noise (bool): Whether to add random noise to data
        save_path (str): Path to save CSV file, or None to return only

    Returns:
        pd.DataFrame: Generated historical data
    """
    # Generate year range
    years = np.arange(start_year, end_year + 1, interval)
    num_years = len(years)

    # Time scale for calculations (0 to 1)
    time_scale = (years - start_year) / (end_year - start_year)

    # Generate historical knowledge index with key periods
    # 1. Knowledge Index - Growth of human knowledge and technology

    # Base exponential growth with periods of acceleration
    knowledge = 10 * (np.exp(3 * time_scale) - 1) / (np.exp(3) - 1)

    # Add key historical periods

    # Dark Ages decline (if start year is before 1400)
    if start_year < 1400:
        dark_ages_mask = (years >= start_year) & (years < 1400)
        if np.any(dark_ages_mask):
            # Slow growth during Dark Ages
            dark_ages_scale = (years[dark_ages_mask] - start_year) / (1400 - start_year)
            knowledge[dark_ages_mask] = 1 + 2 * dark_ages_scale

    # Renaissance effect (1400-1600)
    renaissance_mask = (years >= 1400) & (years <= 1600)
    if np.any(renaissance_mask):
        renaissance_scale = (years[renaissance_mask] - 1400) / 200
        renaissance_effect = 3 * np.sin(np.pi * renaissance_scale)
        knowledge[renaissance_mask] += renaissance_effect

    # Enlightenment effect (1650-1800)
    enlightenment_mask = (years >= 1650) & (years <= 1800)
    if np.any(enlightenment_mask):
        enlightenment_scale = (years[enlightenment_mask] - 1650) / 150
        enlightenment_effect = 5 * np.sin(np.pi * enlightenment_scale)
        knowledge[enlightenment_mask] += enlightenment_effect

    # Industrial Revolution effect (1760-1900)
    industrial_mask = (years >= 1760) & (years <= 1900)
    if np.any(industrial_mask):
        industrial_scale = (years[industrial_mask] - 1760) / 140
        industrial_effect = 7 * np.sin(np.pi * industrial_scale)
        knowledge[industrial_mask] += industrial_effect

    # Information Age effect (1950-present)
    info_age_mask = (years >= 1950)
    if np.any(info_age_mask):
        info_age_scale = (years[info_age_mask] - 1950) / 70
        info_age_effect = 15 * (1 - np.exp(-3 * info_age_scale))
        knowledge[info_age_mask] += info_age_effect

    # Add noise if requested
    if add_noise:
        knowledge += np.random.normal(0, 0.5, num_years)

    # Ensure knowledge is positive and normalized to 0-100 scale
    knowledge = np.maximum(0, knowledge)
    knowledge = 100 * knowledge / np.max(knowledge)

    # 2. Suppression Index - Resistance to new knowledge and freedom

    # Base suppression starts high and generally decreases
    suppression = 80 * np.exp(-2 * time_scale) + 20

    # Add specific historical periods of high suppression

    # Dark Ages effect (500-1300)
    dark_ages_mask = (years >= start_year) & (years <= 1300)
    if np.any(dark_ages_mask):
        dark_ages_scale = (years[dark_ages_mask] - start_year) / max(1, 1300 - start_year)
        suppression[dark_ages_mask] += 10 * (1 - dark_ages_scale)

    # Religious persecution effects (1450-1750)
    persecution_mask = (years >= 1450) & (years <= 1750)
    if np.any(persecution_mask):
        persecution_scale = (years[persecution_mask] - 1450) / 300
        persecution_effect = 15 * np.sin(np.pi * persecution_scale)
        suppression[persecution_mask] += persecution_effect

    # World Wars effects
    ww1_mask = (years >= 1914) & (years <= 1918)
    ww2_mask = (years >= 1939) & (years <= 1945)
    cold_war_mask = (years >= 1947) & (years <= 1991)

    if np.any(ww1_mask):
        suppression[ww1_mask] += 20
    if np.any(ww2_mask):
        suppression[ww2_mask] += 25
    if np.any(cold_war_mask):
        cold_war_scale = (years[cold_war_mask] - 1947) / 44
        cold_war_effect = 15 * (1 - cold_war_scale)
        suppression[cold_war_mask] += cold_war_effect

    # Add noise if requested
    if add_noise:
        suppression += np.random.normal(0, 1.5, num_years)

    # Ensure suppression is in 0-100 range
    suppression = np.clip(suppression, 0, 100)

    # 3. Intelligence Index - Effective knowledge application

    # Intelligence is related to knowledge but reduced by suppression
    intelligence = knowledge * (1 - 0.5 * suppression / 100)

    # Add specific intelligence boosts during certain periods

    # Scientific revolution (1550-1700)
    scientific_mask = (years >= 1550) & (years <= 1700)
    if np.any(scientific_mask):
        scientific_scale = (years[scientific_mask] - 1550) / 150
        scientific_effect = 5 * np.sin(np.pi * scientific_scale)
        intelligence[scientific_mask] += scientific_effect

    # Enlightenment thinking (1650-1800)
    if np.any(enlightenment_mask):
        intelligence[enlightenment_mask] += enlightenment_effect * 0.7

    # Modern scientific methods (1850-present)
    modern_science_mask = (years >= 1850)
    if np.any(modern_science_mask):
        modern_science_scale = (years[modern_science_mask] - 1850) / (end_year - 1850)
        modern_science_effect = 10 * (1 - np.exp(-2 * modern_science_scale))
        intelligence[modern_science_mask] += modern_science_effect

    # Add noise if requested
    if add_noise:
        intelligence += np.random.normal(0, 1.0, num_years)

    # Normalize to 0-100 scale
    intelligence = np.clip(intelligence, 0, None)
    intelligence = 100 * intelligence / np.max(intelligence)

    # 4. Truth Index - Factual accuracy and consensus reality

    # Truth starts low and gradually increases over time
    truth = 20 + 80 * (1 - np.exp(-3 * time_scale))

    # Add specific truth fluctuations

    # Pre-scientific era has low truth values
    pre_scientific_mask = (years < 1600)
    if np.any(pre_scientific_mask):
        truth[pre_scientific_mask] *= 0.7

    # Scientific method improves truth (1600-1850)
    early_science_mask = (years >= 1600) & (years <= 1850)
    if np.any(early_science_mask):
        early_science_scale = (years[early_science_mask] - 1600) / 250
        early_science_effect = 10 * early_science_scale
        truth[early_science_mask] += early_science_effect

    # Modern empiricism boosts truth (1850-present)
    modern_empiric_mask = (years >= 1850)
    if np.any(modern_empiric_mask):
        modern_empiric_scale = (years[modern_empiric_mask] - 1850) / (end_year - 1850)
        modern_empiric_effect = 20 * (1 - np.exp(-2 * modern_empiric_scale))
        truth[modern_empiric_mask] += modern_empiric_effect

    # Information age brings both truth and misinformation (post-1990)
    info_chaos_mask = (years >= 1990)
    if np.any(info_chaos_mask):
        info_chaos_scale = (years[info_chaos_mask] - 1990) / (end_year - 1990)
        # Initial boost followed by slight decline due to misinformation
        info_chaos_effect = 5 * np.sin(np.pi * info_chaos_scale)
        truth[info_chaos_mask] += info_chaos_effect

    # Add noise if requested
    if add_noise:
        truth += np.random.normal(0, 1.0, num_years)

    # Normalize to 0-100 scale
    truth = np.clip(truth, 0, 100)

    # Combine into dataframe
    data = pd.DataFrame({
        "year": years,
        "knowledge_index": knowledge,
        "suppression_index": suppression,
        "intelligence_index": intelligence,
        "truth_index": truth
    })

    # Save to CSV if path provided
    if save_path is not None:
        # Ensure directory exists
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(save_path, index=False)
        print(f"Historical data saved to {save_path}")

    return data


def visualize_historical_data(data, figsize=(15, 10), save_path=None):
    """
    Visualize the generated historical data.

    Parameters:
        data (pd.DataFrame): Historical data
        figsize (tuple): Figure size
        save_path (str): Path to save figure, or None to display only

    Returns:
        matplotlib.figure.Figure: Figure object
    """
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    # Metrics to plot
    metrics = ["knowledge_index", "suppression_index", "intelligence_index", "truth_index"]

    # Plot metrics
    for i, metric in enumerate(metrics):
        ax = axes[i]
        ax.plot(data["year"], data[metric], linewidth=2)

        # Add title and labels
        metric_name = metric.replace("_", " ").title()
        ax.set_title(metric_name)
        ax.set_xlabel("Year")
        ax.set_ylabel("Index Value")
        ax.grid(True)

        # Highlight key historical periods
        highlight_periods = [
            ("Dark Ages", 800, 1400, "gray"),
            ("Renaissance", 1400, 1600, "gold"),
            ("Enlightenment", 1650, 1800, "lightblue"),
            ("Industrial\nRevolution", 1760, 1900, "lightgreen"),
            ("World Wars", 1914, 1945, "salmon"),
            ("Information\nAge", 1950, 2020, "lightcoral")
        ]

        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]

        for name, start, end, color in highlight_periods:
            if start >= min(data["year"]) and end <= max(data["year"]):
                ax.axvspan(start, end, alpha=0.2, color=color)

                # Add period label
                label_x = (start + end) / 2
                ax.text(label_x, 5, name, ha='center', va='bottom',
                        fontsize=8, rotation=90, color='black')

    # Add overall title
    fig.suptitle("Historical Data (1000-2020)", fontsize=16)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    # Save figure if path provided
    if save_path is not None:
        plt.savefig(save_path)
        print(f"Figure saved to {save_path}")

    return fig


if __name__ == "__main__":
    # Example usage
    start_year = 1000
    end_year = 2020
    interval = 10

    # Generate historical data
    data = generate_historical_data(
        start_year=start_year,
        end_year=end_year,
        interval=interval,
        add_noise=True,
        save_path="outputs/data/historical_data.csv"
    )

    # Visualize data
    visualize_historical_data(
        data,
        save_path="outputs/plots/historical_data_visualization.png"
    )

    print("Done!")