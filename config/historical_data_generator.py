import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

"""
This script generates example historical data for the validation model.
In a real-world scenario, you would replace this with actual historical data.
"""


def generate_historical_data(start_year=1000, end_year=2020, interval=10, add_noise=True, save_path=None,

    """
    Historical data generation function.

    Physics Domain: multi_system
    Scale Level: civilization
    Application Domains: knowledge, validation
    """

    """
    Historical data generation function.

    Physics Domain:
    """
    Historical data generation function.
    
    Physics Domain: multi_system
    Scale Level: civilization
    Application Domains: knowledge, validation
    """
 multi_system
    Scale Level: civilization
    Application Domains: knowledge, validation
    """
                             max_noise=2.0, max_index_value=100.0, min_index_value=0.0):
    """
    Generate synthetic historical data with realistic patterns.

    Parameters:
        start_year (int): Starting year for data generation
        end_year (int): Ending year for data generation
        interval (int): Year interval between data points
        add_noise (bool): Whether to add random noise to data
        save_path (str): Path to save CSV file, or None to return only
        max_noise (float): Maximum amplitude of random noise
        max_index_value (float): Maximum value for any index
        min_index_value (float): Minimum value for any index

    Returns:
        pd.DataFrame: Generated historical data
    """
    # Apply parameter bounds
    start_year = max(0, min(end_year - 1, start_year))
    end_year = max(start_year + 1, end_year)
    interval = max(1, interval)
    max_noise = max(0, max_noise)

    # Generate year range
    years = np.arange(start_year, end_year + 1, interval)
    num_years = len(years)

    # Time scale for calculations (0 to 1)
    time_scale = (years - start_year) / max(1, (end_year - start_year))

    # Generate historical knowledge index with key periods
    # 1. Knowledge Index - Growth of human knowledge and technology

    # Base exponential growth with periods of acceleration
    # Use bounded exponential to prevent overflow
    exp_input = np.minimum(3 * time_scale, 10)  # Cap at 10 to prevent exp overflow
    knowledge = 10 * (np.exp(exp_input) - 1) / (np.exp(3) - 1)

    # Add key historical periods

    # Dark Ages decline (if start year is before 1400)
    if start_year < 1400:
        dark_ages_mask = (years >= start_year) & (years < 1400)
        if np.any(dark_ages_mask):
            # Slow growth during Dark Ages
            dark_ages_scale = (years[dark_ages_mask] - start_year) / max(1, 1400 - start_year)
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
        # Bound exponential input to prevent overflow
        exp_input = np.minimum(-3 * info_age_scale, 0)  # Cap at 0 (since it's negative)
        info_age_effect = 15 * (1 - np.exp(exp_input))
        knowledge[info_age_mask] += info_age_effect

    # Add noise if requested
    if add_noise:
        noise = np.random.normal(0, min(0.5, max_noise / 2), num_years)
        knowledge += noise

    # Ensure knowledge is positive and normalized to 0-100 scale
    knowledge = np.maximum(min_index_value, knowledge)
    knowledge = min_index_value + (max_index_value - min_index_value) * knowledge / np.max(knowledge)

    # 2. Suppression Index - Resistance to new knowledge and freedom

    # Base suppression starts high and generally decreases
    # Bound exponential input to prevent overflow
    exp_input = np.minimum(-2 * time_scale, 0)  # Cap at 0 (since it's negative)
    suppression = 80 * np.exp(exp_input) + 20

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
        noise = np.random.normal(0, min(1.5, max_noise), num_years)
        suppression += noise

    # Ensure suppression is in 0-100 range
    suppression = np.clip(suppression, min_index_value, max_index_value)

    # 3. Intelligence Index - Effective knowledge application

    # Intelligence is related to knowledge but reduced by suppression
    intelligence = knowledge * (1 - 0.5 * suppression / max_index_value)

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
        modern_science_scale = (years[modern_science_mask] - 1850) / max(1, end_year - 1850)
        # Bound exponential input to prevent overflow
        exp_input = np.minimum(-2 * modern_science_scale, 0)  # Cap at 0 (since it's negative)
        modern_science_effect = 10 * (1 - np.exp(exp_input))
        intelligence[modern_science_mask] += modern_science_effect

    # Add noise if requested
    if add_noise:
        noise = np.random.normal(0, min(1.0, max_noise / 1.5), num_years)
        intelligence += noise

    # Normalize to 0-100 scale
    intelligence = np.clip(intelligence, min_index_value, None)
    if np.max(intelligence) > 0:  # Prevent division by zero
        intelligence = min_index_value + (max_index_value - min_index_value) * intelligence / np.max(intelligence)

    # 4. Truth Index - Factual accuracy and consensus reality

    # Truth starts low and gradually increases over time
    # Bound exponential input to prevent overflow
    exp_input = np.minimum(-3 * time_scale, 0)  # Cap at 0 (since it's negative)
    truth = 20 + 80 * (1 - np.exp(exp_input))

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
        modern_empiric_scale = (years[modern_empiric_mask] - 1850) / max(1, end_year - 1850)
        # Bound exponential input to prevent overflow
        exp_input = np.minimum(-2 * modern_empiric_scale, 0)  # Cap at 0 (since it's negative)
        modern_empiric_effect = 20 * (1 - np.exp(exp_input))
        truth[modern_empiric_mask] += modern_empiric_effect

    # Information age brings both truth and misinformation (post-1990)
    info_chaos_mask = (years >= 1990)
    if np.any(info_chaos_mask):
        info_chaos_scale = (years[info_chaos_mask] - 1990) / max(1, end_year - 1990)
        # Initial boost followed by slight decline due to misinformation
        info_chaos_effect = 5 * np.sin(np.pi * info_chaos_scale)
        truth[info_chaos_mask] += info_chaos_effect

    # Add noise if requested
    if add_noise:
        noise = np.random.normal(0, min(1.0, max_noise / 1.5), num_years)
        truth += noise

    # Normalize to 0-100 scale
    truth = np.clip(truth, min_index_value, max_index_value)

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


def visualize_historical_data(data, figsize=(15, 10), save_path=None,

    """
    Historical data generation function.

    Physics Domain: multi_system
    Scale Level: civilization
    Application Domains: knowledge, validation
    """

    """
    Historical data generation function.

    Physics Domain:
    """
    Historical data generation function.
    
    Physics Domain: multi_system
    Scale Level: civilization
    Application Domains: knowledge, validation
    """
 multi_system
    Scale Level: civilization
    Application Domains: knowledge, validation
    """
                              max_yval=100, min_yval=0, dpi=300):
    """
    Visualize the generated historical data.

    Parameters:
        data (pd.DataFrame): Historical data
        figsize (tuple): Figure size
        save_path (str): Path to save figure, or None to display only
        max_yval (float): Maximum y-axis value
        min_yval (float): Minimum y-axis value
        dpi (int): DPI for saved figure

    Returns:
        matplotlib.figure.Figure: Figure object
    """
    # Validate input data
    if data is None or len(data) == 0 or 'year' not in data.columns:
        raise ValueError("Invalid data format: Must include 'year' column")

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    # Metrics to plot
    metrics = ["knowledge_index", "suppression_index", "intelligence_index", "truth_index"]

    # Plot metrics
    for i, metric in enumerate(metrics):
        if metric not in data.columns:
            continue

        ax = axes[i]
        ax.plot(data["year"], data[metric], linewidth=2)

        # Add title and labels
        metric_name = metric.replace("_", " ").title()
        ax.set_title(metric_name)
        ax.set_xlabel("Year")
        ax.set_ylabel("Index Value")
        ax.grid(True)

        # Set consistent y-axis limits
        ax.set_ylim(min_yval, max_yval)

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

        min_year = min(data["year"])
        max_year = max(data["year"])

        for name, start, end, color in highlight_periods:
            # Only add periods that overlap with the data range
            if end >= min_year and start <= max_year:
                # Adjust period to data range
                visible_start = max(start, min_year)
                visible_end = min(end, max_year)

                # Add highlight
                ax.axvspan(visible_start, visible_end, alpha=0.2, color=color)

                # Add period label if enough space
                if (visible_end - visible_start) > (max_year - min_year) * 0.05:
                    label_x = (visible_start + visible_end) / 2
                    ax.text(label_x, 5, name, ha='center', va='bottom',
                            fontsize=8, rotation=90, color='black')

    # Add overall title with data range
    title = f"Historical Data ({min(data['year'])}-{max(data['year'])})"
    fig.suptitle(title, fontsize=16)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    # Save figure if path provided
    if save_path is not None:
        try:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        except Exception as e:
            print(f"Error saving figure: {e}")

    return fig


if __name__ == "__main__":
    # Example usage
    start_year = 1000
    end_year = 2020
    interval = 10

    try:
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
    except Exception as e:
        print(f"An error occurred: {e}")