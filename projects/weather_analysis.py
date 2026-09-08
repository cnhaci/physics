"""
Weather Data Analysis
=====================

Numerical analysis of 2024 NOAA weather data for:
- Honolulu International Airport, Hawaii
- Ted Stevens Anchorage International Airport, Alaska

The script reproduces the main analysis and visualizations developed
in the FIZ228 Numerical Analysis project.

Required packages:
    pandas
    numpy
    matplotlib
    seaborn

Expected input files:
    data/Hawaii_Honolulu_2024.csv
    data/Alaska_Anchorage_2024.csv

Usage:
    python weather_analysis.py

Outputs:
    PNG figures are saved to the figures/ directory.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DATA_DIR = Path("data")
OUTPUT_DIR = Path("figures")

HAWAII_FILE = DATA_DIR / "Hawaii_Honolulu_2024.csv"
ALASKA_FILE = DATA_DIR / "Alaska_Anchorage_2024.csv"

SELECTED_COLUMNS = [
    "DATE",
    "HourlyDryBulbTemperature",
    "HourlyRelativeHumidity",
    "HourlyWindSpeed",
    "HourlyPrecipitation",
    "HourlySeaLevelPressure",
]

NUMERIC_COLUMNS = [
    "HourlyDryBulbTemperature",
    "HourlyRelativeHumidity",
    "HourlyWindSpeed",
    "HourlyPrecipitation",
    "HourlySeaLevelPressure",
]

sns.set_style("whitegrid")
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 20)


# ---------------------------------------------------------------------
# Data loading and preparation
# ---------------------------------------------------------------------

def load_data():
    """Load the two NOAA station datasets."""
    if not HAWAII_FILE.exists():
        raise FileNotFoundError(f"Missing file: {HAWAII_FILE}")

    if not ALASKA_FILE.exists():
        raise FileNotFoundError(f"Missing file: {ALASKA_FILE}")

    hawaii = pd.read_csv(HAWAII_FILE, low_memory=False)
    alaska = pd.read_csv(ALASKA_FILE, low_memory=False)

    return hawaii, alaska


def prepare_data(hawaii, alaska):
    """Select variables, create time/station features, clean data."""
    hawaii = hawaii.copy()
    alaska = alaska.copy()

    # Convert DATE to datetime.
    hawaii["DATE"] = pd.to_datetime(hawaii["DATE"])
    alaska["DATE"] = pd.to_datetime(alaska["DATE"])

    # Create derived time variables.
    hawaii["Month"] = hawaii["DATE"].dt.month
    hawaii["Hour"] = hawaii["DATE"].dt.hour

    alaska["Month"] = alaska["DATE"].dt.month
    alaska["Hour"] = alaska["DATE"].dt.hour

    # Add station labels.
    hawaii["Station"] = "Honolulu"
    alaska["Station"] = "Anchorage"

    # Select the variables used in the analysis.
    selected_columns = SELECTED_COLUMNS + ["Month", "Hour", "Station"]

    hawaii_selected = hawaii[selected_columns].copy()
    alaska_selected = alaska[selected_columns].copy()

    weather = pd.concat(
        [hawaii_selected, alaska_selected],
        ignore_index=True,
    )

    # Convert selected measurements to numeric values.
    for column in NUMERIC_COLUMNS:
        weather[column] = pd.to_numeric(
            weather[column],
            errors="coerce",
        )

    # Remove duplicate rows.
    weather = weather.drop_duplicates().copy()

    # Sort chronologically for time-series interpolation.
    weather = weather.sort_values(
        ["Station", "DATE"]
    ).reset_index(drop=True)

    return weather


# ---------------------------------------------------------------------
# Statistical analysis
# ---------------------------------------------------------------------

def calculate_monthly_temperature(weather):
    """Calculate mean monthly temperature by station."""
    return (
        weather.groupby(["Month", "Station"])["HourlyDryBulbTemperature"]
        .mean()
        .reset_index()
    )


def calculate_correlation(weather):
    """Calculate the correlation matrix for numerical variables."""
    return weather[NUMERIC_COLUMNS].corr()


def print_summary(weather):
    """Print a concise dataset summary."""
    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)

    print(f"Rows: {len(weather):,}")
    print(f"Columns: {len(weather.columns)}")
    print(f"Stations: {', '.join(weather['Station'].unique())}")
    print(f"Time range: {weather['DATE'].min()} → {weather['DATE'].max()}")

    print("\nMissing values:")
    print(weather.isnull().sum())

    print("\nDuplicate rows after cleaning:")
    print(weather.duplicated().sum())

    print("\nDescriptive statistics:")
    print(weather[NUMERIC_COLUMNS].describe())


# ---------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------

def save_figure(filename):
    """Save the current Matplotlib figure."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def plot_monthly_temperature(weather):
    """Figure 1: Monthly average temperature."""
    monthly_temp = calculate_monthly_temperature(weather)

    plt.figure(figsize=(12, 6))

    sns.lineplot(
        data=monthly_temp,
        x="Month",
        y="HourlyDryBulbTemperature",
        hue="Station",
        marker="o",
    )

    plt.title("Monthly Average Temperature")
    plt.xlabel("Month")
    plt.ylabel("Temperature")

    save_figure("monthly-average-temperature.png")


def plot_temperature_comparison(weather):
    """Figure 2: Temperature comparison between stations."""
    plt.figure(figsize=(10, 6))

    sns.boxplot(
        data=weather,
        x="Station",
        y="HourlyDryBulbTemperature",
    )

    plt.title("Temperature Comparison Between Stations")
    plt.xlabel("Station")
    plt.ylabel("Temperature")

    save_figure("temperature-comparison.png")


def plot_temperature_distribution(weather):
    """Figure 3: Temperature distribution."""
    plt.figure(figsize=(10, 6))

    sns.histplot(
        data=weather,
        x="HourlyDryBulbTemperature",
        hue="Station",
        kde=True,
        bins=30,
    )

    plt.title("Temperature Distribution")
    plt.xlabel("Temperature")
    plt.ylabel("Frequency")

    save_figure("temperature-distribution.png")


def plot_humidity_temperature(weather):
    """Figure 4: Relative humidity versus temperature."""
    plt.figure(figsize=(10, 6))

    sns.scatterplot(
        data=weather,
        x="HourlyRelativeHumidity",
        y="HourlyDryBulbTemperature",
        hue="Station",
        alpha=0.5,
    )

    plt.title("Humidity vs Temperature")
    plt.xlabel("Relative Humidity")
    plt.ylabel("Temperature")

    save_figure("humidity-vs-temperature.png")


def plot_correlation_heatmap(weather):
    """Figure 5: Correlation heatmap."""
    correlation = calculate_correlation(weather)

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
    )

    plt.title("Correlation Heatmap")

    save_figure("correlation-heatmap.png")


# ---------------------------------------------------------------------
# Interpolation analysis
# ---------------------------------------------------------------------

def interpolate_missing_values(weather):
    """Apply linear interpolation to the selected numerical variables."""
    interpolated = weather.copy()

    interpolated[NUMERIC_COLUMNS] = (
        interpolated.groupby("Station")[NUMERIC_COLUMNS]
        .transform(lambda group: group.interpolate(method="linear"))
    )

    return interpolated


def plot_original_vs_interpolated(original, interpolated):
    """Figure 6: Compare original and interpolated temperature data."""
    original_temperature = original["HourlyDryBulbTemperature"].copy()

    plt.figure(figsize=(12, 6))

    plt.plot(
        original_temperature.iloc[:500],
        label="Original",
        alpha=0.7,
    )

    plt.plot(
        interpolated["HourlyDryBulbTemperature"].iloc[:500],
        label="Interpolated",
        linestyle="--",
    )

    plt.title("Original vs Interpolated Temperature Data")
    plt.xlabel("Index")
    plt.ylabel("Temperature")
    plt.legend()

    save_figure("original-vs-interpolated-temperature.png")


# ---------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------

def main():
    """Run the complete weather-data analysis."""
    print("=" * 70)
    print("FIZ228 NUMERICAL ANALYSIS PROJECT")
    print("NOAA WEATHER DATA ANALYSIS")
    print("=" * 70)

    # Load raw datasets.
    hawaii, alaska = load_data()

    print("\nDatasets loaded successfully.")
    print(f"Honolulu:  {hawaii.shape}")
    print(f"Anchorage: {alaska.shape}")

    # Prepare and clean the combined dataset.
    weather = prepare_data(hawaii, alaska)

    print_summary(weather)

    # Generate Figures 1–5.
    plot_monthly_temperature(weather)
    plot_temperature_comparison(weather)
    plot_temperature_distribution(weather)
    plot_humidity_temperature(weather)
    plot_correlation_heatmap(weather)

    # Interpolation analysis.
    interpolated_weather = interpolate_missing_values(weather)

    print("\nRemaining missing values after interpolation:")
    print(interpolated_weather[NUMERIC_COLUMNS].isnull().sum())

    # Generate Figure 6.
    plot_original_vs_interpolated(
        weather,
        interpolated_weather,
    )

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print(f"Figures saved to: {OUTPUT_DIR.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
