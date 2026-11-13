import ee
ee.Initialize()

points_fc = ee.FeatureCollection("projects/ee-thomashallopeau26/assets/ssl4eo-s12")

img = ee.Image("JRC/GHSL/P2023A/GHS_BUILT_S/2020").select("built_surface")

# Buffer each point (square ≈ 27x27 pixels)
buffered = points_fc.map(
    lambda f: f.buffer(1350).bounds()
)

# Reduce raster over each buffered geometry
reduced = img.reduceRegions(
    collection=buffered,
    reducer=ee.Reducer.sum(),
    scale=100
)

task = ee.batch.Export.table.toDrive(
    collection=reduced,
    description="ghsl_built_surface_sum_27px",
    fileFormat="CSV",
    selectors=["id", "sum"]
)

task.start()



import pandas as pd
import geopandas as gpd

POINTS_CSV = "./ssl4eo-s12_center_coords.csv"                  
GEE_CSV = "./ghsl_built_surface_sum_27px.csv"      
OUTPUT_GPKG = "ssl4eo_points_ghsl.gpkg"


df_points = pd.read_csv(POINTS_CSV, header=None)
df_points.columns = ["id", "lon", "lat"]


gdf = gpd.GeoDataFrame(
    df_points,
    geometry=gpd.points_from_xy(df_points.lon, df_points.lat),
    crs="EPSG:4326"
)


df_gee = pd.read_csv(GEE_CSV)
df_gee = df_gee[["id", "sum"]]


gdf_out = gdf.merge(df_gee, on="id", how="left")


gdf_out.to_file(OUTPUT_GPKG, driver="GPKG")



from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

INPUT_GPKG = Path("ssl4eo_points_ghsl.gpkg")
OUTPUT_GPKG = Path("ssl4eo_points_ghsl_bins.gpkg")
OUTPUT_DIR = Path("id_lists_bins")

COLUMN_NAME = "sum"
ID_COLUMN = "id"
PERCENTILE_COLUMN = "bin"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def format_ssl4eo_id(sample_id: int) -> str:
    """
    Format a numeric sample ID into the SSL4EO-S12 string convention.

    Parameters
    ----------
    sample_id : int
        Numeric sample identifier.

    Returns
    -------
    str
        Formatted sample identifier.
    """
    return f"ssl4eos12_train_seasonal_data_{sample_id:07d}"


def save_id_list(ids: pd.Series, output_txt: Path) -> None:
    """
    Save a series of numeric IDs to a text file using SSL4EO-S12 format.

    Parameters
    ----------
    ids : pandas.Series
        Series of numeric IDs.
    output_txt : pathlib.Path
        Output text file path.
    """
    formatted_ids = [
        format_ssl4eo_id(int(sample_id))
        for sample_id in ids.dropna().astype(int).tolist()
    ]

    with output_txt.open("w", encoding="utf-8") as file:
        for sample_id in formatted_ids:
            file.write(sample_id + "\n")


def build_equal_size_bins(
    df: pd.DataFrame,
    value_column: str,
    id_column: str,
    n_bins: int = 100,
) -> pd.Series:
    """
    Assign equal-size bins after sorting by a value column.

    Rows are sorted by value_column, then by id_column to ensure
    deterministic output when multiple rows share the same value.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    value_column : str
        Column used for sorting.
    id_column : str
        Secondary sort column for deterministic tie-breaking.
    n_bins : int, optional
        Number of bins to create.

    Returns
    -------
    pandas.Series
        Series indexed like df, containing labels such as p00_01.
    """
    df_sorted = df[[value_column, id_column]].copy()
    df_sorted = df_sorted.sort_values(
        by=[value_column, id_column],
        ascending=[True, True],
        kind="mergesort",
    )

    n_rows = len(df_sorted)

    bin_indices = np.floor(np.arange(n_rows) * n_bins / n_rows).astype(int)
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    labels = [
        f"p{bin_index:02d}_{bin_index + 1:02d}"
        for bin_index in bin_indices
    ]

    percentile_series = pd.Series(
        labels,
        index=df_sorted.index,
        name=PERCENTILE_COLUMN,
        dtype="object",
    )

    return percentile_series.reindex(df.index)


# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------

gdf = gpd.read_file(INPUT_GPKG).copy().to_crs("EPSG:4326")

gdf = gdf.dropna(subset=[ID_COLUMN, COLUMN_NAME])

gdf[ID_COLUMN] = gdf[ID_COLUMN].astype(int)


# ---------------------------------------------------------------------
# Build 100 equal-size groups
# ---------------------------------------------------------------------

gdf[PERCENTILE_COLUMN] = build_equal_size_bins(
    df=gdf,
    value_column=COLUMN_NAME,
    id_column=ID_COLUMN,
    n_bins=100,
)


# ---------------------------------------------------------------------
# Save updated GeoPackage
# ---------------------------------------------------------------------

gdf.to_file(OUTPUT_GPKG, driver="GPKG")


# ---------------------------------------------------------------------
# Export one text file per group
# ---------------------------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

summary_rows = []

for bin_index in range(100):
    label = f"p{bin_index:02d}_{bin_index + 1:02d}"
    gdf_bin = gdf.loc[gdf[PERCENTILE_COLUMN] == label].copy()

    output_file = OUTPUT_DIR / f"allowed_ids_{label}.txt"
    save_id_list(gdf_bin[ID_COLUMN], output_file)

    min_sum = gdf_bin[COLUMN_NAME].min() if not gdf_bin.empty else np.nan
    max_sum = gdf_bin[COLUMN_NAME].max() if not gdf_bin.empty else np.nan

    summary_rows.append(
        {
            "percentile": label,
            "count": len(gdf_bin),
            "min_sum": min_sum,
            "max_sum": max_sum,
            "file": output_file.name,
        }
    )

    print(
        f"{output_file.name} -> {len(gdf_bin)} IDs "
        f"(sum range: [{min_sum}, {max_sum}])"
    )


# ---------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------

df_summary = pd.DataFrame(summary_rows)

print()
print(df_summary.to_string(index=False))
print()
print(f"Updated GeoPackage saved to: {OUTPUT_GPKG}")



import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib import font_manager

font_path = "/usr/share/fonts/truetype/cmu/cmunss.ttf" 
font_manager.fontManager.addfont(font_path)
font_prop = font_manager.FontProperties(fname=font_path)
plt.rcParams["font.family"] = font_prop.get_name()

# ---------------------------------------------------------------------
# Plot median value per bin
# ---------------------------------------------------------------------

df_median = (
    gdf.groupby(PERCENTILE_COLUMN, observed=False)[COLUMN_NAME]
    .median()
    .reset_index()
    .rename(columns={COLUMN_NAME: "median_sum"})
)

df_median["bin_index"] = (
    df_median[PERCENTILE_COLUMN]
    .str.extract(r"p(\d+)_")[0]
    .astype(int)
)

df_median = df_median.sort_values("bin_index")

# ---------------------------------------------------------------------
# Create plot
# ---------------------------------------------------------------------

plt.figure(figsize=(7, 4))

plt.plot(
    df_median[PERCENTILE_COLUMN],
    df_median["median_sum"] / 7_290_000,
    color="#5500D4",
    marker="o",
    markersize=3,
    linewidth=1,
)

# ---------------------------------------------------------------------
# Configure X-axis ticks
# ---------------------------------------------------------------------

tick_positions = [
    0,
    *range(9, len(df_median), 10),
]

tick_labels = (
    df_median["bin_index"].iloc[tick_positions] + 1
)

plt.xticks(
    ticks=tick_positions,
    labels=tick_labels,
)

# ---------------------------------------------------------------------
# Configure Y-axis limits and formatting
# ---------------------------------------------------------------------

plt.ylim(-0.015, 0.365)
plt.yticks(np.arange(0, 0.351, 0.05))
plt.gca().yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

# ---------------------------------------------------------------------
# Labels and styling
# ---------------------------------------------------------------------

plt.xlabel("Urban Coverage Percentile Bins", fontsize=20)
plt.ylabel("Median Built-Up Ratio", fontsize=20)

plt.tick_params(axis="both", labelsize=18)

plt.grid(True, alpha=0.3)
plt.tight_layout()

# ---------------------------------------------------------------------
# Save and display
# ---------------------------------------------------------------------

plt.savefig(
    "median_urban_coverage_per_bin.png",
    dpi=600,
    bbox_inches="tight",
)

plt.show()
