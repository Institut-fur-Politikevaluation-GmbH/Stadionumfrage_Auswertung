#%% HEADER 
###############################################################################################################################################################

# NOTE: Project details 
'''
@Author(s): Ella Jurk 
E-mail: e.jurk@ipe-evaluation.de 
Project name:  Wertschöpfungsanalyse Mainz 05 
Project timeframe: Oct 2024 - Mai 2025 
Description / Purpose: Calculate visitor spending and learns about visitor characteristics.
Purpose of this code: This code processes the survey data gathered in the statium, cleans it, performs geospatial calculations, and runs regression analyses to estimate the indirect economic effect of visitor spendings.
'''

# NOTE: Disclaimer / Copyright 
''' 
The information in the document is non-binding and is for information purposes only. 
No action should be taken based on the provided information without specific professional advice. 
It may not be passed on and/or may not be made available to third parties without prior written consent from IPE Institute for Policy Evaluation GmbH.  
Liability claims against IPE Institute for Policy Evaluation GmbH caused by the use of the information contained in the publication are generally excluded.

© 2024 IPE Institut für Politikevaluation GmbH. All rights reserved.
''' 

#%% SETUP: Install Missing Packages
###############################################################################################################################################################

import subprocess
import sys

# Define the required packages
required_packages = [
    "numpy", "pandas", "matplotlib", "geopy", 
    "geopandas", "statsmodels", "folium","scipy"
]

# Function to install a package
def install_package(package_name):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"Successfully installed {package_name}.")
    except Exception as e:
        print(f"Error installing package {package_name}: {e}")

# Track if any packages were missing
missing_packages = []

# Iterate through the required packages and install missing ones
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        missing_packages.append(package)
        install_package(package)

# Print final messages
if not missing_packages:
    print("All necessary packages are already installed.")
else:
    print(f"The following packages were missing and have been installed: {', '.join(missing_packages)}")



#%% IMPORT PACKAGES, PRINT USED PYTHON VERSION, AND DOCUMENT REQUIREMENTS 
###############################################################################################################################################################
import os
import pathlib
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from geopy.geocoders import Nominatim # For geocoding (converting addresses into coordinates)
from geopy.distance import geodesic # For calculating distances between geographic points
import folium  # For interactive maps
import geopandas as gpd  # For handling GeoJSON and spatial data
import json  # For reading GeoJSON files
from datetime import datetime
from statsmodels.miscmodels.ordinal_model import OrderedModel  # For ordinal logistic regression analysis
from statsmodels.discrete.discrete_model import Logit  # For binary logistic regression analysis used to test the proportional odds assumption
from statsmodels.discrete.discrete_model import MNLogit # Alternatively run a multinominal logit regression
from scipy.stats import norm

# Tested Python version
TESTED_VERSION = (3, 11, 9)  # Python 3.11.9

# Get and print the current Python version
current_version = sys.version_info[:3]  # Extract major, minor, and micro versions
print(f"Python version in use: {'.'.join(map(str, current_version))}")

# Check if the current version matches the tested version and provide a clear warning if not
if current_version == TESTED_VERSION:
    print(f"Code is running in the tested Python version: {'.'.join(map(str, TESTED_VERSION))}")
else:
    print(f"Warning: Tested in Python {'.'.join(map(str, TESTED_VERSION))}, running in {'.'.join(map(str, current_version))}")


#%% IMPORTANT: ADJUSTMENTS TO MAKE FOR RUNNING THE CODE SUCCESSFULLY
############################################################################################################

# === 1. Define Export and Input Paths ===
# Base path: Adjust this path to the specific project directory on your system
base_path = pathlib.Path(os.getenv("OneDrive")) / 'Dokumente/Github/Stadionumfrage_Auswertung'
# Input path: Directory where raw data files are located
import_path = base_path 
# Ensure the following files are available in the import_path:
# - suf.RData: Survey data file
# - umfrage_dates.xlsx: Contains the survey dates
# - postleitzahlen.csv: File of all postal codes in Germany with coordinates
# - georef-germany-kreis.geojson: GeoJSON file of Germany's geographic data

# Output path: Directory where analysis results and outputs will be saved
export_path = base_path 
# === 2. Adjust Independent Variables for Regression Analysis ===
# Independent variables may need to be customized to ensure successful regression analysis
# Regressions may fail when datasets are small relative to the number of independent variables due to overfitting or insufficient data points.
# Variables with "_mean" are continuous and typically easier to process in regression than categorical variables, as they avoid issues related to dummy variable encoding and high dimensionality.

# List of potential independent variables:
# - 'geschlecht': Gender (useful for demographic segmentation)
# - 'alter': Age (continuous variable, easy to interpret)
# - 'einkommen', 'einkommen_mean', or 'log_einkommen_mean': Income (likely influences spending)
# - 'fantyp': Fan type (captures behavioral differences)
# - 'besuch_mainzer_stadion' or 'besuch_mainzer_stadion_mean': Frequency of visiting Mainz games (behavioral)
# - 'besuch_spiele_allgemein' or 'besuch_spiele_allgemein_mean': Frequency of attending games in general
# - 'anreise_bezahlt': Whether travel was paid for (Yes/No)
# - 'anreise_mittel': Mode of transport (affects travel spending)
# - 'entfernung_km': Distance traveled (key factor for travel costs)
# - 'übernachtung': Whether staying overnight (Yes/No)
# - 'übernachtung_art': Accommodation type (directly linked to spending)
# - 'anzahl_erwachsene' or 'anzahl_erwachsene_mean': Number of adults in the group (group effects)
# - 'match': Game that was visited (type of game effects)

# Each model is defined with:
# - Dependent variable: The outcome we want to predict
# - Independent variables: Predictors chosen  (Adjust the list as needed to suit the analysis)


# this defines the regression  model, determined by adding and removing variables based in their significance in the model
models = {
    "anreise_kosten_alle": {
        "dependent_var": "anreise_kosten_alle",  # Total travel costs
        "independent_vars": [
            "entfernung_km", "log_einkommen_mean", "anzahl_erwachsene_mean"
        ]
        },
    "übernachtung_kosten": {
        "dependent_var": "übernachtung_kosten",  # Total travel costs
        "independent_vars": [
            "entfernung_km","fantyp", "anreise_bezahlt"
    ]
    },
    "verpflegung_kosten": {
        "dependent_var": "verpflegung_kosten",  # Food costs
        "independent_vars": ["anzahl_erwachsene_mean", "log_einkommen_mean", 
                             "entfernung_km", "übernachtung"]
    },
     "freizeit_kosten": {
        "dependent_var": "freizeit_kosten",  # Leisure costs
        "independent_vars": ["entfernung_km", "übernachtung", "besuch_spiele_allgemein_mean"]
    }
}


# %% IMPORT DATA
########################################################################################################

# Get the current date in a readable format
current_date = datetime.now().strftime("%Y%m%d")

# Ensure the base path and subdirectories exist
if not base_path.exists():
    raise FileNotFoundError(f"Base path does not exist: {base_path}")

# Import data files
try:
    # Load survey data from .csv file (this is the current survey data, it is not uploaded in the github but just locally saved - the last match is missing)
    suf_data = pd.read_csv(import_path / 'final.csv',   dtype={'G01Q08': 'object', 'G01Q19': 'object' })

    # Load additional datasets (Adjust file name/path if necessary)
    dates = pd.read_excel(import_path / 'umfrage_dates.xlsx')  # File with survey dates 
    ticketverkauf = pd.read_excel(import_path / 'ticketverkauf.xlsx', sheet_name="Ticketverkauf")  # Tickets sold last season: not used yet
    postleitzahlen = pd.read_csv(import_path / 'postleitzahlen.csv', sep=';')  # All PLZs in Germany with coordinates 

    # Load GeoJSON file as JSON
    geojson_path = import_path / "georef-germany-kreis.geojson" # GeoJASON file of germany
    with open(geojson_path, 'r') as f:
        geojson_data = json.load(f)

except FileNotFoundError as e:
    print(f"Error: {e}")
    raise  # Re-raise the error for debugging/logging purposes
except json.JSONDecodeError as e:
    print(f"Error: Failed to decode GeoJSON file at {geojson_path}. {e}")
    raise
except Exception as e:
    print(f"Unexpected error occurred: {e}")
    raise

# Extract total visitors from the ticketverkauf DataFrame
total_visitors = ticketverkauf.loc[11, 'Summe']

# %% TRANSFORM AND CLEAN SURVEY DATA
########################################################################################################

# 0. CREATE A COLUMN THAT MATCHES SURVEY DATES TO GAME
# ----------------------------------------------------------------------------------------------------------------------

match_df = pd.DataFrame({
'Dates': pd.date_range(start="2024-10-24", end="2024-12-31")
})

match_df['Match'] = match_df['Dates'].apply(
lambda x: "BorMG" if x < pd.Timestamp("2024-10-30") else
"POKBAY" if pd.Timestamp("2024-10-29") < x < pd.Timestamp("2024-11-09") else
"BVB" if pd.Timestamp("2024-11-08") < x < pd.Timestamp("2024-12-01") else
"Hoff" if pd.Timestamp("2024-11-30") < x < pd.Timestamp("2024-12-14") else
"DFLBAY"
)

suf_data['Dates'] = pd.to_datetime(suf_data['datestamp']).dt.strftime('%Y-%m-%d')
match_df['Dates'] = pd.to_datetime(match_df['Dates']).dt.strftime('%Y-%m-%d')
suf_data = pd.merge(suf_data, match_df, on='Dates', how='left')
suf_data = suf_data.dropna(subset=['Match'])

# %% TRANSFORM AND CLEAN SURVEY DATA
########################################################################################################

# 1. DEFINE COLUMNS TO KEEP AND RENAME
# ----------------------------------------------------------------------------------------------------------------------
# Map the original survey variables to easier-to-understand names
columns_to_keep = {
    'G01Q01': 'geschlecht', 'G01Q02': 'alter', 'G01Q19': 'wohnort', 'G01Q20': 'beschäftigungsstatus',
    'G01Q21': 'bildungsabschluss', 'G01Q22': 'einkommen', 'G02Q03': 'fantyp',
    'G01Q04[SQ001]': 'motivation_nähe', 'G01Q04[SQ002]': 'motivation_stadt', 
    'G01Q04[SQ003]': 'motivation_stimmung', 'G01Q04[SQ004]': 'motivaton_mannschaft',
    'G01Q04[SQ005]': 'motivation_bekannte', 'G01Q04[SQ006]': 'motivation_stadion', 
    'G01Q04[SQ007]': 'motivation_fussball', 'G01Q04[SQ008]': 'motivation_sonstiges', 
    'G01Q05': 'besuch_mainzer_stadion', 'G02Q23': 'besuch_spiele_allgemein',
    'G02Q24': 'anzahl_erwachsene', 'G02Q25': 'anzahl_kinder', 'G01Q08': 'anreiseort',
    'G01Q09': 'anreise_bezahlt', 'G03Q10': 'anreise_mittel', 'G03Q23': 'anreise_kosten_auto',
    'G03Q26': 'anreise_kosten_öffis', 'G02Q27': 'anreise_kosten_kombi', 'G04Q12': 'übernachtung',
    'G04Q13': 'übernachtung_art', 'G04Q14': 'übernachtung_kosten', 'G04Q15': 'übernachtung_anzahl',
    'G05Q16': 'verpflegung_kosten', 'G05Q17': 'freizeit_kosten', 'G05Q25': 'interesse_kombiticket',
    'G05Q26[SQ001]': 'kombiticket_kultur', 'G05Q26[SQ002]': 'kombiticket_aktiv', 
    'G05Q26[SQ003]': 'kombiticket_kulinarik', 'G05Q26[SQ004]': 'kombiticket_unterhaltung', 
    'G05Q26[SQ005]': 'kombiticket_sonstiges', 'interviewtime': 'interviewtime', 'Dates': 'date', 'Match':'match'
}

# Retain only the relevant columns and rename them
suf_data_filtered = suf_data[list(columns_to_keep.keys())].copy()  # Filter dataset to specified columns
suf_data_filtered.rename(columns=columns_to_keep, inplace=True)  # Apply the mapping


# 2. REMOVE UNREASONABLE DATA
# ----------------------------------------------------------------------------------------------------------------------
########################################################################################################
# Set these variables to True or False to control which cleaning steps to apply
apply_step_1 = False   # only consider answers on survey dates
apply_step_2 = True    # remove unreasonable data based on age and educational attainment (age ≤ 14 but bildungsabschluss ≠ "Kein Abschluss")
apply_step_3 = True    # remove rows with missing values in all specified keycolumns
apply_step_4 = True    # remove rows where alter > 100 but leave missing alter intact

# Define key columns for step 3
key_columns = ['geschlecht', 'anreise_bezahlt', 'übernachtung',  'verpflegung_kosten', 'freizeit_kosten', 'interesse_kombiticket' ]

if apply_step_1:
    # Filter data to include only responses matching the survey dates
    initial_row_count = len(suf_data_filtered)

    suf_data_filtered['date'] = pd.to_datetime(suf_data_filtered['date'], errors='coerce')   # Ensure date columns are in datetime format
    dates['date'] = pd.to_datetime(dates['date'], errors='coerce')
    suf_data_filtered = suf_data_filtered[suf_data_filtered['date'].isin(dates['date'])].copy()    # Filter rows where 'date' is in the list of valid dates

    final_row_count = len(suf_data_filtered)
    print(f"Cleaning step 1: Removed {initial_row_count - final_row_count} rows filtering data based on survey dates. Remaining: {final_row_count}")
else:
    print("Cleaning step 1 skipped: Filtering data based on survey dates")

if apply_step_2:
    # Remove data with unreasonable age and educational attainment
    initial_row_count = len(suf_data_filtered)

    # Filter out rows with age ≤ 14 but not "Kein Abschluss"
    suf_data_filtered = suf_data_filtered[~((suf_data_filtered['alter'] <= 14) & (suf_data_filtered['bildungsabschluss'] != "Kein Abschluss"))]

    final_row_count = len(suf_data_filtered)
    print(f"Cleaning step 2: Removed {initial_row_count - final_row_count} rows based on unreasonable age and educational attainment. Remaining: {final_row_count}")
else:
    print("Cleaning step 2 skipped: Removing unreasonable data based on age and educational attainment")

if apply_step_3:
    # Remove rows with empty key columns
    initial_row_count = len(suf_data_filtered)

    # Create mask for rows where all specified key columns are missing or empty
    mask = suf_data_filtered[key_columns].apply(lambda x: x.str.strip().eq("") | x.isna()).all(axis=1)

    # Step 3 removes a lot of columns. I save them in a seperate dataframe to analyze them.
    removed_rows = suf_data_filtered[mask].copy() 

    # Keep rows where at least one key column is non-empty
    suf_data_filtered = suf_data_filtered[~mask].copy()

    final_row_count = len(suf_data_filtered)
    print(f"Cleaning step 3: Removed {initial_row_count - final_row_count} rows where all specified key columns are empty. Remaining: {final_row_count}")
else:
    print("Cleaning step 3 skipped: Removing rows where all specified key columns are empty")

if apply_step_4:
    # Remove rows where alter > 100 but keep missing alter intact
    initial_row_count = len(suf_data_filtered)

    # Filter out rows where alter > 100, but keep NaN
    suf_data_filtered = suf_data_filtered[(suf_data_filtered['alter'] <= 100) | (suf_data_filtered['alter'].isna())]

    final_row_count = len(suf_data_filtered)
    print(f"Cleaning step 4: Removed {initial_row_count - final_row_count} rows where 'alter' > 100. Remaining: {final_row_count}")
else:
    print("Cleaning step 4 skipped: Removing rows where 'alter' > 100")


# %% FILL IN MISSING DATA CONSISTENT WITH OTHER INFORMATION IN THE DATA
########################################################################################################
# 1. Fill missing values for 'besuch_spiele_allgemein' based on the responses in 'besuch_mainzer_stadion'.
# 2. Fill travel-related fields:
#    - Set 'anreise_mittel' to 'Nicht bezahlt' if no travel costs were incurred.
#    - Combine multiple travel cost columns ('auto', 'öffis', 'kombi') into a unified column 'anreise_kosten_alle'.
#    - Set unified travel costs to "0 €" for unpaid travel.
# 3. Fill accommodation-related fields:
#    - Mark 'übernachtung_art' as "Keine" if no overnight stay occurred.
#    - Set 'übernachtung_kosten' to "0 €" for private or no accommodation.
# 4. Fill missing 'anreiseort' values using the respondent's 'wohnort'.

# Step 1: Fill missing values for 'besuch_spiele_allgemein' based on the responses in 'besuch_mainzer_stadion'.
def fill_besuch_spiele_allgemein(data):
    mapping = {
        "Zum ersten Mal": "Zum ersten Mal",
        "Selten (1–3-mal pro Saison)": "Selten (1–6-mal pro Saison)",
        "Gelegentlich (4–6-mal pro Saison)": "Selten (1–6-mal pro Saison)",
        "Regelmäßig (7–11-mal pro Saison)": "Gelegentlich (7–11-mal pro Saison)",
        "Oft (Mehr als 11-mal pro Saison)": "Regelmäßig (12–25-mal pro Saison)"
    }
    for key, value in mapping.items():
        data.loc[
            (data['besuch_spiele_allgemein'].isna()) & (data['besuch_mainzer_stadion'] == key),
            'besuch_spiele_allgemein'
        ] = value
    return data

# Step 2: Fill travel-related fields
def fill_anreise_fields(data):
    # Set 'anreise_mittel' to 'Nicht bezahlt' if no travel costs were incurred.
    data.loc[
        data['anreise_bezahlt'] == "Nein (z.B. Deutschlandticket, im Stadionticket enthaltenes ÖVPN-Ticket, Fahrrad, zu Fuß)", 
        'anreise_mittel'
    ] = "Nicht bezahlt"

    # Combine multiple travel cost columns ('auto', 'öffis', 'kombi') into a unified column 'anreise_kosten_alle'.
    data['anreise_kosten_alle'] = data.apply(
        lambda row: row['anreise_kosten_auto']
        if pd.notna(row['anreise_kosten_auto']) and row['anreise_kosten_auto'].strip()
        else row['anreise_kosten_öffis']
        if pd.notna(row['anreise_kosten_öffis']) and row['anreise_kosten_öffis'].strip()
        else row['anreise_kosten_kombi']
        if pd.notna(row['anreise_kosten_kombi']) and row['anreise_kosten_kombi'].strip()
        else "",
        axis=1
    )

    # Set unified travel costs to "0 €" for unpaid travel.
    data.loc[
        data['anreise_mittel'] == "Nicht bezahlt", 
        'anreise_kosten_alle'
    ] = "0 €"

    # Position 'anreise_kosten_alle' directly after 'anreise_mittel'
    columns = list(data.columns)  # Get the list of columns
    position = columns.index('anreise_mittel') + 1  # Find position after 'anreise_mittel'
    columns.insert(position, columns.pop(columns.index('anreise_kosten_alle')))  # Move 'anreise_kosten_alle'
    data = data[columns]  # Reorder the DataFrame

    return data

# Step 3: Fill accommodation-related fields
def fill_accommodation_fields(data):
    # Mark 'übernachtung_art' as "Keine" if no overnight stay occurred.
    data.loc[data['übernachtung'] == "Nein", 'übernachtung_art'] = "Keine"

    # Set 'übernachtung_kosten' to "0 €" for private or no accommodation.
    data.loc[
        data['übernachtung_art'].isin(["Privat (z.B. bei Freunden, Familie)", "Keine"]), 
        'übernachtung_kosten'
    ] = "0 €"
    return data


# Step 4: Fill missing 'anreiseort' values using the respondent's 'wohnort'.
def fill_anreiseort(data):
    data.loc[data['anreiseort'].isna(), 'anreiseort'] = data['wohnort']
    return data


# Apply the functions defined in Steps 1-4
suf_data_filtered = fill_besuch_spiele_allgemein(suf_data_filtered)
suf_data_filtered = fill_anreise_fields(suf_data_filtered)
suf_data_filtered = fill_accommodation_fields(suf_data_filtered)
suf_data_filtered = fill_anreiseort(suf_data_filtered)


# %% CREATE MEAN VALUE COLUMNS FOR NUMBER CATEGORIES
########################################################################################################

def map_to_mean_column(data, column_name, mapping, new_column_name):
    """
    Maps categorical values in a column to their mean or numeric equivalents
    and places the new column next to the original column.

    Parameters:
    - data: DataFrame to modify
    - column_name: Original column name to map
    - mapping: Dictionary with the mapping logic
    - new_column_name: Name of the new column to create

    Returns:
    - Updated DataFrame with the new column placed next to the original column
    """
    # Map the values to the new column
    data[new_column_name] = data[column_name].map(mapping)

    # Reorder columns to place the new column next to the original
    columns = list(data.columns)  # Get the list of columns
    original_index = columns.index(column_name)  # Find the index of the original column
    # Move the new column to be right after the original column
    columns.insert(original_index + 1, columns.pop(columns.index(new_column_name)))
    return data[columns]

# Replace "1 € - 9 €" with "Unter 10 €" in the verpflegung_kosten column (there was a error in the survey: two answers for the same spending category)
if 'verpflegung_kosten' in suf_data_filtered.columns:
    suf_data_filtered['verpflegung_kosten'] = suf_data_filtered['verpflegung_kosten'].replace("1 € - 9 €", "Unter 10 €")

# Define mappings for all relevant columns
mappings = {
    "einkommen": {
        "Kein Einkommen": 0,
        "Unter 1.500 €": 750,
        "1.500 € – 2.499 €": 2000,
        "2.500 € – 3.499 €": 3000,
        "3.500 € – 4.999 €": 4250,
        "5.000 € und mehr": 5000,
        "Keine Angabe": np.nan,
        "": np.nan
    },
    "besuch_mainzer_stadion": {
        "Zum ersten Mal": 1,
        "Selten (1–3-mal pro Saison)": 2,
        "Gelegentlich (4–6-mal pro Saison)": 5,
        "Regelmäßig (7–11-mal pro Saison)": 9,
        "Oft (Mehr als 11-mal pro Saison)": 14.5,
        "": np.nan
    },
    "besuch_spiele_allgemein": {
         "Zum ersten Mal": 1,
        "Selten (1–6-mal pro Saison)": 3.5,
        "Gelegentlich (7–11-mal pro Saison)": 9,
        "Regelmäßig (12–25-mal pro Saison)": 18.5,
        "Oft (mehr als 25-mal pro Saison)": 30,
        "": np.nan
    },
    "anzahl_erwachsene": {
        "Allein": 1,
        "Zu zweit": 2,
        "Zu dritt": 3,
        "Mit mehr als 3 Personen": 4,
        "": np.nan
    },
    "anzahl_kinder": {
        "Keinem": 0,
        "Einem": 1,
        "Zwei": 2,
        "Mit mehr als 2 Kindern": 3,
        "": np.nan
    },
    "übernachtung_anzahl": {
        "1 Nacht": 1,
        "2 Nächte": 2,
        "Mehr als 2 Nächte": 3,
        "": np.nan
    },
    "anreise_kosten_alle": {
        "0 €": 0,
        "Unter 10 €": 5,
        "10 € – 50 €": 30,
        "51 € – 100 €": 75.5,
        "Mehr als 100 €": 125,
        "": np.nan
    },
    "übernachtung_kosten": {
        "0 €": 0,
        "Unter 50 €": 25,
        "51€ – 100 €": 75.5,
        "101 € – 150 €": 125.5,
        "Mehr als 150 €": 200,
        "": np.nan
    },
    "verpflegung_kosten": {
        "Unter 10 €": 5,
        "10 € – 50 €": 30,
        "51 € – 100 €": 75.5,
        "Mehr als 100 €": 125,
        "": np.nan
    },
    "freizeit_kosten": {
        "0 €": 0,
        "1 € -  9 €": 5,
        "10 € – 50 €": 30,
        "51 € – 100 €": 75.5,
        "Mehr als 100 €": 125,
        "": np.nan
    }
}

# Apply mappings to the respective columns
for column, mapping in mappings.items():
    new_column = f"{column}_mean"  # Generate new column name dynamically
    suf_data_filtered = map_to_mean_column(suf_data_filtered, column, mapping, new_column)

# %% CREATE A DISTANCE COLUMN TRAVELED TO THE ARENA
########################################################################################################

# Set up the geolocator
geolocator = Nominatim(user_agent="geo_distance_calculator")

# Define the target PLZ (e.g., Mainz)
TARGET_PLZ = "55128"
postleitzahlen['name'] = postleitzahlen['name'].astype(str).str.strip()  # Ensure consistency

# Step 1: Validate 'anreiseort' values and replace invalid PLZs with NaN
valid_plz_list = postleitzahlen['name'].tolist()

def validate_plz(plz, valid_plz_list):
    """
    Validates if a PLZ exists in the list of valid PLZs.
    Returns the PLZ if valid, otherwise NaN.
    """
    return plz if plz in valid_plz_list else np.nan

suf_data_filtered['anreiseort'] = suf_data_filtered['anreiseort'].astype(str).str.strip()
suf_data_filtered['anreiseort'] = suf_data_filtered['anreiseort'].apply(lambda plz: validate_plz(plz, valid_plz_list))

# Step 2: Extract coordinates for valid PLZs (once for all PLZs)
def extract_coordinates(plz, postleitzahlen_df):
    """
    Extracts coordinates for a given PLZ from the postleitzahlen DataFrame.
    Returns NaN if the PLZ is invalid or not found.
    """
    if pd.notna(plz):
        coords = postleitzahlen_df.loc[postleitzahlen_df['name'] == plz, 'geo_point_2d'].values
        if len(coords) > 0:
            if isinstance(coords[0], str):
                return tuple(map(float, coords[0].split(',')))
    return (np.nan, np.nan)

# Extract the target coordinates
target_coordinates = extract_coordinates(TARGET_PLZ, postleitzahlen)

# Extract coordinates for 'anreiseort' using the validated PLZs
suf_data_filtered['coordinates'] = suf_data_filtered['anreiseort'].apply(
    lambda plz: extract_coordinates(plz, postleitzahlen)
)

# Step 3: Calculate distance for valid coordinates
def calculate_distance(row_coordinates, target_coords):
    """
    Calculates the geodesic distance between two coordinate pairs.
    Returns NaN if any coordinate is invalid.
    """
    if pd.notna(target_coords[0]) and pd.notna(row_coordinates[0]):
        try:
            return geodesic(row_coordinates, target_coords).km
        except ValueError as e:
            print(f"Error calculating distance for coordinates {row_coordinates} to target {target_coords}: {e}")
            return np.nan
    return np.nan

# Calculate the distance from each 'coordinates' to the target coordinates
suf_data_filtered['entfernung_km'] = suf_data_filtered['coordinates'].apply(
    lambda coords: calculate_distance(coords, target_coordinates)
)

# Step 4: Clean up and reorder columns
suf_data_filtered.drop(columns=['coordinates'], inplace=True)

# Reorder 'entfernung_km' column to be next to 'anreiseort'
columns = list(suf_data_filtered.columns)
anreiseort_index = columns.index('anreiseort')
columns.insert(anreiseort_index + 1, columns.pop(columns.index('entfernung_km')))
suf_data_filtered = suf_data_filtered[columns]

# %% PLZ MAP PLOTTING
########################################################################################################

# Filter for valid PLZs in `anreiseort`
valid_plz_df = suf_data_filtered[['anreiseort']].dropna()  # Drop rows with NaN in 'anreiseort'

# Merge with `postleitzahlen` to get coordinates for the PLZs in `anreiseort`
valid_plz_df = valid_plz_df.merge(
    postleitzahlen[['name', 'geo_point_2d']],
    left_on='anreiseort',
    right_on='name',
    how='inner'
)

# Extract latitude and longitude from 'geo_point_2d'
valid_plz_df['coordinates'] = valid_plz_df['geo_point_2d'].apply(
    lambda x: tuple(map(float, x.split(','))) if isinstance(x, str) else (np.nan, np.nan)
)

# Initialize a Folium map centered on Germany
map_germany = folium.Map(location=[51.1657, 10.4515], zoom_start=6)

# Add markers for each PLZ in 'anreiseort'
for _, row in valid_plz_df.iterrows():
    plz = row['anreiseort']
    coords = row['coordinates']
    if pd.notna(coords[0]):  # Skip invalid or NaN coordinates
        folium.Marker(
            location=[coords[0], coords[1]],  # Latitude, Longitude
            popup=f"PLZ: {plz}",              # Add popup info
            icon=folium.Icon(color="blue", icon="home")
        ).add_to(map_germany)


# Save the plot to the specified file path
# Define the output file path
output_map_path = os.path.join(export_path, f"{current_date}_plz_map.html")

# Save the map as an HTML file
map_germany.save(output_map_path)

# %% CHOROPLETH MAP PLOTTING
########################################################################################################

# Step 1: Merge and prepare the data
# Merge survey data with PLZ data
merged_df = pd.merge(
    suf_data_filtered,
    postleitzahlen[['name', 'krs_code']],
    left_on='anreiseort',
    right_on='name',
    how='left'
)

# Drop the redundant 'name' column
merged_df = merged_df.drop(columns=['name'])

# Format 'krs_code' for consistency
merged_df['krs_code'] = (
    merged_df['krs_code']
    .astype('Int64')  # Support nullable integers
    .astype(str)      # Convert to string for uniformity
    .apply(lambda x: x.zfill(5) if len(x) == 4 and x.isdigit() else x)  # Add leading zeros
)

# Filter and count unique occurrences of 'krs_code'
filtered_krs_code = merged_df['krs_code'][merged_df['krs_code'].notna() & (merged_df['krs_code'] != '<NA>')]
krs_code_counts = filtered_krs_code.value_counts()

# Convert counts to DataFrame for merging with GeoJSON data
krs_code_counts_df = krs_code_counts.reset_index()
krs_code_counts_df.columns = ['krs_code', 'count']


# Step 2: Prepare the GeoJSON file 
# Extract all 'krs_code' values from the GeoJSON file
geo_krs_codes = [
    feature["properties"]["krs_code"][0]  # Use first element if 'krs_code' is a list
    if isinstance(feature["properties"]["krs_code"], list)
    else feature["properties"]["krs_code"]
    for feature in geojson_data["features"]
]

# Create a DataFrame for GeoJSON 'krs_code'
geo_krs_df = pd.DataFrame({"krs_code": geo_krs_codes})
geo_krs_df["krs_code"] = geo_krs_df["krs_code"].astype(str)

# Merge counts with GeoJSON 'krs_code'
complete_krs_df = pd.merge(
    geo_krs_df,
    krs_code_counts_df,
    on="krs_code",
    how="left"
).fillna({"count": 0})  # Fill missing values with 0
complete_krs_df["count"] = complete_krs_df["count"].astype(int)

# Convert GeoJSON to a GeoDataFrame
geo_gdf = gpd.GeoDataFrame.from_features(geojson_data["features"])

# Ensure 'krs_code' consistency in GeoDataFrame
geo_gdf["krs_code"] = geo_gdf["krs_code"].apply(
    lambda x: x[0] if isinstance(x, list) else x
).astype(str)

# Merge GeoDataFrame with counts
choropleth_gdf = geo_gdf.merge(
    complete_krs_df,
    on="krs_code",
    how="left"
)
choropleth_gdf["count"] = choropleth_gdf["count"].fillna(0).astype(int)


# Step 3: Plot the Choropleth map
# Darken shading of a colormap
def truncate_colormap(cmap, minval=0, maxval=1.0, n=100):
    """Truncate an existing colormap to a narrower range"""
    new_cmap = LinearSegmentedColormap.from_list(
        f"trunc({cmap.name},{minval:.2f},{maxval:.2f})",
        cmap(np.linspace(minval, maxval, n))
    )
    return new_cmap

# Use a colormap and darken its shading to improve visibility of all parts
original_cmap = plt.cm.Reds
darkened_cmap = truncate_colormap(original_cmap, minval=0.05, maxval=1.3)

# Plot the Choropleth Map with a custom colormap
fig, ax = plt.subplots(figsize=(12, 12))
choropleth_gdf.plot(
    column="count",            # Column to color by
    cmap=darkened_cmap,        # Custom colormap
    legend=True,               # Add legend
    edgecolor="darkgrey",          # Remove black outlines
    linewidth=0.2,   # Adjust border thickness 
    vmax=30,                    # Maximum value for the color bar
    missing_kwds={"color": "lightgrey", "label": "No Data"},  # Handle missing data
    legend_kwds={
        "shrink": 0.4,         # Shrink the color bar size
        "orientation": "vertical",  # Keep it vertical
        "pad": 0.02            # Adjust space between the map and the legend
    },
    ax=ax                      # Axes to plot on
)

# Remove axes for cleaner visualization
ax.axis("off")

# Remove the legend border
colorbar = ax.get_figure().axes[-1]  # Access the color bar
colorbar.spines[:].set_visible(False)  # Remove the border

# Save the plot
plt.savefig(os.path.join(export_path, f"{current_date}_choropleth_map.png"), dpi=300, bbox_inches="tight")
print(f"Choropleth Map saved successfully.")

# Set the output file path
output_file = os.path.join(export_path, f"{current_date}_choropleth_data.xlsx")

# Save the data to Excel
choropleth_gdf.to_excel(output_file, index=False, sheet_name="Choropleth Counts")

#%% DESCRIPTIVE STATISTICS 
################################################################################################################################################################

# Data counts 
initial_count = len(suf_data)  
cleaned_count = len(suf_data_filtered)
data_counts_df = pd.DataFrame([{'Data': 'Initial', 'Count': initial_count}, {'Data': 'Cleaned', 'Count': cleaned_count}])

# Set a threshold for categorical variables
CATEGORICAL_THRESHOLD = 20

# Initialize lists to hold descriptive statistics
categorical_statistics = []
numerical_statistics = []

# Function to calculate numerical statistics for a column
def calculate_numerical_stats(column, name):
    stats = column.agg(['max', 'min', 'median', 'mean', 'std']).to_dict()
    return [
        {'Variable': name, 'Statistic': 'Max', 'Value': stats['max']},
        {'Variable': name, 'Statistic': 'Min', 'Value': stats['min']},
        {'Variable': name, 'Statistic': 'Median', 'Value': stats['median']},
        {'Variable': name, 'Statistic': 'Mean', 'Value': stats['mean']},
        {'Variable': name, 'Statistic': 'Std Dev', 'Value': stats['std']}
    ]

# Iterate through columns to calculate stats
for column in suf_data_filtered.columns:
    unique_values = suf_data_filtered[column].nunique()
    if suf_data_filtered[column].dtype == 'object' and unique_values <= CATEGORICAL_THRESHOLD:
        # Categorical statistics
        value_counts = suf_data_filtered[column].value_counts()
        non_empty_count = suf_data_filtered[column].notna().sum()
        for category, count in value_counts.items():
            percentage_non_empty = (count / non_empty_count * 100) if non_empty_count > 0 else 0
            categorical_statistics.append({
                'Variable': column,
                'Category': category,
                'Count': count,
                'Percentage_Total': count / cleaned_count * 100,
                'Percentage_Non_Empty': percentage_non_empty
            })
    elif pd.api.types.is_numeric_dtype(suf_data_filtered[column]):
        # Numerical statistics
        numerical_statistics.extend(calculate_numerical_stats(suf_data_filtered[column], column))

# Convert statistics to DataFrames
categorical_statistics_df = pd.DataFrame(categorical_statistics)
numerical_statistics_df = pd.DataFrame(numerical_statistics)

# Write descriptive statistics to Excel
with pd.ExcelWriter(export_path / f"{current_date}_descriptive_statistics.xlsx", engine='openpyxl') as writer:
    data_counts_df.to_excel(writer, sheet_name='Data Counts', index=False)
    categorical_statistics_df.to_excel(writer, sheet_name='Categorical Statistics', index=False)
    numerical_statistics_df.to_excel(writer, sheet_name='Numerical Statistics', index=False)

print(f"\nDescriptive statistics saved successfully.")

#%% PREPERATION OF VARIABLES FOR REGRESSION ANLYSIS
################################################################################################################################################################

# Add a logarithmic transformation of 'einkommen_mean' to the DataFrame
suf_data_filtered['log_einkommen_mean'] = np.log1p(suf_data_filtered['einkommen_mean'])

# Define categorical variables
categorical_vars = [
    'geschlecht', 'wohnort', 'beschäftigungsstatus', 'bildungsabschluss',
    'fantyp', 'motivation_nähe', 'motivation_stadt', 'motivation_stimmung',
    'motivaton_mannschaft', 'motivation_bekannte', 'motivation_stadion', 
    'motivation_fussball', 'anreise_bezahlt', 'anreise_mittel', 'motivation_sonstiges', 'übernachtung', 
    'übernachtung_art', 'interesse_kombiticket', 'kombiticket_kultur', 
    'kombiticket_aktiv', 'kombiticket_kulinarik', 'kombiticket_unterhaltung', 
    'kombiticket_sonstiges', 'date', "match"
]

# Define numeric variables
numeric_vars = [
    'alter', 'einkommen_mean', 'besuch_mainzer_stadion_mean', 
    'besuch_spiele_allgemein_mean', 'anzahl_erwachsene_mean', 
    'anzahl_kinder_mean', 'entfernung_km', 'anreise_kosten_alle_mean',
    'übernachtung_kosten_mean', 'übernachtung_anzahl_mean', 
    'verpflegung_kosten_mean', 'freizeit_kosten_mean', 'interviewtime', 'log_einkommen_mean'
]

# Define ordinal variables and their respective orders
ordinal_vars = {
    'einkommen': [
        'Kein Einkommen', 'Unter 1.500 €', '1.500 € – 2.499 €',
        '2.500 € – 3.499 €', '3.500 € – 4.999 €', '5.000 € und mehr'
    ],
    'besuch_mainzer_stadion': [
        'Zum ersten Mal', 'Selten (1–3-mal pro Saison)', 
        'Gelegentlich (4–6-mal pro Saison)', 'Regelmäßig (7–11-mal pro Saison)', 
        'Oft (Mehr als 11-mal pro Saison)'
    ],
    'besuch_spiele_allgemein': [
        'Zum ersten Mal', 'Selten (1–6-mal pro Saison)', 
        'Gelegentlich (7–11-mal pro Saison)', 'Regelmäßig (12–25-mal pro Saison)', 
        'Oft (mehr als 25-mal pro Saison)'
    ],
    'anzahl_erwachsene': [
        'Allein', 'Zu zweit', 'Zu dritt', 'Mit mehr als 3 Personen'
    ],
    'anzahl_kinder': [
        'Keinem', 'Einem', 'Zwei', 'Mit mehr als 2 Kindern'
    ],
    'anreise_kosten_alle': [
        '0 €', 'Unter 10 €', '10 € – 50 €', '51 € – 100 €', 'Mehr als 100 €'
    ],
    'anreise_kosten_auto': [
        'Unter 10 €', '10 € – 50 €', '51 € – 100 €'
    ],
    'anreise_kosten_öffis': [
        '10 € – 50 €', '51 € – 100 €', 'Mehr als 100 €'
    ],
    'anreise_kosten_kombi': [
        'Unter 10 €', 'Mehr als 100 €'
    ],
    'übernachtung_kosten': [
        '0 €', 'Unter 50 €', '51€ – 100 €',  '101 € – 150 €', 'Mehr als 150 €'
    ],
    'übernachtung_anzahl': [
        '1 Nacht', '2 Nächte', 'Mehr als 2 Nächte'
    ],
    'verpflegung_kosten': [
        'Unter 10 €', '10 € – 50 €', '51 € – 100 €', 'Mehr als 100 €'
    ],
    'freizeit_kosten': [
        '0 €', '1 € -  9 €', '10 € – 50 €', '51 € – 100 €', 'Mehr als 100 €'
    ]
}

# Convert `date` to categorical
if 'date' in suf_data_filtered.columns:
    suf_data_filtered['date'] = suf_data_filtered['date'].astype(str).astype('category')

# Convert ordinal variables to ordered categories
for var, order in ordinal_vars.items():
    if var in suf_data_filtered.columns:
        suf_data_filtered[var] = pd.Categorical(
            suf_data_filtered[var],
            categories=order,
            ordered=True
        )


# Convert other categorical variables to unordered `category` with explicit categories
for var in set(categorical_vars) - set(ordinal_vars.keys()):
    if var in suf_data_filtered.columns:
        # Exclude NaN values when defining categories
        unique_values = [val for val in suf_data_filtered[var].unique() if pd.notnull(val)]
        # Convert to categorical using only non-null categories
        suf_data_filtered[var] = pd.Categorical(
            suf_data_filtered[var],
            categories=unique_values,  # Set explicit categories (excluding NaN)
            ordered=False  # Explicitly mark as unordered
        )


#%% DEFINE FUNCTIONS FOR RUNNING ORDINAL LOGISTIC REGRESSION ANALYSIS
################################################################################################################################################################

# Function for Handling Missing Values 
def handle_missing_values(df, method='impute', df_name='DataFrame'):
    """
    Handles missing values in the DataFrame.

    Parameters:
        df (DataFrame): The input DataFrame with potential missing values.
        method (str): The method to handle missing values ('impute' or 'drop').
                      - 'impute': Imputes missing values.
                      - 'drop': Drops rows with missing values.
        df_name (str): Name of the DataFrame for logging purposes.

    Returns:
        DataFrame: Processed DataFrame with missing values handled.
    """
    df = df.copy()  # Avoid modifying the original DataFrame
    
    if method == 'impute':
        # Impute missing values for ordinal variables
        for var in ordinal_vars:
            if var in df.columns:
                df[var] = df[var].fillna(df[var].mode()[0])

        # Impute missing values for other categorical variables
        for col in categorical_vars:
            if col in df.columns and col not in ordinal_vars:
                df[col] = df[col].fillna(df[col].mode()[0])
        
        # Impute missing values for numeric variables
        for col in numeric_vars:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
    elif method == 'drop':
        # Drop rows with any missing values
        df = df.dropna()

    return df


# Function for the ordinal logit regression
def run_ordinal_logit_regression(dependent_var, independent_vars, data, model_name):
    """
    Runs an ordinal logistic regression and returns the results and predictions.

    Parameters:
        dependent_var (str): Dependent variable for regression.
        independent_vars (list): List of independent variables.
        data (DataFrame): Input dataset.
        model_name (str): Name of the model for logging.

    Returns:
        tuple: Regression result, predicted probabilities, and category labels.
    """
    # Prepare the dataset
    subset = data[[dependent_var] + independent_vars].copy()
    subset = handle_missing_values(subset, method='impute', df_name=model_name)

    # Check for remaining missing values in each row
    missing_values_per_row = subset.isnull().sum(axis=1)
    print("\nRows with missing values after imputation:")
    rows_with_missing = missing_values_per_row[missing_values_per_row > 0]
    if rows_with_missing.empty:
        print("No missing values remain in the subset.")
    else:
        print(rows_with_missing)
   
    # Prepare independent and dependent variables
    X = subset[independent_vars]
    y = subset[dependent_var]

    # Validate dependent variable is categorical and convert to numeric
    if not isinstance(y.dtype, pd.CategoricalDtype):
        raise ValueError(f"Dependent variable '{dependent_var}' must be categorical!")
    y_numeric = y.cat.codes

    # Identify categorical and numeric columns in independent variables
    categorical_cols = [col for col in X.columns if isinstance(X[col].dtype, pd.CategoricalDtype)]
    numeric_cols = [col for col in X.columns if col not in categorical_cols]

    # Apply one-hot encoding to categorical columns and drop the reference category
    if categorical_cols:
        # Drop the first category to set it as the reference category
        X_encoded_categorical = pd.get_dummies(X[categorical_cols], drop_first=True).astype(int)
    else:
        X_encoded_categorical = pd.DataFrame()

    # Keep numeric columns unchanged
    if numeric_cols:
        X_encoded_numeric = X[numeric_cols]
    else:
        X_encoded_numeric = pd.DataFrame()
    

    # Combine numeric and encoded categorical variables
    if not X_encoded_numeric.empty and not X_encoded_categorical.empty:
        X_encoded = pd.concat([X_encoded_numeric, X_encoded_categorical], axis=1)
    elif not X_encoded_numeric.empty:
        X_encoded = X_encoded_numeric
    elif not X_encoded_categorical.empty:
        X_encoded = X_encoded_categorical
    else:
        raise ValueError("Both numeric and categorical variables are missing in the independent variables!")

    # Validate alignment between X_encoded and y_numeric
    if X_encoded.shape[0] != y_numeric.shape[0]:
        raise ValueError(
            f"Mismatch between rows in independent variables ({X_encoded.shape[0]}) "
            f"and dependent variable ({y_numeric.shape[0]})."
        )

    # Fit the ordinal logistic regression model
    model = OrderedModel(y_numeric, X_encoded, distr='logit')
    result = model.fit(method='bfgs', disp=False)  # Suppress detailed optimization output for cleaner logs

    # Print the regression results
    print(f"\nSummary of {model_name} regression results:")
    print(result.summary())

    # Predict probabilities for each category
    predicted_probs = result.predict(X_encoded)

    return result, predicted_probs, y.cat.categories


# Function for the multinominal logit regression
def run_multinomial_logit_regression(dependent_var, independent_vars, data, model_name):
    """
    Runs a multinomial logistic regression and returns the results and predicted probabilities.

    Parameters:
        dependent_var (str): Dependent variable for regression.
        independent_vars (list): List of independent variables.
        data (DataFrame): Input dataset.
        model_name (str): Name of the model for logging.

    Returns:
        tuple: Regression result, predicted probabilities, and category labels.
    """
    # Prepare the dataset
    subset = data[[dependent_var] + independent_vars].copy()
    subset = handle_missing_values(subset, method='impute', df_name=model_name)

    # Prepare independent and dependent variables
    X = subset[independent_vars]
    y = subset[dependent_var]

    # Ensure dependent variable is categorical and encoded as integers
    if not isinstance(y.dtype, pd.CategoricalDtype):
        y = y.astype('category')
    y_encoded = y.cat.codes

    # Identify categorical and numeric columns in independent variables
    categorical_cols = [col for col in X.columns if isinstance(X[col].dtype, pd.CategoricalDtype)]
    numeric_cols = [col for col in X.columns if col not in categorical_cols]

    # Apply one-hot encoding to categorical columns and drop the reference category
    if categorical_cols:
        X_encoded_categorical = pd.get_dummies(X[categorical_cols], drop_first=True).astype(int)
    else:
        X_encoded_categorical = pd.DataFrame()

    # Keep numeric columns unchanged
    if numeric_cols:
        X_encoded_numeric = X[numeric_cols]
    else:
        X_encoded_numeric = pd.DataFrame()

    # Combine numeric and encoded categorical variables
    if not X_encoded_numeric.empty and not X_encoded_categorical.empty:
        X_encoded = pd.concat([X_encoded_numeric, X_encoded_categorical], axis=1)
    elif not X_encoded_numeric.empty:
        X_encoded = X_encoded_numeric
    elif not X_encoded_categorical.empty:
        X_encoded = X_encoded_categorical
    else:
        raise ValueError("Both numeric and categorical variables are missing in the independent variables!")

    # Validate alignment between X_encoded and y_encoded
    if X_encoded.shape[0] != y_encoded.shape[0]:
        raise ValueError(
            f"Mismatch between rows in independent variables ({X_encoded.shape[0]}) "
            f"and dependent variable ({y_encoded.shape[0]})."
        )

    # Fit the multinomial logistic regression model
    model = MNLogit(y_encoded, X_encoded)
    result = model.fit(method='bfgs', maxiter=2000, disp=True)

    # Print the regression results
    print(f"\nSummary of {model_name} multinomial regression results:")
    print(result.summary())

    # Predict probabilities for each category
    predicted_probs = result.predict(X_encoded)

    return result, predicted_probs, y.cat.categories


# Function for summarizing regression results
def analyze_results(predicted_probs, categories, mappings, dependent_var, total_visitors=total_visitors, ci_level=95):
    """
    Analyzes regression results to calculate likelihoods, confidence intervals, 
    and total expected spending while including totals for Average Likelihood, Midpoint, 
    and Seasonal Total (if total_visitors is provided).

    Parameters:
        predicted_probs (DataFrame): Predicted probabilities for each category.
        categories (list): Categories of the dependent variable.
        mappings (dict): Mapping of categories to midpoints.
        dependent_var (str): Dependent variable name.
        total_visitors (int): Total number of visitors for seasonal scaling (optional).
        ci_level (float): Confidence level for intervals (default is 95%).

    Returns:
        tuple: DataFrame with likelihoods, confidence intervals, totals row, 
               and total expected spending.
    """
    # Calculate average likelihoods (mean)
    average_likelihoods = predicted_probs.mean(axis=0)
    
    # Calculate standard errors for predicted probabilities
    standard_errors = np.sqrt(average_likelihoods * (1 - average_likelihoods) / len(predicted_probs))
    
    # Critical Z-value for confidence interval
    z_value = norm.ppf(1 - (1 - ci_level / 100) / 2)
    
    # Calculate confidence intervals
    ci_lower = average_likelihoods - z_value * standard_errors
    ci_upper = average_likelihoods + z_value * standard_errors

    # Create DataFrame for results
    likelihoods_df = pd.DataFrame({
        'Category': categories,
        'Average Likelihood': average_likelihoods,
        'CI Lower': ci_lower,
        'CI Upper': ci_upper,
        'Midpoint': [mappings[dependent_var][cat] for cat in categories]
    })
    
    # Calculate total expected spending
    total_mean_spending = (likelihoods_df['Average Likelihood'] * likelihoods_df['Midpoint']).sum()
    total_ci_lower_spending = (likelihoods_df['CI Lower'] * likelihoods_df['Midpoint']).sum()
    total_ci_upper_spending = (likelihoods_df['CI Upper'] * likelihoods_df['Midpoint']).sum()
    
    # Append "Total Spending per Person" row
    totals_row = pd.DataFrame({
        'Category': ['Total Spending per Person'],
        'Average Likelihood': [total_mean_spending],
        'CI Lower': [total_ci_lower_spending],
        'CI Upper': [total_ci_upper_spending],
        'Midpoint': [np.nan]
    })
    likelihoods_df = pd.concat([likelihoods_df, totals_row], ignore_index=True)

    # Append "Seasonal Total" row if total_visitors is provided
    if total_visitors:
        seasonal_totals_row = pd.DataFrame({
            'Category': ['Seasonal Total'],
            'Average Likelihood': [total_mean_spending * total_visitors],
            'CI Lower': [total_ci_lower_spending * total_visitors],
            'CI Upper': [total_ci_upper_spending * total_visitors],
            'Midpoint': [np.nan]
        })
        likelihoods_df = pd.concat([likelihoods_df, seasonal_totals_row], ignore_index=True)
    
    return likelihoods_df, total_mean_spending



# Function for Saving Consolidated and Cleaned Results in an excel file
def save_consolidated_results_clean(results, export_path, file_name=f"{current_date}_regression_results.xlsx"):
    """
    Saves all regression results and likelihoods to a single cleaned Excel file.

    Parameters:
        results (dict): Dictionary containing model results, likelihoods, and names.
        export_path (str): Directory to save the file.
        file_name (str): Name of the output file.

    Returns:
        None
    """
    # Ensure the export path exists
    os.makedirs(export_path, exist_ok=True)

    # Clean data for Excel compatibility
    def clean_data(df):
        """Remove potential formula-like strings and strip whitespace."""
        return df.apply(lambda col: col.map(lambda x: str(x).replace("=", "").strip() if isinstance(x, str) else x))

    # Save to Excel
    full_file_path = os.path.join(export_path, file_name)
    with pd.ExcelWriter(full_file_path, engine="openpyxl") as writer:
        for model_name, data in results.items():
            # Clean and save result summary
            result_summary_df = pd.DataFrame({"Summary": data["result"].summary().as_text().split("\n")})
            result_summary_df = clean_data(result_summary_df)
            result_summary_df.to_excel(writer, sheet_name=f"{model_name}", index=False)

            # Clean and save likelihoods
            likelihoods_df = clean_data(data["likelihoods_df"])
            likelihoods_df.to_excel(writer, sheet_name=f"{model_name}_sum", index=False)

    
    print(f"All cleaned results have been saved successfully.")

#%% RUN ORDINAL LOGISTIC REGRESSION 
################################################################################################################################################################

# Run and Save All Models
# Updated workflow to run both ordinal and multinomial regressions
results = {}

# Run ordinal logistic regression
for model_name, config in models.items():
    dependent_var = config["dependent_var"]
    independent_vars = config["independent_vars"]

    print(f"Running ordinal logistic regression for {model_name}...")
    result_ord, predicted_probs_ord, categories_ord = run_ordinal_logit_regression(
        dependent_var, independent_vars, suf_data_filtered, model_name + "_ordinal"
    )
    likelihoods_df_ord, total_mean_spending_ord = analyze_results(
        predicted_probs_ord, categories_ord, mappings, dependent_var
    )
    print(f"Total average spending (ordinal) for {model_name}: {total_mean_spending_ord}")

    # Store ordinal regression results
    results[model_name + "_ord"] = {
        "result": result_ord,
        "likelihoods_df": likelihoods_df_ord
    }

# Run multinomial logistic regression
for model_name, config in models.items():
    dependent_var = config["dependent_var"]
    independent_vars = config["independent_vars"]

    print(f"Running multinomial logistic regression for {model_name}...")
    try:
        # Run multinomial regression
        result_multi, predicted_probs_multi, categories_multi = run_multinomial_logit_regression(
            dependent_var, independent_vars, suf_data_filtered, model_name + "_mult"
        )
        likelihoods_df_multi, total_mean_spending_multi = analyze_results(
            predicted_probs_multi, categories_multi, mappings, dependent_var
        )
        print(f"Total average spending (multinomial) for {model_name}: {total_mean_spending_multi}")

        # Store multinomial regression results only if successful
        results[model_name + "_mult"] = {
            "result": result_multi,
            "likelihoods_df": likelihoods_df_multi
        }
    except Exception as e:
        print(f"Multinomial logistic regression for {model_name} failed: {e}")

# Save all results to one consolidated file
save_consolidated_results_clean(results, export_path)


#%% COMPARE MODEL RESULTS
################################################################################################################################################################

def compare_multiple_regression_results(results, model_names, export_path=export_path, current_date=current_date):
    """
    Compare results of multiple ordinal and multinomial logistic regressions with interpretations.

    Parameters:
        results (dict): Dictionary containing regression results.
        model_names (list): List of base model names to compare.
        export_path (str): Path to save the consolidated results.
        current_date (str): Current date for file naming.

    Returns:
        dict: A dictionary containing coefficient comparison tables, model fit metrics, and interpretations.
    """
    all_comparisons = {}

    # Prepare the Excel output
    output_file = os.path.join(export_path, f"{current_date}_model_comparison.xlsx")
    os.makedirs(export_path, exist_ok=True)

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        for model_name in model_names:
            print(f"\nComparing regression results for {model_name}...")
            try:
                # Retrieve ordinal and multinomial results
                result_ord = results[f"{model_name}_ord"]["result"]
                result_multi = results[f"{model_name}_mult"]["result"]

                # Extract ordinal coefficients and filter out thresholds
                ord_coefs = result_ord.params.rename("Ordinal Coefficients")
                ord_coefs_filtered = ord_coefs[~ord_coefs.index.str.contains("/")]

                # Extract multinomial coefficients and pivot
                multi_coefs = result_multi.params.T  # Transpose for better alignment
                multi_coefs_flat = multi_coefs.stack().rename("Multinomial Coefficients").reset_index()
                multi_coefs_flat.columns = ["Category", "Predictor", "Multinomial Coefficients"]
                multi_coefs_pivot = multi_coefs_flat.pivot(index="Predictor", columns="Category", values="Multinomial Coefficients")

                # Combine ordinal and multinomial coefficients
                coef_comparison = pd.DataFrame(ord_coefs_filtered).rename_axis("Predictor")
                coef_comparison = coef_comparison.join(multi_coefs_pivot, how="outer")

                # Model fit metrics
                metrics = {
                    "Log-Likelihood": [result_ord.llf, result_multi.llf],
                    "AIC": [result_ord.aic, result_multi.aic],
                    "BIC": [result_ord.bic, None],  # BIC may not exist for multinomial
                    "Pseudo R-squared": [None, result_multi.prsquared],  # Only for multinomial
                    "Converged": [result_ord.mle_retvals['converged'], result_multi.mle_retvals['converged']]
                }
                metrics_df = pd.DataFrame(metrics, index=["Ordinal Model", "Multinomial Model"])

                # Save results for this model
                all_comparisons[model_name] = {"coef_comparison": coef_comparison, "metrics": metrics_df}

                # Write to Excel
                coef_comparison.to_excel(writer, sheet_name=f"{model_name}_coefs")
                metrics_df.to_excel(writer, sheet_name=f"{model_name}_metrics")

                # Interpret Results
                print("\n--- Coefficient Interpretation ---")
                for predictor in coef_comparison.index:
                    if predictor in ord_coefs_filtered.index:  # Only interpret predictors, not thresholds
                        ordinal_value = coef_comparison.at[predictor, "Ordinal Coefficients"]
                        multi_values = coef_comparison.loc[predictor].dropna().iloc[1:].values

                        if len(multi_values) > 0 and any(abs(ordinal_value - mv) > 0.05 for mv in multi_values):
                            print(f"Predictor '{predictor}': Coefficients vary significantly across multinomial thresholds.")
                        else:
                            print(f"Predictor '{predictor}': Coefficients are relatively consistent.")

                print("\n--- Model Fit Interpretation ---")
                print(f"Ordinal Model Log-Likelihood: {result_ord.llf:.3f}")
                print(f"Multinomial Model Log-Likelihood: {result_multi.llf:.3f}")
                print(f"AIC: Ordinal = {result_ord.aic:.3f}, Multinomial = {result_multi.aic:.3f}")
                if result_multi.prsquared is not None:
                    print(f"Pseudo R-squared (Multinomial): {result_multi.prsquared:.3f}")

                if result_ord.llf > result_multi.llf:
                    print("The ordinal model fits the data better based on Log-Likelihood.")
                else:
                    print("The multinomial model fits the data better based on Log-Likelihood.")

            except KeyError as e:
                print(f"Missing results for {model_name}: {e}")
            except Exception as e:
                print(f"Error while comparing {model_name}: {e}")

    print(f"\nAll comparisons saved successfully to: {output_file}")
    return all_comparisons



# List of base model names (without _ord or _mult)
model_names = ["anreise_kosten_alle", "übernachtung_kosten"]

# Run the comparison
model_names = list(models.keys())  # Extract the base model names
all_comparisons = compare_multiple_regression_results(results, model_names)
