"""El Niño (ENSO) vs. Rocky Mountain snowpack — data pipeline and analysis.

Sources: NOAA CPC Oceanic Niño Index (ONI), NRCS SNOTEL / snow-course
measurements via the AWDB REST API, and NCEI nClimDiv statewide climate
series. See README.md for the workflow.
"""

__version__ = "0.1.0"

# Every state with a SNOTEL network, north to south then east.
STATES = ["AK", "WA", "OR", "CA", "NV", "ID", "MT", "WY", "UT", "CO", "AZ", "NM", "SD"]
STATE_NAMES = {"AK": "Alaska", "WA": "Washington", "OR": "Oregon", "CA": "California",
               "NV": "Nevada", "ID": "Idaho", "MT": "Montana", "WY": "Wyoming",
               "UT": "Utah", "CO": "Colorado", "AZ": "Arizona", "NM": "New Mexico",
               "SD": "South Dakota"}
