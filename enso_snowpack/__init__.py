"""El Niño (ENSO) vs. Rocky Mountain snowpack — data pipeline and analysis.

Sources: NOAA CPC Oceanic Niño Index (ONI), NRCS SNOTEL / snow-course
measurements via the AWDB REST API, and NCEI nClimDiv statewide climate
series. See README.md for the workflow.
"""

__version__ = "0.1.0"

STATES = ["MT", "ID", "WY", "CO", "UT"]
STATE_NAMES = {"MT": "Montana", "ID": "Idaho", "WY": "Wyoming", "CO": "Colorado", "UT": "Utah"}
