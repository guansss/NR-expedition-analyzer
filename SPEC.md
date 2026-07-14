# Goal

Build a script that recognizes a game's match history from a series of screenshots and outputs a webpage that displays the match history along with relevant statistics and visualizations.

# Terminology

- **Expedition**: A single match.
- **Nightlord**: The final boss of an expedition.
- **Day**: Progress of the expedition, which can be Day 1, Day 2, or Day 3. The player can only achieve a victory after completing Day 3.

# Tech Stack

- **Script**: Python, OpenCV, PaddleOCR, YAML
- **Webpage**: Vanilla JavaScript, Bootstrap, ECharts

# Project Structure

```
assets/
  expedition-nightlords/    # Nightlord icons
  days/                     # Day icons
  template.html             # HTML template for the webpage
   results-schema.yaml       # Schema for the results.yaml file
data/
  expeditions/              # Screenshots of expeditions
  expeditions-recognized/   # Cropped images of recognized expeditions
scripts/
  recognize.py              # Script to recognize expeditions from screenshots
  generate_webpage.py       # Script to generate the webpage with expeditions
results.yaml              # Intermediate results containing recognized expedition information along with some customizable fields
index.html                  # Webpage to display expeditions and statistics
translations.yaml           # Translations for the webpage
```

# Usage

1. Place screenshots of expeditions in the `data/expeditions/` directory.
2. Run the `recognize.py` script to process the screenshots and extract expedition information into the `results.yaml` file.
3. Edit the `results.yaml` file to customize the expedition information if needed.
4. Run the `generate_webpage.py` script to create the `index.html` webpage.

# Implementation Details

## `results.yaml`

The `results.yaml` file contains a list of recognized expeditions. The schema is in `assets/results-schema.yaml`.

User can edit the `results.yaml` file to customize the expedition information. When the recognition script is run, it will update the `results.yaml` file with the recognized information while preserving the user-customized fields.

## Webpage features

#### Single File

The webpage is a single html file with all js and css embedded. External libraries are loaded from CDNs.

#### Translations

The translations are stored in the `translations.yaml` file. When generating the webpage, the translations will be converted into JSON and embedded into the webpage. The webpage will use the translations when displaying text.

#### Statistics

The webpage will display the following statistics on the top:

- Total number of expeditions
- Total number of victories (day3-victory)
- A line chart showing the points trend over time, with the x-axis representing expedition order and the y-axis representing the points. The expedition details will be displayed in a tooltip when hovering over the points on the chart.
- A bar chart showing the number of expeditions for each Nightlord, with the x-axis representing the Nightlord names and the y-axis representing the number of expeditions. A nightlord with name "X" and "X-everdark" will be counted as the same nightlord. A bar will be displayed in two colors, with a darker color at the bottom representing the everdark variant, and a lighter color at the top representing the non-everdark variant.

If an expedition's `excluded` has value true, the expedition will not be counted in the statistics.

#### Expeditions Table

Below the statistics is a table that lists all expeditions with all fields (except `excluded`) from the `results.yaml` file.

If `nightlord` is everdark variant (indicated by the suffix `-everdark`), the nightlord name will be highlighted with a purple background.

If `day` has value `day3-victory`, the row will be highlighted with a light green background.

If `reason` has value `bug`, the row will have a lower opacity.

If `excluded` has value true, the row will not be displayed in the table.

The `image` field displays an icon, and hovering over the icon will show the full image in a popup.

## Recognizing Expeditions

Each screenshot has a region that contains 8 rows, with each row representing an expedition in the following format:

```
Nightlord (icon) | Day (icon) | Points (number) | Date (two-line text)
```

The Date has a format of `YYYY/MM/DD` on the first line and `HH:MM:SS` on the second line. After recognizing the Date, the two lines will be joined into a single string.

To recognize the expeditions, the `recognize.py` script will:

1. Set up an image recognition pipeline.
2. Create a list to store the recognized expeditions as dictionaries with the following fields:
   - `nightlord` (string)
   - `day` (string)
   - `points` (integer)
   - `date` (string)
   - `image` (string)
   - `excluded` (boolean)
   - `reason` (string)
   - `note` (string)
3. Clean up the `data/expeditions-recognized/` directory.
4. Load each screenshot from the `data/expeditions/` directory.
5. Starting from 7.8% of the screenshot width and 21.5% of the screenshot height, recognize all 8 Nightlords using the Nightlord icons in the `assets/expedition-nightlords/` directory. If less than 8 Nightlords are recognized, log a warning and continue.
6. For each Nightlord:
   1. Define a row region as:
      - Top: Nightlord's bounding box top
      - Bottom: Nightlord's bounding box bottom
      - Left: Nightlord's bounding box left minus 2.7% of the screenshot width
      - Right: Left plus 57% of the screenshot width
   2. Crop the row region for further processing.
   3. Starting from the left of the row region plus 87% of the row width, crop a region to the right edge of the row to recognize the date using OCR. If not recognized, log a warning and skip the current expedition.
   4. Check if the expeditions list already contains an expedition with the same date. If it does, log a message indicating that the expedition is a duplicate and skip the current expedition.
   5. Starting from the right of the Nightlord icon, recognize the Day using the Day icons in the `assets/days/` directory. If not recognized, use a value `day0`, log a warning and continue.
   6. Starting from the left of the row region plus 77% of the row width, crop a region of 7% of the row width to recognize the Points using OCR. If not recognized, use a value `-1`.
   7. Save the cropped row image to the `data/expeditions-recognized/` directory. The filename should be in the format `<screenshot_name>_<row_index>.jpg`. Store the filename in the `image` field of the expedition dictionary.
   8. Store the recognized information as a dictionary in the expeditions list.
   9. Log the recognized information in a single line.
7. After processing all screenshots, sort the expeditions list by date in descending order.
8. Validate the points:
   1. In the expedition list, find the first expedition where the `points` is not `-1`.
   2. If found, find the next expedition where the `points` is not `-1`.
   3. If the previous expedition's `day` is `day3-victory` but its `points` is not greater than the next expedition's `points`, or if the previous expedition's `day` is not `day3-victory` but its `points` is greater than the next expedition's `points`, log a warning indicating that the points are abnormal on the previous expedition.
   4. Repeat from step 2 until the end of the expedition list.
9. Read the `results.yaml` file. For each expedition in the list, find the corresponding entry in the `results.yaml` file based on the date. If found, copy the `excluded`, `reason` and `note` fields from the entry to the dictionary. Keep track of the expeditions in the `results.yaml` file that have been matched. For any expedition in the `results.yaml` file that has not been matched, log a warning indicating that the customized information is discarded.
10. After processing all expeditions, write the expeditions list to the `results.yaml` file.

## Generating the Webpage

The `generate_webpage.py` script will:

1. Read the `results.yaml` file and parse the expedition information.
2. Read the `translations.yaml` file and parse the translations.
3. Render the `assets/template.html` file with the expedition information and translations, and write the output to `index.html`.
