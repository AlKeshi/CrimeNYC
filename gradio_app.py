"""
NYC Crime Prediction - Interactive Gradio Interface
Displays predicted crime counts on a map of NYC with hover functionality.

Usage:
    python gradio_app.py

Requirements:
    pip install gradio numpy geopandas shapely folium branca
"""

import gradio as gr
import numpy as np
import geopandas as gpd
import shapely
import folium
import branca.colormap as cm


def load_data_and_mask():
    """Load predictions, test data, and create the geographic mask."""
    # Load NYC borough shapefile
    shapefile_path = 'data/shapefile/nyc_boroughs.shp'
    nyc = gpd.read_file(shapefile_path)
    nyc = nyc.to_crs("EPSG:4326")
    nyc = nyc.dissolve()

    # Load predictions (shape: 358, 67, 50, 1)
    predictions = np.load('data/homo_convlstm.npy')

    # Load test data for comparison (shape: 365, 67, 50)
    test_data = np.load('data/test_data.npy')

    # Grid dimensions
    n_y_cells = 67
    n_x_cells = 50

    # Get bounds from shapefile
    xmin, ymin, xmax, ymax = nyc.total_bounds
    x_cell_size = (xmax - xmin) / n_x_cells
    y_cell_size = (ymax - ymin) / n_y_cells

    # Create mask
    mask = np.ones((n_y_cells, n_x_cells), dtype=bool)
    nyc_geom = nyc.geometry.values[0]

    x_arange = np.arange(xmin, xmax + x_cell_size, x_cell_size)
    y_arange = np.arange(ymin, ymax + y_cell_size, y_cell_size)

    for i, y0 in zip(range(n_y_cells - 1, -1, -1), y_arange):
        for j, x0 in zip(range(n_x_cells), x_arange):
            x1 = x0 + x_cell_size
            y1 = y0 + y_cell_size
            box = shapely.geometry.box(x0, y0, x1, y1)
            if not nyc_geom.intersects(box):
                mask[i, j] = False

    # Pre-compute cell bounds for all valid cells (optimization)
    cell_info = []
    for row in range(n_y_cells):
        for col in range(n_x_cells):
            if mask[row, col]:
                x0 = xmin + col * x_cell_size
                x1 = x0 + x_cell_size
                y0 = ymax - (row + 1) * y_cell_size
                y1 = y0 + y_cell_size
                cell_info.append({
                    'row': row,
                    'col': col,
                    'bounds': [[y0, x0], [y1, x1]]
                })

    return predictions, test_data, mask, xmin, ymin, xmax, ymax, x_cell_size, y_cell_size, nyc, cell_info


# Load data once at startup
print("Loading data and creating mask...")
(predictions, test_data, mask, xmin, ymin, xmax, ymax,
 x_cell_size, y_cell_size, nyc_gdf, cell_info) = load_data_and_mask()
print(f"Predictions shape: {predictions.shape}")
print(f"Test data shape: {test_data.shape}")
print(f"Valid NYC cells: {np.sum(mask)} / {mask.size}")

# Constants
N_Y_CELLS = 67
N_X_CELLS = 50
MAX_FORECAST_DAY = predictions.shape[0]  # 358 days

# Training date range (data starts from 2020-01-01)
TRAIN_START_DATE = "2020-01-01"
TRAIN_END_DATE = "2024-01-01"


def create_crime_map(forecast_day):
    """
    Create an interactive map of NYC showing predicted crimes for a given day.

    Args:
        forecast_day: Day number after last training day (1 to 358)

    Returns:
        tuple: (HTML string of the folium map, summary text)
    """
    # Validate input
    forecast_day = int(forecast_day)
    if forecast_day < 1:
        forecast_day = 1
    if forecast_day > MAX_FORECAST_DAY:
        forecast_day = MAX_FORECAST_DAY

    # Get prediction for this day (0-indexed)
    day_idx = forecast_day - 1
    day_predictions = predictions[day_idx]

    # Handle channel dimension if present
    if len(day_predictions.shape) == 3:
        day_predictions = day_predictions[:, :, 0]

    # Clip negative values to 0 (crimes can't be negative) and round to integers
    day_predictions = np.clip(day_predictions, 0, None)
    day_predictions = np.round(day_predictions).astype(int)

    # Apply mask - set non-NYC cells to 0
    day_predictions_masked = day_predictions * mask

    # Calculate total crimes for the city (sum of all integer crime counts)
    total_crimes = int(np.sum(day_predictions_masked))

    # Create base map centered on NYC
    nyc_center = [40.7128, -73.9560]
    m = folium.Map(
        location=nyc_center,
        zoom_start=11,
        tiles='CartoDB positron'
    )

    # Add NYC boundary
    folium.GeoJson(
        nyc_gdf,
        style_function=lambda x: {
            'fillColor': 'transparent',
            'color': 'black',
            'weight': 2,
            'fillOpacity': 0
        },
        name='NYC Boundary'
    ).add_to(m)

    # Get valid crime values for color scaling
    valid_values = day_predictions_masked[mask]
    vmax = max(valid_values.max(), 1) if len(valid_values) > 0 else 1

    # Create colormap (yellow to red gradient)
    colormap = cm.LinearColormap(
        colors=['#ffffcc', '#ffeda0', '#fed976', '#feb24c', '#fd8d3c', '#fc4e2a', '#e31a1c', '#bd0026', '#800026'],
        vmin=0,
        vmax=vmax,
        caption=f'Predicted Crime Count (Day {forecast_day})'
    )

    # Add grid cells using pre-computed cell info (optimized)
    for cell in cell_info:
        row, col = cell['row'], cell['col']
        crime_count = day_predictions_masked[row, col]
        bounds = cell['bounds']

        # Color based on crime count
        color = colormap(crime_count) if crime_count > 0 else '#ffffcc'
        fill_opacity = 0.6 if crime_count > 0 else 0.3

        # Create rectangle with tooltip showing crime count (integer)
        tooltip_text = f"<b>Grid Cell [{row}, {col}]</b><br>Predicted Crimes: <b>{crime_count}</b>"

        folium.Rectangle(
            bounds=bounds,
            color='gray',
            weight=0.5,
            fill=True,
            fill_color=color,
            fill_opacity=fill_opacity,
            tooltip=folium.Tooltip(tooltip_text, sticky=True)
        ).add_to(m)

    # Add colormap legend
    colormap.add_to(m)

    # Add info box with total crimes
    title_html = f'''
    <div style="position: fixed;
                top: 10px; left: 50px; width: 350px;
                background-color: white;
                border: 2px solid gray;
                border-radius: 5px;
                padding: 10px;
                z-index: 9999;
                font-family: Arial, sans-serif;">
        <h4 style="margin: 0 0 5px 0;">NYC Crime Prediction</h4>
        <p style="margin: 0;"><b>Forecast Day:</b> {forecast_day} (after training period)</p>
        <p style="margin: 0;"><b>Total Predicted Crimes:</b> {total_crimes}</p>
        <p style="margin: 5px 0 0 0; font-size: 12px; color: gray;">Hover over cells to see individual predictions</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))

    return m._repr_html_(), f"Day {forecast_day}: Total predicted crimes = {total_crimes}"


def predict_crimes(forecast_day):
    """Main function called by Gradio interface."""
    map_html, summary = create_crime_map(forecast_day)
    return map_html, summary


# Create Gradio interface
with gr.Blocks(title="NYC Crime Prediction Map") as demo:
    gr.Markdown(f"""
    # NYC Crime Prediction - Interactive Map

    This application displays predicted crime counts across New York City using a ConvLSTM model.

    **Training Period:** {TRAIN_START_DATE} to {TRAIN_END_DATE}

    **Instructions:**
    - Enter a forecast day (1-358) representing days after the last training day
    - The map will show predicted crime counts for each grid cell
    - Hover over any cell to see the predicted number of crimes
    - The total predicted crimes for the entire city is shown in the info box
    """)

    with gr.Row():
        with gr.Column(scale=1):
            forecast_input = gr.Slider(
                minimum=1,
                maximum=MAX_FORECAST_DAY,
                value=1,
                step=1,
                label="Forecast Day (days after last training day)"
            )
            predict_btn = gr.Button("Generate Prediction Map", variant="primary")
            summary_output = gr.Textbox(label="Summary", interactive=False)

    map_output = gr.HTML(label="Crime Prediction Map")

    # Connect the interface
    predict_btn.click(
        fn=predict_crimes,
        inputs=[forecast_input],
        outputs=[map_output, summary_output]
    )

    # Also update on slider change
    forecast_input.change(
        fn=predict_crimes,
        inputs=[forecast_input],
        outputs=[map_output, summary_output]
    )

    gr.Markdown(f"""
    ---
    **Notes:**
    - **Training period:** {TRAIN_START_DATE} to {TRAIN_END_DATE} (1462 days)
    - Grid dimensions: 67 × 50 cells covering NYC
    - Only cells within NYC boundaries are shown (ocean/outside areas are masked)
    - Predictions are based on a ConvLSTM model trained on historical NYPD crime data
    - The model uses a 7-day lookback period for temporal patterns
    """)


if __name__ == "__main__":
    demo.launch(share=False)
