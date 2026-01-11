import geopandas as gpd
import numpy as np
import shapely
import matplotlib.pyplot as plt
from matplotlib import cm

# Load NYC borough shapefile
path = '../data/shapefile/nyc_boroughs.shp'
nyc = gpd.read_file(path)
# Transform to WGS84 (EPSG:4326) to match crime data coordinates
nyc = nyc.to_crs("EPSG:4326")
nyc = nyc.dissolve()

# Load training and test data
train_data = np.load('../data/train_data.npy')
test_data = np.load('../data/test_data.npy')

# Get dimensions from data
n_y_cells, n_x_cells = train_data.shape[1], train_data.shape[2]

# Create mask using shapefile bounds (like Chicago version)
xmin, ymin, xmax, ymax = nyc.total_bounds
x_cell_size = (xmax - xmin) / n_x_cells
y_cell_size = (ymax - ymin) / n_y_cells
mask = np.ones((n_y_cells, n_x_cells))
nyc_geom = nyc.geometry.values[0]

x_arange = np.arange(xmin, xmax+x_cell_size, x_cell_size)
y_arange = np.arange(ymin, ymax+y_cell_size, y_cell_size)
for i, y0 in zip(range(n_y_cells-1, -1, -1), y_arange):
    for j, x0 in zip(range(n_x_cells), x_arange):
        x1 = x0 + x_cell_size
        y1 = y0 + y_cell_size
        box = shapely.geometry.box(x0, y0, x1, y1)
        if not nyc_geom.intersects(box):
            mask[i,j] = np.nan

# HA: global average
historical_average = np.mean(train_data[-365:], axis=0)

historical_average *= mask
plt.imshow(historical_average, vmax=2, cmap='jet')
plt.axis('off')
plt.show()

mse = np.nanmean(np.square(np.subtract(test_data, np.repeat([historical_average], len(test_data), axis=0))))
rmse = np.sqrt(mse)
print(f'HA global average - MSE: {mse:.4f}')
print(f'HA global average - RMSE: {rmse:.4f}')

# Save global average predictions
ha_global_predictions = np.repeat([historical_average], len(test_data), axis=0)
np.save('../data/ha_global_predictions.npy', ha_global_predictions)
print('Saved global average predictions to ../data/ha_global_predictions.npy')

# HA: weekday average
week_day_indices = np.array([[j for j in range(i, len(train_data), 7)] for i in range(0, 7, 1)], dtype=object)
historical_week_average = np.array([np.mean(train_data[weekday_mask][-52:], axis=0) for weekday_mask in week_day_indices])
weekday_mask = week_day_indices[0]
historical_week_average *= mask

start_test_weekday = np.argmax([np.max(indices) for indices in week_day_indices]) + 1
historical_week_average = np.roll(historical_week_average, 7-start_test_weekday, axis=0)
historical_week_average = np.repeat(historical_week_average, len(test_data)//7+1, axis=0)[:len(test_data)]

plt.imshow(historical_week_average[0], vmax=2, cmap='jet')
plt.axis('off')
plt.show()

mse = np.nanmean(np.square(np.subtract(test_data, historical_week_average)))
rmse = np.sqrt(mse)
print(f'HA weekday average - MSE: {mse:.4f}')
print(f'HA weekday average - RMSE: {rmse:.4f}')

# Save weekday average predictions
np.save('../data/ha_weekday_predictions.npy', historical_week_average)
print('Saved weekday average predictions to ../data/ha_weekday_predictions.npy')
