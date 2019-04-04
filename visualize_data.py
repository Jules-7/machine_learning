"""This code visualizes the first observations and AOR entries
allowing detecting the outliers."""

import os
import pandas as pd
import matplotlib.pyplot as plt
from CoorConv import CoorConv

interim_data_folder = 'interim_data/1_observ'
data_files = os.listdir(interim_data_folder)

for data_file in data_files:
    data = pd.read_pickle(interim_data_folder + '/' + data_file)

    start_date, end_date = data_file.split('_')[3], data_file.split('_')[4][:-4]

    # class for coordinates conversion
    conv_coord = CoorConv(51, 8)  # MUAC airspace center: 51 deg lat, 8 deg long

    # extract AOR boundaries
    aor = pd.read_csv('resources/aor')
    aor[['x', 'y']] = aor.apply(lambda row: pd.Series(conv_coord.lalo_to_xy(row['lat'], row['lon'])), axis=1)

    min_x1, max_x1 = data['X1'].min(), data['X1'].max()
    min_y1, max_y1 = data['Y1'].min(), data['Y1'].max()

    min_xe, max_xe = data['ENTRY_X'].min(), data['ENTRY_X'].max()
    min_ye, max_ye = data['ENTRY_Y'].min(), data['ENTRY_Y'].max()

    min_xaor, max_xaor = aor['x'].min(), aor['x'].max()
    min_yaor, max_yaor = aor['y'].min(), aor['y'].max()

    min_x = min([min_x1, min_xe, min_xaor]) - 50
    max_x = max([max_x1, max_xe, max_xaor]) + 50

    min_y = min([min_y1, min_ye, min_yaor]) - 50
    max_y = max([max_y1, max_ye, max_yaor]) + 50

    ax = data.plot(kind='scatter', x='X1', y='Y1', alpha=0.4, c='b', label='1st observ')
    data.plot(kind='scatter', x='ENTRY_X', y='ENTRY_Y', alpha=0.4, c='r', ax=ax, label='ENTRY')
    aor.plot(x='x', y='y', c='k', label='AOR', ax=ax)

    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

    plt.title("%s_%s"%(start_date, end_date))

plt.show()