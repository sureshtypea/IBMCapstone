import folium
import pandas as pd
from folium.plugins import MarkerCluster, MousePosition
from folium.features import DivIcon
from math import radians, sin, cos, sqrt, atan2

def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6373.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

file_path = 'spacex_launch_geo.csv'
spacex_df = pd.read_csv(file_path)
launch_sites_df = spacex_df[['Launch Site', 'Lat', 'Long']].drop_duplicates()

nasa_coordinate = [29.559684888503615, -95.0830971930759]
site_map = folium.Map(location=nasa_coordinate, zoom_start=10)

for _, row in launch_sites_df.iterrows():
    coordinate = [row['Lat'], row['Long']]
    folium.Marker(
        location=coordinate,
        icon=DivIcon(
            icon_size=(20, 20),
            icon_anchor=(0, 0),
            html='<div style="font-size: 12; color:#d35400;"><b>%s</b></div>' % row['Launch Site']
        ),
        popup=f"Launch Site: {row['Launch Site']}"
    ).add_to(site_map)

marker_cluster = MarkerCluster()
spacex_df['marker_color'] = spacex_df['class'].apply(lambda x: 'green' if x == 1 else 'red')

for _, record in spacex_df.iterrows():
    coordinate = [record['Lat'], record['Long']]
    marker = folium.Marker(
        location=coordinate,
        icon=folium.Icon(color='white', icon_color=record['marker_color'], icon='info-sign'),
        popup=f"Launch Site: {record['Launch Site']}<br>Outcome: {'Success' if record['class'] == 1 else 'Failed'}"
    )
    marker_cluster.add_child(marker)

site_map.add_child(marker_cluster)

mouse_position = MousePosition(
    position='topright',
    separator=' Long: ',
    empty_string='NaN',
    lng_first=False,
    num_digits=20,
    prefix='Lat:',
    lat_formatter="function(num) {return L.Util.formatNum(num, 5);};",
    lng_formatter="function(num) {return L.Util.formatNum(num, 5);};",
)
site_map.add_child(mouse_position)

launch_site_coordinate = [28.562302, -80.577356]
point_of_interest = [28.56367, -80.57163]
distance_to_interest = calculate_distance(
    launch_site_coordinate[0], launch_site_coordinate[1],
    point_of_interest[0], point_of_interest[1]
)

folium.Marker(
    location=point_of_interest,
    icon=DivIcon(
        icon_size=(20, 20),
        icon_anchor=(0, 0),
        html='<div style="font-size: 12; color:#d35400;"><b>%s</b></div>' % "{:.2f} KM".format(distance_to_interest),
    ),
    popup="Point of Interest: Coastline"
).add_to(site_map)

folium.PolyLine(
    locations=[launch_site_coordinate, point_of_interest],
    weight=2,
    color='blue'
).add_to(site_map)

site_map.save('consolidated_space_launch_map.html')
