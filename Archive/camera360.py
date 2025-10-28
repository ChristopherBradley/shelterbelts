# +
# # !pip install gpxpy
# # !pip install pymediainfo
# -

import geopandas as gpd

gdf_watch = gpd.read_file('/Users/christopherbradley/Desktop/Painter walking track 2FPS/painter_streetview_capture_with_gps_device.gpx', layer='track_points')
gdf_section = gdf_watch[(gdf_watch['time'] > '2025-10-21 02:22:00+00:00') & (gdf_watch['time'] < '2025-10-21 02:34:00+00:00')]
# gdf_section.to_file('/Users/christopherbradley/Desktop/Painter walking track 2FPS/strava_cropped.gpx')

# +
# This worked, but apparently all the timepoints were the same in streetview studio
import gpxpy
import gpxpy.gpx
from shapely.geometry import LineString

# --- Crop as you did before ---
gdf_watch = gpd.read_file(
    '/Users/christopherbradley/Desktop/Painter walking track 2FPS/painter_streetview_capture_with_gps_device.gpx',
    layer='track_points'
)
gdf_section = gdf_watch[
    (gdf_watch['time'] > '2025-10-21 02:22:00+00:00') &
    (gdf_watch['time'] < '2025-10-21 02:34:00+00:00')
].sort_values('time')

# --- Create GPX structure ---
gpx = gpxpy.gpx.GPX()
track = gpxpy.gpx.GPXTrack()
gpx.tracks.append(track)
segment = gpxpy.gpx.GPXTrackSegment()
track.segments.append(segment)

# --- Add each track point with time (and optional elevation) ---
for _, row in gdf_section.iterrows():
    point = row.geometry
    time = row['time'].to_pydatetime()  # ensure it's a datetime object
    ele = row.get('ele', None)  # some GPX files have an elevation column
    segment.points.append(
        gpxpy.gpx.GPXTrackPoint(
            latitude=point.y,
            longitude=point.x,
            elevation=ele,
            time=time
        )
    )

# --- Write to file ---
output_path = '/Users/christopherbradley/Desktop/Painter walking track 2FPS/strava_cropped_for_streetview.gpx'
with open(output_path, 'w') as f:
    f.write(gpx.to_xml())

print(f"✅ Saved valid GPX: {output_path}")

# -
# Adding a creation time to the video
# !ffmpeg -i output_2fps.mp4 -metadata creation_time="2025-10-21T02:22:01Z" -codec copy output_2fps_timestamped.mp4

# +
# Checking the mp4 timestamp
from pymediainfo import MediaInfo

video_path = "/Users/christopherbradley/Desktop/Painter walking track 2FPS/output_2fps_timestamped.mp4"
media_info = MediaInfo.parse(video_path)

for track in media_info.tracks:
    if track.track_type == "General":
        print("Recorded date:", track.recorded_date)
        print("Tagged date:", track.tagged_date)
        print("Encoded date:", track.encoded_date)


# +
# Slowing down the video
ffmpeg -i input.mp4 -filter:v "setpts=14.985*PTS" -r 2 output_2fps.mp4

# Recreating the video as a slower speed
ffmpeg -framerate 2 -pattern_type glob -i '*.jpg' -c:v libx264 -pix_fmt yuv420p output_2fps.mp4

