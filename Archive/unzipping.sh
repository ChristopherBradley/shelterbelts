# I just ran this directly in a bash terminal to unzip the NSW dems
for zip in /g/data/xe2/cb8590/NSW_5m_DEM_zips/*.zip; do
    unzip -j "$zip" '*.asc' '*.prj' -d /g/data/xe2/cb8590/NSW_5m_DEMs
done