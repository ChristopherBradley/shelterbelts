# +
from qgis.core import (
    QgsProject, QgsVectorLayer, QgsFeature, QgsGeometry, QgsPointXY, QgsLineString, QgsFields, QgsField, QgsWkbTypes
)
from qgis.PyQt.QtCore import QVariant

# Create a memory layer
layer = QgsVectorLayer("MultiLineString?crs=EPSG:4326", "MyMultiLine", "memory")
provider = layer.dataProvider()

# Optional: add a field
provider.addAttributes([QgsField("id", QVariant.Int)])
layer.updateFields()

# Define your coordinates (lon, lat or x, y)
coords1 = [(149.28268, -35.27635), (149.28316, -35.27666)]
coords2 = [(149.28239, -35.27684), (149.28285, -35.27719)]

# Create a multiline geometry
line1 = QgsLineString([QgsPointXY(*pt) for pt in coords1])
line2 = QgsLineString([QgsPoint(*pt) for pt in coords2])
multi_line = QgsGeometry.fromMultiPolylineXY([line1.points(), line2.points()])

# Create the feature and add it
feature = QgsFeature()
feature.setGeometry(multi_line)
feature.setAttributes([1])
provider.addFeatures([feature])
layer.updateExtents()

# Add to map
QgsProject.instance().addMapLayer(layer)

# -



# +
from qgis.core import (
    QgsProject, QgsVectorLayer, QgsFeature, QgsGeometry, QgsPoint, QgsLineString
)
layer = QgsVectorLayer("MultiLineString?crs=EPSG:4326", "Lines", "memory")
provider = layer.dataProvider()


# Define lines using QgsPoint
line1 = QgsLineString([
    QgsPoint(149.28268, -35.27635),  # top-left
    QgsPoint(149.28316, -35.27666),  # top-right
])
[(149.28239, -35.27684), (149.28285, -35.27719)]
line2 = QgsLineString([
    QgsPoint(149.28239, -35.27684),  # bottom-left
    QgsPoint(149.28285, -35.27719),  # bottom-right
])

# Combine into a MultiLineString geometry
multi_line = QgsGeometry.fromMultiPolyline([line1.points(), line2.points()])

# Create feature and add to the layer
feature = QgsFeature()
feature.setGeometry(multi_line)
provider.addFeatures([feature])
layer.updateExtents()

# Add to the map
QgsProject.instance().addMapLayer(layer)

