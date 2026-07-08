import { useEffect, useMemo, useState } from 'react'
import { MapContainer, TileLayer, GeoJSON } from 'react-leaflet'
import { fetchGeoJson, fetchZones } from '../services/api'

const zoneColors = ['#1f78b4', '#33a02c', '#ff7f00', '#6a3d9a', '#e31a1c', '#00acc1', '#c51b7d', '#8c564b']

const MapView = ({ onZoneSelect, selectedZone, onPredictionRequest }) => {
  const [zones, setZones] = useState([])
  const [geojson, setGeojson] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const loadMapData = async () => {
      try {
        const [zoneResponse, geojsonResponse] = await Promise.all([fetchZones(), fetchGeoJson()])
        setZones(zoneResponse.data)
        setGeojson(geojsonResponse.data)
      } catch (error) {
        console.error(error)
      } finally {
        setLoading(false)
      }
    }
    loadMapData()
  }, [])

  const geoJsonStyle = useMemo(() => (feature) => ({
    color: selectedZone === feature?.properties?.zone_name ? '#ffffff' : zoneColors[(feature?.properties?.zone_name?.length || 0) % zoneColors.length],
    weight: selectedZone === feature?.properties?.zone_name ? 3 : 1,
    fillOpacity: 0.35,
    fillColor: selectedZone === feature?.properties?.zone_name ? '#22d3ee' : zoneColors[(feature?.properties?.zone_name?.length || 0) % zoneColors.length],
  }), [selectedZone])

  const onEachFeature = (feature, layer) => {
    layer.on({
      mouseover: () => layer.setStyle({ weight: 3, color: '#ffffff' }),
      mouseout: () => layer.setStyle(geoJsonStyle(feature)),
      click: () => {
        const zoneName = feature.properties.zone_name
        onZoneSelect(zoneName)
        onPredictionRequest({ zone: zoneName, year: 2022, month: 12 })
      },
    })
    layer.bindPopup(feature.properties.zone_name || 'Zone')
  }

  const mapTitle = loading ? 'Loading zones…' : zones.length ? `${zones.length} zones` : 'Zone data unavailable'

  return (
    <div className="card map-card">
      <div className="card-title-row">
        <h3>Delhi Heat Map</h3>
        <span>{mapTitle}</span>
      </div>
      <MapContainer center={[28.6139, 77.2090]} zoom={10} className="leaflet-map">
        <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" attribution="&copy; OpenStreetMap contributors" />
        {geojson && <GeoJSON data={geojson} style={geoJsonStyle} onEachFeature={onEachFeature} />}
      </MapContainer>
    </div>
  )
}

export default MapView
