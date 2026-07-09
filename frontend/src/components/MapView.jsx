import { useEffect, useMemo, useState } from 'react'
import { MapContainer, TileLayer, GeoJSON, useMap } from 'react-leaflet'
import { fetchGeoJson, fetchZones } from '../services/api'

const FlyToZone = ({ bounds }) => {
  const map = useMap()

  useEffect(() => {
    if (bounds) {
      map.fitBounds(bounds, { padding: [20, 20], maxZoom: 12 })
    }
  }, [bounds, map])

  return null
}

const MapView = ({ onZoneSelect, selectedZone, onZonesLoaded }) => {
  const [zones, setZones] = useState([])
  const [geojson, setGeojson] = useState(null)
  const [loading, setLoading] = useState(true)
  const [searchText, setSearchText] = useState('')
  const [selectedBounds, setSelectedBounds] = useState(null)
  const [showSuggestions, setShowSuggestions] = useState(false)

  useEffect(() => {
    const loadMapData = async () => {
      try {
        const [zoneResponse, geojsonResponse] = await Promise.all([fetchZones(), fetchGeoJson()])
        const zoneNames = zoneResponse.data
        setZones(zoneNames)
        setGeojson(geojsonResponse.data)
        if (onZonesLoaded) {
          onZonesLoaded(zoneNames)
        }
      } catch (error) {
        console.error(error)
      } finally {
        setLoading(false)
      }
    }
    loadMapData()
  }, [onZonesLoaded])

  useEffect(() => {
    if (selectedZone) {
      setSearchText(selectedZone)
      // Highlight and zoom to zone from selectedZone update (e.g. from search)
      if (geojson?.features) {
        const feature = geojson.features.find(
          (f) => f.properties?.zone_name?.toLowerCase() === selectedZone.toLowerCase(),
        )
        if (feature) {
          const bounds = getBoundsFromGeometry(feature.geometry)
          if (bounds) {
            setSelectedBounds(bounds)
          }
        }
      }
    }
  }, [selectedZone, geojson])

  const getBoundsFromGeometry = (geometry) => {
    if (!geometry || !geometry.coordinates) {
      return null
    }

    const points = []
    const collect = (coords) => {
      if (!Array.isArray(coords)) return
      if (typeof coords[0] === 'number') {
        points.push(coords)
        return
      }
      coords.forEach(collect)
    }

    collect(geometry.coordinates)
    if (!points.length) return null

    const lats = points.map((coord) => coord[1])
    const lngs = points.map((coord) => coord[0])
    return [
      [Math.min(...lats), Math.min(...lngs)],
      [Math.max(...lats), Math.max(...lngs)],
    ]
  }

  const handleZoneSelect = (zoneName) => {
    if (!zoneName) return
    const foundZone = zones.find((name) => name?.toLowerCase() === zoneName?.toLowerCase())
    const normalized = foundZone || zoneName
    onZoneSelect(normalized)
  }

  const geoJsonStyle = useMemo(
    () =>
      (feature) => {
        const isSelected = selectedZone?.toLowerCase() === feature?.properties?.zone_name?.toLowerCase()
        return {
          color: '#ffffff', // white borders
          weight: isSelected ? 3.5 : 1.5,
          fillColor: isSelected ? '#2563eb' : '#93c5fd', // bright blue selected, light blue default
          fillOpacity: isSelected ? 0.7 : 0.35,
        }
      },
    [selectedZone],
  )

  const onEachFeature = (feature, layer) => {
    layer.on({
      mouseover: () => {
        const isSelected = selectedZone?.toLowerCase() === feature?.properties?.zone_name?.toLowerCase()
        layer.setStyle({
          weight: 3.5,
          color: '#ffffff',
          fillOpacity: 0.65,
          fillColor: isSelected ? '#2563eb' : '#60a5fa' // bright highlight
        })
      },
      mouseout: () => {
        layer.setStyle(geoJsonStyle(feature))
      },
      click: () => {
        const zoneName = feature.properties.zone_name
        setSelectedBounds(layer.getBounds())
        setSearchText(zoneName)
        handleZoneSelect(zoneName)
      },
    })
    layer.bindPopup(feature.properties.zone_name || 'Zone')
  }

  const filteredSuggestions = useMemo(() => {
    if (!searchText) return zones
    return zones.filter((zone) => zone.toLowerCase().includes(searchText.toLowerCase()))
  }, [zones, searchText])

  const mapTitle = loading ? 'Loading zones…' : zones.length ? `${zones.length} zones loaded` : 'Zone data unavailable'

  return (
    <div className="card map-card">
      <div className="card-title-row">
        <h3>Delhi Heat Map</h3>
        <span>{mapTitle}</span>
      </div>
      
      <div className="map-search-row" onFocus={() => setShowSuggestions(true)} onBlur={() => setTimeout(() => setShowSuggestions(false), 200)}>
        <input
          type="text"
          value={searchText}
          onChange={(event) => {
            setSearchText(event.target.value)
            setShowSuggestions(true)
          }}
          placeholder="Search Delhi zone by name..."
          className="search-input"
        />
        {showSuggestions && filteredSuggestions.length > 0 && (
          <ul className="autocomplete-suggestions">
            {filteredSuggestions.map((zoneName) => (
              <li
                key={zoneName}
                className="suggestion-item"
                onMouseDown={() => {
                  setSearchText(zoneName)
                  handleZoneSelect(zoneName)
                  setShowSuggestions(false)
                }}
              >
                {zoneName}
              </li>
            ))}
          </ul>
        )}
      </div>

      <MapContainer center={[28.6139, 77.209]} zoom={10} className="leaflet-map">
        <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" attribution="&copy; OpenStreetMap contributors" />
        {geojson && <GeoJSON data={geojson} style={geoJsonStyle} onEachFeature={onEachFeature} />}
        {selectedBounds && <FlyToZone bounds={selectedBounds} />}
      </MapContainer>
    </div>
  )
}

export default MapView
