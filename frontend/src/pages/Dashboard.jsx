import { useEffect, useState } from 'react'
import Navbar from '../components/Navbar'
import MapView from '../components/MapView'
import InfoPanel from '../components/InfoPanel'
import PredictionCard from '../components/PredictionCard'
import { fetchZoneSummary, predictZone } from '../services/api'

const Dashboard = () => {
  const [selectedZone, setSelectedZone] = useState('Central-Zone')
  const [summary, setSummary] = useState(null)
  const [prediction, setPrediction] = useState(null)
  const [loadingSummary, setLoadingSummary] = useState(true)
  const [loadingPrediction, setLoadingPrediction] = useState(true)
  const [error, setError] = useState('')

  const loadZoneData = async (zoneName) => {
    if (!zoneName) return
    try {
      setLoadingSummary(true)
      setLoadingPrediction(true)
      setError('')
      const summaryRes = await fetchZoneSummary(zoneName)
      setSummary(summaryRes.data)
      const predictionRes = await predictZone({ zone: zoneName, year: 2022, month: 12 })
      setPrediction(predictionRes.data)
    } catch (err) {
      setError('Unable to load zone insights right now.')
    } finally {
      setLoadingSummary(false)
      setLoadingPrediction(false)
    }
  }

  useEffect(() => {
    loadZoneData(selectedZone)
  }, [selectedZone])

  return (
    <div className="dashboard-shell">
      <Navbar />
      {error && <div className="toast">{error}</div>}
      <div className="dashboard-grid">
        <div className="left-panel">
          <MapView onZoneSelect={setSelectedZone} selectedZone={selectedZone} onPredictionRequest={loadZoneData} />
          <InfoPanel summary={summary} loading={loadingSummary} />
        </div>
        <div className="right-panel">
          <PredictionCard prediction={prediction} loading={loadingPrediction} />
        </div>
      </div>
    </div>
  )
}

export default Dashboard
