import { useEffect, useState } from 'react'
import Navbar from '../components/Navbar'
import MapView from '../components/MapView'
import PredictionCard from '../components/PredictionCard'
import ExplanationPanel from '../components/ExplanationPanel'
import AIChatPanel from '../components/AIChatPanel'
import BudgetPlanner from '../components/BudgetPlanner'
import ZoneCharts from '../components/ZoneCharts'
import { fetchZoneHistory, fetchZoneSummary, generateReport, explainPrediction, planBudget, predictZone, fetchAssistant } from '../services/api'

const Dashboard = () => {
  const [selectedZone, setSelectedZone] = useState('')
  const [zones, setZones] = useState([])
  const [summary, setSummary] = useState(null)
  const [prediction, setPrediction] = useState(null)
  const [analysis, setAnalysis] = useState(null)
  const [history, setHistory] = useState([])
  const [loadingSummary, setLoadingSummary] = useState(false)
  const [loadingPrediction, setLoadingPrediction] = useState(false)
  const [loadingAnalysis, setLoadingAnalysis] = useState(false)
  const [loadingReport, setLoadingReport] = useState(false)
  const [budgetResult, setBudgetResult] = useState(null)
  const [error, setError] = useState('')
  const [chatResponse, setChatResponse] = useState('')
  const [selectedYear] = useState(2025)
  const [selectedMonth] = useState(12)

  const loadZoneData = async (zoneName) => {
    if (!zoneName) return
    try {
      setLoadingSummary(true)
      setLoadingPrediction(true)
      setLoadingAnalysis(true)
      setError('')

      const [summaryRes, predictionRes, analysisRes, historyRes] = await Promise.all([
        fetchZoneSummary(zoneName),
        predictZone({ zone: zoneName, year: selectedYear, month: selectedMonth }),
        explainPrediction({ zone: zoneName, year: selectedYear, month: selectedMonth }),
        fetchZoneHistory(zoneName),
      ])

      setSummary(summaryRes.data)
      setPrediction(predictionRes.data)
      setAnalysis(analysisRes.data)
      setHistory(historyRes.data)
    } catch (err) {
      setError('Unable to load zone insights right now.')
    } finally {
      setLoadingSummary(false)
      setLoadingPrediction(false)
      setLoadingAnalysis(false)
    }
  }

  useEffect(() => {
    if (selectedZone) {
      loadZoneData(selectedZone)
    }
  }, [selectedZone])

  const handleZonesLoaded = (loadedZones) => {
    setZones(loadedZones)
  }

  const handleChatPrompt = async (query) => {
    if (!selectedZone) return 'Please select a Delhi zone first.'
    try {
      const response = await fetchAssistant({
        zone: selectedZone,
        prediction: prediction,
        historical_data: history,
        zone_statistics: summary,
        question: query
      })
      setChatResponse(response.data.answer)
      return response.data.answer
    } catch (err) {
      setError('AI assistant is unavailable.')
      return 'Sorry, the assistant cannot answer right now.'
    }
  }

  const handleBudgetAllocation = async (budget) => {
    try {
      const response = await planBudget({ budget, year: selectedYear, month: selectedMonth })
      setBudgetResult(response.data)
      
      // Update assistant context if possible or let the user query it
      // Let's print a toast or update chat response as requested in Point 10
      const X = (response.data.estimated_population_benefited / 100000).toFixed(1)
      setChatResponse(`This allocation cools approximately ${X} lakh residents while maximizing heat reduction.`)
    } catch (err) {
      setError('Budget planning failed.')
    }
  }

  const handleGenerateReport = async () => {
    try {
      setLoadingReport(true)
      const response = await generateReport({ zone: selectedZone, year: selectedYear, month: selectedMonth, budget: budgetResult?.budget || '' })
      const blob = new Blob([response.data], { type: 'application/pdf' })
      const url = window.URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = `${selectedZone.replace(/\s+/g, '_')}_report.pdf`
      document.body.appendChild(link)
      link.click()
      link.remove()
      window.URL.revokeObjectURL(url)
    } catch (err) {
      setError('Report generation failed.')
    } finally {
      setLoadingReport(false)
    }
  }

  return (
    <div className="dashboard-shell">
      <Navbar />
      {error && <div className="toast">{error}</div>}
      <div className="dashboard-grid">
        <div className="left-panel">
          <MapView
            onZoneSelect={setSelectedZone}
            selectedZone={selectedZone}
            onZonesLoaded={handleZonesLoaded}
          />
          <ZoneCharts history={history} />
        </div>
        <div className="right-panel">
          <PredictionCard prediction={prediction} loading={loadingPrediction} selectedZone={selectedZone} />
          <ExplanationPanel analysis={analysis} loading={loadingAnalysis} selectedZone={selectedZone} />
          <AIChatPanel analysis={analysis} onPrompt={handleChatPrompt} selectedZone={selectedZone} defaultResponse={chatResponse} />
          <BudgetPlanner budget={budgetResult?.budget} onAllocate={handleBudgetAllocation} budgetResult={budgetResult} />
          <button className="primary-button report-button" onClick={handleGenerateReport} disabled={loadingReport || !selectedZone}>
            {loadingReport ? 'Generating report…' : 'Generate Report'}
          </button>
        </div>
      </div>
    </div>
  )
}

export default Dashboard
