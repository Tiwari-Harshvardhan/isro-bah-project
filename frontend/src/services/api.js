import axios from 'axios'

const api = axios.create({
  baseURL: 'http://127.0.0.1:8000',
  timeout: 120000,
})

export const fetchZones = () => api.get('/zones')
export const fetchZoneSummary = (zone) => api.get(`/zone/${encodeURIComponent(zone)}`)
export const fetchZoneHistory = (zone) => api.get(`/zone/${encodeURIComponent(zone)}/history`)
export const predictZone = (payload) => api.post('/predict', payload)
export const fetchGeoJson = () => api.get('/geo/geojson')
export const explainPrediction = (payload) => api.post('/analysis/explain', payload)
export const chatQuery = (payload) => api.post('/analysis/chat', payload)
export const planBudget = (payload) => api.post('/budget/plan', payload)
export const generateReport = (payload) => api.post('/report/generate', payload, { responseType: 'blob' })
export const fetchAssistant = (payload) => api.post('/assistant', payload)
