import axios from 'axios'

const api = axios.create({
  baseURL: 'http://127.0.0.1:8000',
  timeout: 120000,
})

export const fetchZones = () => api.get('/zones')
export const fetchZoneSummary = (zone) => api.get(`/zone/${encodeURIComponent(zone)}`)
export const predictZone = (payload) => api.post('/predict', payload)
