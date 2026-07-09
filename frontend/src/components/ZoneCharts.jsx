import { useMemo, useState } from 'react'
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts'

const ZoneCharts = ({ history }) => {
  const [activeTab, setActiveTab] = useState('lst')

  const chartData = useMemo(() => {
    if (!history || history.length === 0) return []
    return history.map((row) => ({
      name: `${row.month}/${row.year}`,
      lst: row.mean_lst_day_celsius ? parseFloat(row.mean_lst_day_celsius.toFixed(2)) : null,
      ndvi: row.mean_ndvi ? parseFloat(row.mean_ndvi.toFixed(3)) : null,
      population: row.population ? parseInt(row.population) : null,
    }))
  }, [history])

  if (!history || history.length === 0) {
    return (
      <div className="card chart-card placeholder-card">
        <div className="card-title-row">
          <h3>Historical Trends</h3>
        </div>
        <p className="placeholder-text">Select a Delhi zone to view historical temperature, NDVI, and population trends.</p>
      </div>
    )
  }

  const latest = history[history.length - 1]
  const avgTemp = (history.reduce((sum, row) => sum + (row.mean_lst_day_celsius || 0), 0) / history.length).toFixed(2)

  // Configure parameters based on tab selection
  const tabConfig = {
    lst: {
      dataKey: 'lst',
      color: '#f87171',
      label: 'Temperature (°C)',
      domain: ['auto', 'auto'],
    },
    ndvi: {
      dataKey: 'ndvi',
      color: '#4ade80',
      label: 'NDVI Index',
      domain: ['auto', 'auto'],
    },
    population: {
      dataKey: 'population',
      color: '#60a5fa',
      label: 'Population',
      domain: ['auto', 'auto'],
    },
  }

  const currentTab = tabConfig[activeTab]

  return (
    <div className="card chart-card">
      <div className="card-title-row">
        <h3>Historical Trends</h3>
        <span className="chart-records-count">{history.length} records</span>
      </div>

      <div className="chart-summary">
        <div>
          <p>Latest LST</p>
          <strong>{latest.mean_lst_day_celsius?.toFixed(2) || '--'}°C</strong>
        </div>
        <div>
          <p>Average LST</p>
          <strong>{avgTemp}°C</strong>
        </div>
        <div>
          <p>NDVI</p>
          <strong>{latest.mean_ndvi?.toFixed(3) || '--'}</strong>
        </div>
      </div>

      {/* Tabs Row */}
      <div className="chart-tabs-row">
        <button
          className={`tab-button ${activeTab === 'lst' ? 'active' : ''}`}
          onClick={() => setActiveTab('lst')}
        >
          LST Temp
        </button>
        <button
          className={`tab-button ${activeTab === 'ndvi' ? 'active' : ''}`}
          onClick={() => setActiveTab('ndvi')}
        >
          NDVI Cover
        </button>
        <button
          className={`tab-button ${activeTab === 'population' ? 'active' : ''}`}
          onClick={() => setActiveTab('population')}
        >
          Population
        </button>
      </div>

      {/* Recharts Render Container */}
      <div className="chart-recharts-container" style={{ width: '100%', height: 260, marginTop: '1rem' }}>
        {chartData.length > 0 ? (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 5 }}>
              <CartesianGrid stroke="rgba(255, 255, 255, 0.05)" strokeDasharray="3 3" />
              <XAxis
                dataKey="name"
                stroke="#8dacd3"
                tick={{ fill: '#8dacd3', fontSize: 10 }}
                tickLine={false}
              />
              <YAxis
                stroke="#8dacd3"
                domain={currentTab.domain}
                tick={{ fill: '#8dacd3', fontSize: 10 }}
                tickLine={false}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#0c1626',
                  borderColor: 'rgba(255, 255, 255, 0.1)',
                  borderRadius: 12,
                  color: '#ecf7ff',
                }}
                labelStyle={{ fontWeight: 600, marginBottom: 4 }}
              />
              <Line
                type="monotone"
                dataKey={currentTab.dataKey}
                stroke={currentTab.color}
                strokeWidth={2.5}
                dot={{ r: 2, fill: currentTab.color }}
                activeDot={{ r: 6 }}
              />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <div className="chart-placeholder">
            <p>Data unavailable for rendering charts.</p>
          </div>
        )}
      </div>
    </div>
  )
}

export default ZoneCharts
