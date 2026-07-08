const ZoneCharts = ({ history }) => {
  if (!history || history.length === 0) {
    return (
      <div className="card chart-card">
        <div className="card-title-row">
          <h3>Historical Trends</h3>
          <span>Awaiting zone data</span>
        </div>
        <p style={{ marginTop: '1rem', color: '#94a3b8' }}>Select a zone to view historical temperature, NDVI, and population trends.</p>
      </div>
    )
  }

  const latest = history[history.length - 1]
  const avgTemp = (history.reduce((sum, row) => sum + (row.mean_lst_day_celsius || 0), 0) / history.length).toFixed(2)

  return (
    <div className="card chart-card">
      <div className="card-title-row">
        <h3>Historical Trends</h3>
        <span>{history.length} records</span>
      </div>
      <div className="chart-summary">
        <div><p>Latest</p><strong>{latest.mean_lst_day_celsius?.toFixed(2) || '--'}°C</strong></div>
        <div><p>Average</p><strong>{avgTemp}°C</strong></div>
        <div><p>NDVI</p><strong>{latest.mean_ndvi?.toFixed(3) || '--'}</strong></div>
      </div>
      <div className="chart-placeholder">
        <p>Interactive charts coming soon.</p>
      </div>
    </div>
  )
}

export default ZoneCharts
