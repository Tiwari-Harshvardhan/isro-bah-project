const InfoPanel = ({ summary, loading }) => {
  if (loading) {
    return <div className="card skeleton-card"><div className="skeleton-line" /><div className="skeleton-line short" /><div className="skeleton-line" /></div>
  }

  return (
    <div className="card info-card">
      <div className="card-title-row">
        <h3>Zone Snapshot</h3>
        <span>{summary?.zone || 'Select a zone'}</span>
      </div>
      <div className="stats-grid">
        <div><p>Wards</p><strong>{summary?.number_of_wards || 0}</strong></div>
        <div><p>Population</p><strong>{summary?.population?.toFixed?.(0) || 0}</strong></div>
        <div><p>Density</p><strong>{summary?.population_density?.toFixed?.(0) || 0}</strong></div>
        <div><p>Built-up %</p><strong>{summary?.built_up_percent?.toFixed?.(2) || 0}</strong></div>
        <div><p>NDVI</p><strong>{summary?.mean_ndvi?.toFixed?.(3) || 0}</strong></div>
        <div><p>Latest LST</p><strong>{summary?.historical_lst?.slice(-1)[0]?.toFixed?.(2) || 0}</strong></div>
      </div>
    </div>
  )
}

export default InfoPanel
