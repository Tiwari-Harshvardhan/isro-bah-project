import { useState } from 'react'

const BudgetPlanner = ({ budget, onAllocate }) => {
  const [amount, setAmount] = useState(budget || '')

  return (
    <div className="card budget-card">
      <div className="card-title-row">
        <h3>Budget Planner</h3>
        <span>Allocate funds across zones</span>
      </div>
      <div className="budget-input-row">
        <input
          type="text"
          value={amount}
          onChange={(event) => setAmount(event.target.value)}
          placeholder="₹5 Crore"
        />
        <button className="primary-button" onClick={() => onAllocate(amount)}>
          Allocate
        </button>
      </div>
      <div className="budget-summary">
        {budget ? <p>Proposed allocation for ₹{budget}</p> : <p>Enter a budget to build a plan.</p>}
      </div>
    </div>
  )
}

export default BudgetPlanner
