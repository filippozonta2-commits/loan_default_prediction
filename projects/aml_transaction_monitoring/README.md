# AML Transaction Monitoring

Scenario-based financial crime analytics project combining rules, alert generation, customer risk, network signals, and investigator-facing explainability.

## Initial scope

- Synthetic customers, accounts, and transactions
- Structuring and rapid-movement scenarios
- High-risk geography and sanctions signals
- Alert prioritization and case-level aggregation
- Precision, recall, alert volume, and scenario coverage
- Investigator dashboard

## Structure

- data/raw/ — synthetic source data
- data/processed/ — feature and alert tables
- notebooks/ — exploration and model experiments
- src/generation/ — synthetic data generation
- src/features/ — behavioral feature engineering
- src/scenarios/ — rule-based monitoring scenarios
- src/models/ — optional alert-prioritization models
- dashboard/ — investigator and portfolio views
- tests/ — scenario and pipeline tests
- docs/ — methodology, assumptions, and model risk notes
