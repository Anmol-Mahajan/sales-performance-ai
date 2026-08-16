# Sales Performance AI — Implementation Skills

## 1. Project Objective

Build a production-ready Sales Performance Intelligence platform for a Managed Service Provider (MSP) using historical sales, customer, product, activity, opportunity, revenue, and salesperson synergy data.

The application must:

1. Ingest structured Excel files during the demo phase.
2. Validate and standardise the incoming data.
3. Build a reusable feature-engineering pipeline.
4. Compare multiple ML algorithms rather than assuming one model is best.
5. Use time-based validation to avoid data leakage.
6. Select and register the best-performing model using objective business and ML metrics.
7. Explain predictions using feature importance/SHAP.
8. Identify salesperson performance gaps against expected performance and high-performing cohorts.
9. Identify customer whitespace and cross-sell opportunities.
10. Measure internal salesperson synergy and referral effectiveness.
11. Expose predictions through an API/model-serving layer.
12. Present the results through a Plotly Dash application.
13. Be deployable from a local M1 MacBook to Azure.

The system should be treated as a decision-support system, not an autonomous decision-maker. Predictions and recommendations should be presented with appropriate uncertainty and explanation.

---

# 2. Business Domain

The organisation is a Managed Service Provider offering:

## Managed Services

- Cloud IT
- Modern Workplace
- Cyber Security

## Telecom

- Connectivity
- Hosted Telephony

## Support

- IT Support
- Telco Support

Products are related and should not be analysed independently.

Examples:

- Connectivity → Hosted Telephony
- Connectivity → Telco Support
- Cloud IT → Cyber Security
- Cloud IT → Modern Workplace
- Modern Workplace → IT Support
- Cyber Security → Cloud IT
- Existing customers → Cross-sell/upsell opportunities

A salesperson can discover a lead outside their primary speciality and pass it to another salesperson. This internal referral/synergy behaviour must be represented in the data and features.

---

# 3. Recommended Technology Stack

## Development

- Python 3.12+
- VS Code
- Git
- uv for Python environment/package management

## Data

- Pandas
- DuckDB
- OpenPyXL
- SQLAlchemy where database connectivity is required

## Machine Learning

Primary candidates:

- Linear/Elastic Net baseline
- Random Forest
- XGBoost
- LightGBM
- CatBoost

Do not assume XGBoost is automatically the best model.

## Explainability

- SHAP
- permutation importance
- native model feature importance where appropriate

## Experiment Tracking

- MLflow

## Dashboard

- Plotly Dash
- Plotly
- Dash Bootstrap Components if useful

## API

Preferred:

- FastAPI

Alternative:

- MLflow model serving

## Deployment

Development:

- Local Mac

Production:

- Azure App Service
- Azure Container Apps
- Azure SQL / existing SQL Server infrastructure

Containerisation:

- Docker, only when useful/required

---

# 4. Hardware Constraints

Development machine:

- MacBook Pro M1
- 16 GB RAM
- 256 GB SSD

Design accordingly.

Avoid:

- large local LLMs
- unnecessary Docker images
- heavyweight distributed ML frameworks
- excessive local databases
- unnecessarily large model artifacts

Prefer:

- classical/tabular ML
- XGBoost/LightGBM/CatBoost
- Pandas/Polars/DuckDB
- local MLflow
- lightweight Dash development

The sales-performance use case is primarily tabular ML and does not require a dedicated GPU.

---

# 5. Repository Structure

Use:

```text
sales-performance-ai/
│
├── data/
│   ├── raw/
│   ├── validated/
│   └── processed/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_comparison.ipynb
│   └── 04_model_explainability.ipynb
│
├── src/
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── excel_reader.py
│   │   └── validation.py
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── sales_features.py
│   │   ├── customer_features.py
│   │   ├── opportunity_features.py
│   │   ├── synergy_features.py
│   │   └── whitespace.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline.py
│   │   ├── regression.py
│   │   ├── classification.py
│   │   ├── model_selection.py
│   │   └── evaluation.py
│   │
│   ├── explainability/
│   │   ├── __init__.py
│   │   └── shap_analysis.py
│   │
│   ├── predictions/
│   │   ├── __init__.py
│   │   └── predict.py
│   │
│   └── config.py
│
├── dash_app/
│   ├── __init__.py
│   ├── app.py
│   ├── layouts/
│   │   ├── overview.py
│   │   ├── salesperson.py
│   │   ├── drivers.py
│   │   ├── whitespace.py
│   │   └── synergy.py
│   ├── callbacks/
│   │   ├── overview_callbacks.py
│   │   ├── salesperson_callbacks.py
│   │   ├── driver_callbacks.py
│   │   ├── whitespace_callbacks.py
│   │   └── synergy_callbacks.py
│   └── components/
│
├── models/
│   ├── candidate/
│   └── production/
│
├── tests/
│
├── config/
│   └── config.yaml
│
├── scripts/
│   ├── validate_data.py
│   ├── train.py
│   ├── evaluate.py
│   └── run_pipeline.py
│
├── requirements.txt
├── pyproject.toml
├── Dockerfile
├── .gitignore
└── README.md
```

---

# 6. Input Data Contract

The demo initially uses Excel files.

Required logical entities:

## Salespeople

Fields:

- SalespersonID
- Salesperson
- Segment
- PrimarySpecialism
- Region
- Seniority

## Customers

Fields:

- CustomerID
- CustomerName
- Segment
- Region
- CustomerSince
- ExistingCustomer

## CustomerProducts

Fields:

- CustomerID
- Product
- MonthlyRevenue
- Active

## Activities

Fields:

- SalespersonID
- ActivityDate
- ActivityType
- ActivityPurpose
- DurationMinutes
- CustomerID where available

Activity types may include:

- Reachout
- Meeting
- Call
- Email
- Proposal
- AccountReview

Activity purposes may include:

- New Business
- Cross-sell
- Upsell
- Retention

## Opportunities

Fields:

- OpportunityID
- CustomerID
- SalespersonID
- CreatedDate
- CloseDate
- Product
- OpportunityType
- Stage
- PipelineValue
- ExpectedGrossProfit
- SalesCycleDays

Opportunity types:

- New Customer
- Cross-sell
- Upsell
- Renewal

Stages:

- Open
- Won
- Lost

## MonthlyPerformance

Fields:

- Month
- SalespersonID
- CustomerReachouts
- Meetings
- OpportunitiesCreated
- OpportunitiesWon
- NewCustomers
- CrossSellOpportunities
- Revenue
- GrossProfit
- CrossSellRevenue
- TargetAttainment
- RetentionRate
- WinRate

## SynergyReferrals

Fields:

- ReferralID
- ReferralDate
- FromSalespersonID
- ToSalespersonID
- ProductArea
- ReferralType
- ReferralStatus

Referral statuses:

- Accepted
- Rejected
- Converted

## SynergyMap

Fields:

- FromSalespersonID
- ToSalespersonID
- SynergyType
- SynergyStrength

---

# 7. Data Validation

Never train directly from unvalidated Excel data.

Validation must check:

- required sheets exist
- required columns exist
- column names are standardised
- dates parse correctly
- IDs are non-null
- salesperson IDs exist in the salesperson master
- customer IDs exist in the customer master
- opportunity values are non-negative
- revenue is non-negative
- percentages are within valid ranges
- duplicate primary keys are detected
- impossible dates are detected
- future dates are flagged
- missingness is measured
- categorical values are standardised
- unexpected product names are flagged

Produce a validation report containing:

```text
File
Sheet
Rows
Columns
Missing %
Duplicate count
Invalid values
Warnings
Validation status
```

Do not silently fix material data-quality problems.

---

# 8. Data Pipeline

The pipeline should be:

```text
Excel
  ↓
Read
  ↓
Schema Validation
  ↓
Data Quality Checks
  ↓
Standardisation
  ↓
Join Master Data
  ↓
Feature Engineering
  ↓
Model Dataset
```

Separate raw data from processed data.

Never overwrite raw input files.

---

# 9. Feature Engineering

Features should be calculated at appropriate time windows.

Suggested windows:

- last 7 days
- last 30 days
- last 90 days
- quarter-to-date
- year-to-date
- trailing 12 months

## Activity features

- CustomerReachouts
- Meetings
- Calls
- Emails
- AccountReviews
- MeetingRate
- ReachoutToMeetingRate
- MeetingsPerWeek
- ReachoutsPerCustomer

## Pipeline features

- OpportunitiesCreated
- OpportunitiesWon
- PipelineValue
- AverageOpportunityValue
- PipelinePerCustomer
- OpportunityGrowth
- OpportunityWinRate

## Revenue features

- Revenue
- RevenueGrowth
- GrossProfit
- GrossMargin
- AverageDealSize
- RevenuePerMeeting
- RevenuePerOpportunity
- RevenuePerReachout

## New business

- NewCustomers
- NewCustomerRevenue
- NewCustomerRate

## Cross-sell

- CrossSellOpportunities
- CrossSellRevenue
- CrossSellRate
- ProductsPerCustomer
- CustomerProductPenetration
- WhitespaceCount
- WhitespaceValue

## Retention

- RetentionRate
- ExistingCustomerRevenue
- ChurnedCustomers where data is available

## Efficiency

- SalesCycleDays
- OpportunityToCloseRate
- RevenuePerActivity
- GrossProfitPerOpportunity

---

# 10. Synergy Feature Engineering

Internal referrals are a first-class feature.

Calculate:

- SynergyReferralsSent
- SynergyReferralsReceived
- ReferralAcceptanceRate
- ReferralConversionRate
- ReferralWinRate
- ReferralRevenue
- ReferralGrossProfit
- RevenueGeneratedFromSynergy
- AverageReferralValue
- SynergyContributionToRevenue

For each salesperson pair:

```text
Salesperson A
      ↓
Salesperson B
      ↓
Referral
      ↓
Opportunity
      ↓
Won
      ↓
Revenue
```

Calculate historical pair performance.

Example:

```text
Alice → Ben

Referrals: 18
Accepted: 16
Converted: 9
Win rate: 56%
Revenue: £145K
```

Use this to identify strong internal collaboration patterns.

Do not assume that high referral volume means high synergy. Measure outcomes.

---

# 11. Customer Whitespace

For each customer:

1. Determine products currently held.
2. Determine products potentially addressable.
3. Identify missing products.
4. Estimate whitespace value where sufficient data exists.
5. Rank opportunities.

Example:

```text
Customer
Existing:
- Connectivity
- Hosted Telephony

Whitespace:
- Cyber Security
- Cloud IT
- IT Support
```

Features:

- CurrentProductCount
- AddressableProductCount
- MissingProductCount
- ProductPenetration
- WhitespaceScore
- EstimatedWhitespaceRevenue

Do not make revenue estimates appear as guaranteed revenue. Label them as estimates/potential.

---

# 12. Define ML Targets Before Training

Do not create features first and choose targets later.

The initial demo should evaluate at least two separate ML tasks.

## Task A: Regression

Predict future revenue.

Example:

```text
Target:
Revenue_next_90_days
```

or:

```text
Revenue_next_quarter
```

## Task B: Classification

Predict whether an opportunity will be won.

Example:

```text
Target:
OpportunityWon = 1 / 0
```

Optional future task:

## Task C: Performance classification

Classify whether a salesperson is likely to achieve a defined target threshold.

Example:

```text
TargetAttainment >= 100%
```

Do not combine unrelated targets into a single model.

---

# 13. Avoid Data Leakage

This is mandatory.

If predicting future revenue, only use information available before the prediction date.

Example:

```text
January–March data
        ↓
Predict April–June
```

Never use:

- future revenue
- future opportunities
- future meetings
- future wins
- future referrals
- future customer products

in the feature set.

Use time-based splits.

Example:

```text
Train:
Jan–Jun

Validation:
Jul–Sep

Test:
Oct–Dec
```

For stronger evaluation, use rolling/expanding time-series validation.

---

# 14. Model Candidates

For regression:

1. Mean/median baseline
2. Linear Regression
3. Elastic Net
4. Random Forest
5. XGBoost
6. LightGBM
7. CatBoost

For classification:

1. Dummy baseline
2. Logistic Regression
3. Random Forest
4. XGBoost
5. LightGBM
6. CatBoost

Models should be trained using consistent feature sets and comparable validation splits.

---

# 15. Model Selection

Do not choose a model based only on a single metric.

## Regression metrics

Calculate:

- MAE
- RMSE
- R²
- MAPE where appropriate
- Median Absolute Error

## Classification metrics

Calculate:

- ROC-AUC
- PR-AUC
- Precision
- Recall
- F1
- Log Loss
- Calibration

## Business metrics

Also evaluate:

- revenue ranking quality
- top-performer identification
- high-value opportunity identification
- recommendation usefulness
- prediction stability
- performance across salesperson segments
- performance across regions
- performance across customer segments

The final model selection should balance:

```text
Predictive performance
+
Business usefulness
+
Stability
+
Explainability
+
Operational simplicity
```

---

# 16. Model Comparison Output

Create an MLflow experiment for each training run.

Store:

- model name
- parameters
- features
- training period
- validation period
- test period
- metrics
- feature importance
- model artifact
- dataset version
- code version

Example comparison:

```text
Model              RMSE      MAE       R²
---------------------------------------------
Baseline           £95K      £71K     0.42
Linear Regression  £82K      £64K     0.61
Random Forest      £54K      £41K     0.78
XGBoost            £43K      £34K     0.86
LightGBM           £45K      £36K     0.84
CatBoost           £47K      £37K     0.82
```

These numbers are examples only. Never hard-code demo performance claims.

---

# 17. Model Explainability

Use SHAP for tree-based models where appropriate.

For every salesperson prediction, expose:

- top positive drivers
- top negative drivers
- magnitude of contribution
- baseline/expected value
- prediction
- uncertainty where available

Example:

```text
Predicted Revenue: £625K

Positive:
+ Opportunities Created
+ Win Rate
+ Existing Customer Revenue
+ Synergy Revenue

Negative:
- Cross-Sell Activity
- Meeting Frequency
- New Customer Acquisition
```

Do not present SHAP as causal proof.

Use language such as:

- "associated with"
- "contributing to the prediction"
- "model driver"

Avoid:

- "causes"
- "will definitely increase revenue"

unless supported by a separate causal analysis.

---

# 18. Salesperson Benchmarking

Create a performance benchmark based on appropriate cohorts.

Possible cohorts:

- Segment
- Seniority
- Region
- Territory
- Customer base size
- Sales role/specialism

Do not compare every salesperson blindly.

Example:

```text
Actual Revenue       £580K
Expected Revenue     £625K
Benchmark Revenue    £640K

Performance Gap      -£45K
```

Create a performance score from multiple validated metrics, but keep the underlying metrics visible.

Avoid creating an opaque score without explaining its calculation.

---

# 19. Top Performer Analysis

Define top performers using a transparent rule.

Possible rule:

```text
Top 10% by risk-adjusted revenue performance
```

or:

```text
Top quartile by target attainment + profitability
```

Do not define "best salesperson" using only revenue.

Consider:

- revenue
- gross profit
- target attainment
- new customer acquisition
- cross-sell
- retention
- win rate
- sales efficiency

Control for differences in territory/customer opportunity where possible.

---

# 20. Recommendations

Recommendations should be generated from measurable gaps.

Example:

```text
Salesperson:
Emma

Gap:
Cross-sell activity 31% below peer benchmark.

Potential opportunity:
42 customers have relevant whitespace.

Synergy:
Cyber Security referral partner available.

Recommendation:
Prioritise Cyber Security discovery conversations
with existing Cloud/IT customers and collaborate
with the highest-performing Cyber Security specialist.
```

Recommendations should be ranked.

Example:

```text
Priority 1:
Cross-sell Cyber Security

Priority 2:
Increase qualified meetings

Priority 3:
Use internal Cyber Security referral synergy
```

---

# 21. Dash Application

Create five primary pages.

## 1. Executive Overview

Display:

- Total Revenue
- Revenue vs Target
- Average Performance Score
- Top Performers
- Performance Distribution
- Predicted Revenue
- Pipeline
- Cross-sell Potential

## 2. Salesperson Performance

Dropdown:

```text
Select Salesperson
Select Period
```

Display:

- Actual Revenue
- Expected Revenue
- Performance Gap
- Performance Score
- Meetings
- Opportunities
- Win Rate
- New Customers
- Cross-sell
- Synergy
- Trend

## 3. Performance Drivers

Display:

- SHAP feature importance
- Actual vs benchmark
- Driver waterfall
- activity-to-outcome relationships

## 4. Customer Whitespace

Display:

- customer
- current products
- missing products
- whitespace score
- estimated potential
- recommended product
- recommended salesperson/synergy partner

## 5. Sales Synergy

Display:

- referrals sent
- referrals received
- referral conversion
- referral revenue
- best salesperson pairs
- unused synergy opportunities

Use network graphs where useful.

---

# 22. Dash Design Principles

The dashboard should be:

- executive-friendly
- visually simple
- interactive
- fast
- explainable
- focused on actions

Avoid putting every available metric on screen.

Use:

```text
Summary
→ Diagnosis
→ Opportunity
→ Recommendation
```

as the main user journey.

---

# 23. Prediction API

Separate Dash from model inference.

Preferred architecture:

```text
Dash
  ↓
FastAPI
  ↓
Production Model
```

Example endpoint concept:

```text
POST /predict/revenue
POST /predict/opportunity
GET  /salesperson/{id}/performance
GET  /salesperson/{id}/drivers
GET  /customer/{id}/whitespace
GET  /synergy/{id}
```

The Dash application should not contain model-training logic.

---

# 24. Model Training Pipeline

Training should be executable from the command line.

Example:

```bash
python scripts/run_pipeline.py
```

Pipeline:

```text
Load Excel
    ↓
Validate
    ↓
Transform
    ↓
Feature Engineering
    ↓
Create Time-Based Splits
    ↓
Train Baselines
    ↓
Train Candidate Models
    ↓
Evaluate
    ↓
Compare
    ↓
Select
    ↓
Log to MLflow
    ↓
Register Best Model
```

---

# 25. Production Model Registry

Use MLflow model stages/tags or an equivalent registry workflow.

At minimum maintain:

```text
Candidate
Validation
Production
Archived
```

Never overwrite the production model without recording:

- model version
- training date
- data version
- features
- metrics
- reason for promotion

---

# 26. Monitoring

After deployment, monitor:

## Data drift

- missingness
- categorical changes
- feature distributions
- new product types
- new salespeople
- changes in customer segments

## Prediction drift

- predicted revenue distribution
- predicted win probability
- performance score distribution

## Model performance

When actual outcomes become available:

- MAE
- RMSE
- R²
- AUC
- calibration
- business KPIs

A model that performed well six months ago may not remain optimal.

Retrain periodically based on observed drift and business requirements.

---

# 27. Security

Do not put:

- passwords
- API keys
- database credentials
- secrets

in Git or Excel.

Use:

- environment variables locally
- Azure Key Vault in production
- managed identities where possible

Anonymise demo data.

Never use real customer information in the manager demonstration unless approved.

---

# 28. Testing

Tests must cover:

## Data

- schema
- missing values
- duplicates
- invalid IDs
- date validation

## Features

- win rate calculations
- revenue aggregation
- cross-sell calculations
- whitespace calculations
- synergy calculations

## ML

- target construction
- leakage checks
- train/test separation
- prediction schema
- model loading

## API

- valid requests
- invalid requests
- missing salesperson
- missing customer

## Dash

- page rendering
- callbacks
- filters
- empty states
- error states

---

# 29. Demo Workflow

For the first manager demo, do NOT build the entire production system.

Build this vertical slice:

```text
Excel
 ↓
Validation
 ↓
Feature Engineering
 ↓
3–6 Candidate Models
 ↓
Model Comparison
 ↓
Best Model
 ↓
SHAP
 ↓
Salesperson Benchmark
 ↓
Dash
```

The demo should allow the manager to:

1. Select a salesperson.
2. See actual performance.
3. See expected performance.
4. See the performance gap.
5. See what drives the prediction.
6. See customer whitespace.
7. See synergy opportunities.
8. See recommended actions.
9. View the model comparison page.

---

# 30. Manager Demo Story

Use this narrative:

### Step 1

"We start with the data we already have."

### Step 2

"We standardise and validate it."

### Step 3

"We create behavioural, commercial, customer and synergy features."

### Step 4

"We don't assume a particular ML algorithm works best."

### Step 5

"We test multiple models using historical time-based validation."

### Step 6

"We select the best model based on predictive accuracy and business usefulness."

### Step 7

"The model predicts expected performance."

### Step 8

"We compare expected performance against actual performance and peer benchmarks."

### Step 9

"We identify the factors contributing to the performance gap."

### Step 10

"We identify customer whitespace and internal sales synergy."

### Step 11

"We provide actionable recommendations through Dash."

### Step 12

"Once validated, the same pipeline can move from Excel to SQL/CRM data and from the local Mac to Azure."

---

# 31. Critical Business Caveats

The system must not:

- label a salesperson as "bad" solely from model output
- assume correlation equals causation
- recommend unrealistic activity levels
- use future information
- rank people without considering opportunity/territory differences
- expose sensitive individual performance unnecessarily
- treat predictions as guaranteed revenue
- hide uncertainty
- automatically make employment/performance-management decisions

The system is intended to support sales coaching and commercial planning.

---

# 32. Definition of Success

The project succeeds if it can demonstrate:

```text
1. Reliable data ingestion
2. Reproducible feature engineering
3. Leakage-free ML evaluation
4. Evidence-based model selection
5. Accurate predictions
6. Explainable predictions
7. Useful salesperson benchmarking
8. Customer whitespace identification
9. Salesperson synergy analysis
10. Actionable recommendations
11. Interactive Dash experience
12. Clear path to production deployment
```

The most important final question is not:

> "Which model has the highest accuracy?"

It is:

> "Does this system provide reliable, explainable and actionable information that helps sales managers improve commercial performance?"

---

# 33. Two-Part Application Architecture

The project must be separated into two clear application experiences:

1. **Sales Manager Application**
2. **Data Scientist / ML Monitoring Application**

These two experiences may live inside the same Plotly Dash project during the local demo, but they should be logically separated using routes, pages, roles, and permissions.

Recommended structure:

```text
dash_app/
│
├── app.py
├── auth.py
├── routes.py
│
├── manager_portal/
│   ├── layouts/
│   │   ├── executive_overview.py
│   │   ├── salesperson_performance.py
│   │   ├── performance_drivers.py
│   │   ├── customer_whitespace.py
│   │   ├── sales_synergy.py
│   │   └── recommendations.py
│   │
│   └── callbacks/
│       ├── overview_callbacks.py
│       ├── salesperson_callbacks.py
│       ├── driver_callbacks.py
│       ├── whitespace_callbacks.py
│       ├── synergy_callbacks.py
│       └── recommendation_callbacks.py
│
├── data_scientist_portal/
│   ├── layouts/
│   │   ├── data_quality.py
│   │   ├── feature_drift.py
│   │   ├── model_comparison.py
│   │   ├── model_accuracy.py
│   │   ├── model_explainability.py
│   │   ├── prediction_monitoring.py
│   │   ├── retraining.py
│   │   └── feature_lab.py
│   │
│   └── callbacks/
│       ├── data_quality_callbacks.py
│       ├── drift_callbacks.py
│       ├── comparison_callbacks.py
│       ├── accuracy_callbacks.py
│       ├── explainability_callbacks.py
│       ├── monitoring_callbacks.py
│       ├── retraining_callbacks.py
│       └── feature_lab_callbacks.py
│
└── shared/
    ├── components.py
    ├── charts.py
    ├── filters.py
    └── utils.py
```

The manager portal should focus on business actions.

The data scientist portal should focus on data quality, model performance, feature behaviour, model drift, explainability, and retraining decisions.

---

# 34. Sales Manager Portal

The Sales Manager Portal is the business-facing application.

It should answer:

```text
Who is performing well?
Who needs support?
Why is performance different?
Which customers have whitespace?
Which opportunities should we prioritise?
Which salesperson should collaborate with whom?
What actions should the manager take?
```

The manager should not need to understand model training, hyperparameters, SHAP internals, or MLflow runs.

## 34.1 Pages

### Page 1: Executive Overview

Purpose:

Give a high-level view of team performance.

Display:

- total revenue
- target attainment
- pipeline value
- gross profit
- win rate
- new customers
- cross-sell revenue
- top performers
- under-supported performers
- revenue forecast
- forecast gap
- team-level recommendations

### Page 2: Salesperson Performance

Purpose:

Allow the manager to select a salesperson and understand their actual vs expected performance.

Display:

- actual revenue
- predicted/expected revenue
- performance gap
- benchmark revenue
- performance score
- win rate
- meetings
- reachouts
- opportunities created
- opportunities won
- new customers
- cross-sell activity
- gross profit
- retention rate

The manager should be able to filter by:

- salesperson
- team
- region
- segment
- product area
- time period
- peer group

### Page 3: Performance Drivers

Purpose:

Explain what is driving the salesperson’s predicted or actual performance.

Display:

- top positive model drivers
- top negative model drivers
- actual vs peer benchmark
- gap analysis
- driver trend over time

Use careful language:

```text
"These features contributed most to the model prediction."
```

Avoid:

```text
"These features caused the performance."
```

### Page 4: Recommendations

Purpose:

Translate model output into practical sales-management actions.

Example outputs:

```text
Recommendation 1:
Increase Cyber Security cross-sell discovery across existing Cloud IT customers.

Reason:
32 customers have Cyber Security whitespace.

Recommended partner:
Ben Carter.

Evidence:
Ben has historically converted 54% of Cyber Security referrals from Cloud IT accounts.
```

Recommendations should be ranked by:

- potential revenue
- confidence
- ease of action
- urgency
- available internal synergy

### Page 5: Customer Whitespace

Purpose:

Help managers identify customers that may be suitable for cross-sell or upsell.

Display:

- current products
- missing products
- whitespace score
- estimated potential value
- recommended product
- recommended salesperson or specialist
- relevant historical wins
- next recommended action

### Page 6: Sales Synergy

Purpose:

Show how salespeople pass leads to each other and where collaboration can improve.

Display:

- referrals sent
- referrals received
- accepted referrals
- converted referrals
- referral win rate
- referral revenue
- best-performing salesperson pairs
- unused collaboration opportunities
- network graph of salesperson relationships

Example insight:

```text
Alice has 23 Cloud IT customers with Cyber Security whitespace,
but only 2 referrals to the Cyber Security specialist.
```

### Page 7: What-If Scenario

Purpose:

Allow the manager to explore possible performance scenarios.

Example:

```text
Current:
Meetings = 42
Cross-sell opportunities = 8

Scenario:
Meetings = 55
Cross-sell opportunities = 14

Estimated revenue impact:
+£34,000
```

This must be labelled as a scenario estimate, not a guarantee.

---

# 35. Data Scientist / ML Monitoring Portal

The Data Scientist Portal is the technical monitoring and model-governance application.

It should answer:

```text
Is the data good enough?
Which model performs best?
Is the model still accurate?
Are predictions drifting?
Are important features changing?
Do we need to retrain?
What happens if we add new data sources?
```

This portal is not primarily for sales managers.

## 35.1 Pages

### Page 1: Data Quality

Purpose:

Show whether the uploaded or connected data is suitable for training and prediction.

Display:

- uploaded files
- row counts
- missing values
- duplicate records
- invalid IDs
- unexpected categories
- invalid dates
- product mismatches
- data freshness
- validation pass/fail status

Example:

```text
MonthlyPerformance: PASS
Opportunities: WARNING
SynergyReferrals: PASS
Activities: WARNING

Warning:
12 opportunities have missing CloseDate.
```

### Page 2: Feature Store / Feature Summary

Purpose:

Show all engineered features available for modelling.

Display:

- feature name
- source table
- feature type
- calculation window
- missingness
- distribution
- correlation with target
- feature importance
- leakage risk
- included/excluded status

Example:

```text
Feature:
SynergyReferralConversionRate_90D

Source:
SynergyReferrals + Opportunities

Included:
Yes

Leakage Risk:
Low
```

### Page 3: Model Comparison

Purpose:

Compare all trained candidate models fairly.

Display:

For revenue prediction:

- MAE
- RMSE
- R²
- MAPE
- median absolute error
- error by segment
- error by region
- error by salesperson seniority

For opportunity win prediction:

- ROC-AUC
- PR-AUC
- precision
- recall
- F1
- log loss
- calibration
- confusion matrix

Models to compare:

- baseline model
- linear / elastic net
- random forest
- XGBoost
- LightGBM
- CatBoost

The page should clearly show:

```text
Best model by predictive performance
Best model by explainability
Best model by business usefulness
Selected production model
```

### Page 4: Model Accuracy Over Time

Purpose:

Show whether model accuracy changes as new sales data arrives.

Display:

- monthly MAE
- monthly RMSE
- rolling R²
- prediction error by month
- prediction error by salesperson segment
- prediction error by product
- prediction error by region
- actual vs predicted trend

Example:

```text
Model performance deterioration detected:
RMSE increased from £43K to £61K over the last 3 months.
```

### Page 5: Prediction Monitoring

Purpose:

Track how predictions behave in production.

Display:

- prediction volume
- average predicted revenue
- prediction distribution
- prediction confidence
- outlier predictions
- predictions by salesperson
- predictions by segment
- predictions by product
- failed prediction requests

### Page 6: Feature Drift

Purpose:

Detect when new data behaves differently from training data.

Display:

- feature distribution in training data
- feature distribution in new data
- drift score
- missingness change
- new categories
- changed ranges
- warning thresholds

Examples of drift:

```text
Average meetings per salesperson has dropped 28%.
Cyber Security opportunity volume has doubled.
New product category detected: AI Managed Services.
```

### Page 7: Explainability Monitoring

Purpose:

Track whether the model is relying on stable and sensible drivers.

Display:

- global feature importance
- feature importance over time
- SHAP summary
- driver changes
- top positive drivers
- top negative drivers
- feature dependence plots

Warning example:

```text
Model is increasingly relying on Region rather than activity or pipeline features.
Review fairness/business validity.
```

### Page 8: Retraining Control

Purpose:

Allow the data scientist to retrain models locally.

Actions:

- select training period
- select validation period
- select target
- select feature groups
- run candidate models
- compare results
- register candidate model
- promote model to local production

The sales manager should not have access to this page.

### Page 9: Feature Lab

Purpose:

Test new features such as call transcriptions, salesperson notes, CRM notes, meeting summaries, email sentiment, or product-interest signals.

Display:

- new feature source
- feature extraction method
- feature coverage
- missingness
- feature importance
- model impact
- accuracy before/after
- business usefulness
- approval status

---

# 36. Future Data Sources and Behavioural Features

The project must leave room for new behavioural data sources.

Possible future inputs:

- call transcriptions
- salesperson notes
- CRM notes
- meeting notes
- email summaries
- proposal text
- objection notes
- customer support tickets
- customer satisfaction scores
- renewal notes
- deal-loss reasons
- call sentiment
- meeting sentiment
- product-interest keywords
- competitor mentions
- pricing objections
- urgency signals
- decision-maker mentions

These data points can improve the behavioural understanding of sales performance.

However, they should be added carefully and measured objectively.

---

# 37. Text Data Extension Layer

Text data should not be added directly into the model as raw text during the first implementation.

Instead, create a text feature extraction layer.

Recommended structure:

```text
src/
├── text_features/
│   ├── __init__.py
│   ├── transcription_loader.py
│   ├── notes_loader.py
│   ├── text_cleaning.py
│   ├── keyword_features.py
│   ├── sentiment_features.py
│   ├── topic_features.py
│   ├── embedding_features.py
│   └── text_feature_registry.py
```

Pipeline:

```text
Call transcripts / Notes
        ↓
Text cleaning
        ↓
Feature extraction
        ↓
Aggregation by salesperson/customer/opportunity/time window
        ↓
Join to model dataset
        ↓
Model comparison
        ↓
Accuracy and behaviour change analysis
```

---

# 38. Example Behavioural Features from Notes and Transcripts

## Call transcript features

Possible features:

- CallCount
- AverageCallDuration
- TalkToListenRatio
- FollowUpMentioned
- DecisionMakerMentioned
- BudgetMentioned
- TimelineMentioned
- CompetitorMentioned
- PricingObjectionMentioned
- SecurityConcernMentioned
- CloudMigrationMentioned
- RenewalRiskMentioned
- PositiveSentimentScore
- NegativeSentimentScore
- CustomerUrgencyScore
- NextStepClarityScore

## Salesperson note features

Possible features:

- NotesCount
- AverageNoteLength
- ProductInterestMentions
- ObjectionMentions
- DecisionMakerIdentified
- BudgetConfirmed
- TimelineConfirmed
- RiskFlagMentioned
- NextStepCaptured
- FollowUpDateCaptured
- CrossSellSignalMentioned
- CompetitorMentioned
- RenewalRiskMentioned

## Opportunity text features

Possible features:

- DealComplexityScore
- ProductFitScore
- ObjectionIntensityScore
- BuyingIntentScore
- DecisionReadinessScore
- RelationshipStrengthSignal
- CrossSellSignalScore
- CompetitiveRiskScore

## Customer support text features

Possible features:

- TicketVolume
- ComplaintMentions
- SLAIssueMentions
- PositiveSupportSentiment
- NegativeSupportSentiment
- RenewalRiskSignal
- UpsellReadinessSignal

---

# 39. How New Data Points Change Model Behaviour

When new data points are added, the system must compare model behaviour before and after adding them.

Example:

```text
Model v1:
Uses activity, opportunity, revenue, customer, product, and synergy features.

Model v2:
Adds call transcription and salesperson note features.
```

Compare:

```text
Accuracy before
Accuracy after
Feature importance before
Feature importance after
Prediction changes
Recommendation changes
Segment-level impact
Salesperson-level impact
Business usefulness
```

The data scientist portal should show:

```text
Did the new features improve model accuracy?
Did they reduce prediction error?
Did they make recommendations more useful?
Did they introduce bias or instability?
Did they increase missing data?
Did they change which behaviours are considered important?
```

Example behaviour change:

```text
Before adding notes:
The model mainly uses meetings, opportunities, win rate, and cross-sell.

After adding notes:
The model may discover that clear next steps, budget confirmation,
decision-maker involvement, and pricing objections are strong predictors
of opportunity conversion.
```

Another example:

```text
Before adding call transcripts:
A salesperson with many meetings appears highly active.

After adding call transcripts:
The model may learn that meeting quality matters more than meeting volume.
For example, calls where budget, timeline, decision-maker, and next steps are
confirmed may be associated with better conversion.
```

This allows the project to evolve from:

```text
Activity-volume intelligence
```

to:

```text
Activity-quality intelligence
```

---

# 40. New Feature Evaluation Rules

New data sources should only be added to the production model if they pass evaluation.

Evaluate:

## Coverage

```text
Do we have this data for enough salespeople/customers/opportunities?
```

## Quality

```text
Is the text clean, structured, complete, and consistently captured?
```

## Timeliness

```text
Is the data available before the prediction is made?
```

## Leakage risk

```text
Does the note/transcript contain future information?
```

Example leakage risk:

```text
"Customer confirmed they signed the contract yesterday."
```

This should not be used to predict whether the opportunity will be won if the signing event is already known.

## Predictive improvement

```text
Does the feature improve validation/test performance?
```

## Business usefulness

```text
Can the manager act on this insight?
```

## Fairness and governance

```text
Does the feature unfairly penalise certain salespeople, regions, or customer types?
```

## Operational effort

```text
Is it worth the cost and complexity of collecting this data?
```

---

# 41. Feature Versioning

Every feature set must be versioned.

Example:

```text
Feature Set v1:
Core activity + opportunity + revenue + customer + synergy features.

Feature Set v2:
Adds customer whitespace features.

Feature Set v3:
Adds salesperson notes features.

Feature Set v4:
Adds call transcript features.
```

MLflow must record:

- feature set version
- feature list
- feature calculation logic
- data extraction date
- training window
- target definition
- model metrics

This ensures the data scientist can answer:

```text
Why did model v4 behave differently from model v3?
```

---

# 42. Text Feature Governance

Text-derived features require additional care.

The system should not:

- expose full private call transcripts unnecessarily
- rank salespeople using sensitive or inappropriate text content
- use personal characteristics
- use discriminatory language patterns
- make employment decisions automatically
- treat sentiment as objective truth
- punish salespeople for note-writing style without context

Prefer aggregated behavioural signals.

Example:

Use:

```text
NextStepCaptured = 1
BudgetDiscussed = 1
DecisionMakerMentioned = 1
PricingObjectionMentioned = 1
```

Avoid directly exposing raw text unless authorised.

---

# 43. Local Demo Scope for Two-Portals Version

For the local demo, create two tabs or sections:

```text
Manager View
Data Scientist View
```

## Manager View should include:

- Executive Overview
- Salesperson Performance
- Performance Drivers
- Recommendations
- Customer Whitespace
- Sales Synergy
- What-If Scenario

## Data Scientist View should include:

- Data Quality
- Model Comparison
- Model Accuracy Over Time
- Feature Importance
- Feature Drift
- Prediction Monitoring
- Feature Lab for new data sources

The first demo does not need real call transcripts.

Use simulated fields such as:

- NotesCount
- BudgetMentioned
- DecisionMakerMentioned
- NextStepCaptured
- PricingObjectionMentioned
- SentimentScore

These prove the architecture can support richer behavioural data later.

---

# 44. Updated Demo Story

The manager-facing story:

```text
We help the sales manager understand team performance,
identify gaps, find customer whitespace, and recommend action.
```

The data-scientist story:

```text
We monitor whether the model is accurate, stable, explainable,
and still suitable as new data comes in.
```

The future-data story:

```text
As richer behavioural data becomes available, such as call transcripts
or salesperson notes, we can add those features, compare model behaviour
before and after, and only promote them if they improve accuracy and
business usefulness without introducing unacceptable risk.
```

This makes the platform extensible rather than fixed.

---

# 45. Two-Portal Success Criteria

The project succeeds when:

## Sales Manager Portal

- Helps managers understand performance
- Highlights gaps clearly
- Identifies customer whitespace
- Reveals useful salesperson synergy
- Gives practical recommendations
- Supports coaching conversations
- Avoids black-box judgement

## Data Scientist Portal

- Shows data quality clearly
- Tracks model accuracy over time
- Compares candidate models
- Detects drift
- Explains model behaviour
- Supports feature experimentation
- Shows impact of new data sources
- Enables controlled retraining

## Future Data Extension

- New behavioural data can be added without rewriting the system
- New features are versioned
- Model behaviour before/after is measurable
- Text-derived features are governed carefully
- Production promotion is evidence-based

---

# 46. Contract and Billing Linkage Layer

The project must include a formal contract layer between customers, billing, services, and sales ownership.

A customer can have multiple contracts. A contract can have multiple services. A billing row must link back to a valid contract or be flagged for review.

Recommended relationship model:

```text
Customers
  1 ──< Contracts
          1 ──< ContractServices
          1 ──< ContractAuditLog
          1 ──< ExistingCustomerBilling

Salespeople
  1 ──< Contracts.AccountOwnerID
  1 ──< ExistingCustomerBilling.AccountOwnerID
```

The key business rule is:

```text
Customer billing should not be analysed only as product revenue.
Billing should be connected to the contract that governs the service,
the salesperson/account owner responsible for that customer, and the
contract renewal or rollback history.
```

---

# 47. Contract Data Contract

## Contracts

Required fields:

- ContractID
- CustomerID
- AccountOwnerID
- ContractName
- ContractType
- ContractStatus
- OriginalStartDate
- CurrentStartDate
- OriginalEndDate
- CurrentEndDate
- ContractTermMonths
- RenewalWindowDays
- DaysToRenewal
- AutoRenewal
- EndDateChangeCount
- RollbackCount
- RollbackAllowed
- HealthCheckRequired
- HealthCheckReason
- SuggestedAction
- ContractMRR
- ContractARR
- EstimatedAnnualGrossProfit
- RenewalRisk
- SnapshotDate

Contract statuses may include:

- Active
- Upcoming Renewal
- Expired - Rolling
- Expired - Needs Review

## ContractServices

Required fields:

- ContractServiceID
- ContractID
- CustomerID
- ServiceCategory
- Service
- AccountOwnerID
- BillingFrequency
- ServiceMRR
- ServiceARR
- GrossMarginPct
- MonthlyGrossProfit
- ServiceStatus
- ServiceStartDate
- ServiceEndDate
- ServiceRole
- BillingContractServiceKey

This table exists because a single customer can have multiple contracts and each contract can include multiple services.

Example:

```text
Customer ABC
  Contract 1: Connectivity + Hosted Telephony
  Contract 2: Cloud IT + Cyber Security
  Contract 3: IT Support
```

## ContractAuditLog

Required fields:

- AuditID
- ContractID
- CustomerID
- ChangeDate
- ChangedByRole
- ChangeType
- PreviousEndDate
- NewEndDate
- RollbackFlag
- RollbackReason
- DaysChanged
- ApprovalStatus
- SalespersonLinked
- ChangeReason

Audit-log change types may include:

- Extension
- Shortening
- Renewal Pushback
- Commercial Amendment
- Rollback

This allows the system to measure how many times a contract end date has changed and whether a contract was rolled back.

---

# 48. Contract Features for ML

Contract features must be engineered and tested as part of the model comparison process.

Suggested features:

- ContractCount
- ActiveContractCount
- ServiceCountPerContract
- ContractMRR
- ContractARR
- EstimatedAnnualGrossProfit
- DaysToRenewal
- NearestRenewalDate
- UpcomingRenewalCount
- RollingExpiredContractFlag
- EndDateChangeCount
- RollbackCount
- RollbackAllowed
- ContractChangeFrequency
- RenewalRisk
- HealthCheckRequired
- ContractServiceMix
- AutoRenewalFlag

These features can help predict:

- future revenue
- renewal risk
- cross-sell probability
- customer health
- salesperson performance
- opportunity conversion

Do not assume contract features improve the model. The data scientist portal must compare model behaviour before and after adding contract features.

Example comparison:

```text
Model v1:
Activity + opportunity + billing + synergy features

Model v2:
Adds contract, renewal and audit-log features

Compare:
- MAE
- RMSE
- R²
- feature importance
- prediction drift
- recommendation changes
- business usefulness
```

---

# 49. Upcoming Renewal and Health-Check Logic

The application should surface upcoming renewals for the sales manager.

A contract should be flagged if:

```text
DaysToRenewal <= RenewalWindowDays
```

or:

```text
CurrentEndDate < SnapshotDate
```

Priority logic:

```text
Critical: contract is already expired or rolling
High:     renewal due within 30 days
Medium:   renewal due within 60 days
Planned:  renewal due within 120 days
```

The manager-facing app should show:

- CustomerID / CustomerName
- ContractID
- AccountOwnerID / salesperson
- current services linked to the contract
- current contract end date
- days to renewal
- contract MRR/ARR
- number of end-date changes
- rollback count
- renewal risk
- suggested action

Example manager action:

```text
Customer needs a renewal health check.
Reason: Cyber Security contract renews in 43 days and has had 2 end-date changes.
Owner: SP004
Suggested action: schedule service-value review and renewal conversation.
```

---

# 50. Contract Audit and Rollback Governance

The audit log is not just historical record-keeping. It becomes a behavioural and risk signal.

Important measures:

- number of end-date changes
- total days extended or shortened
- count of rollback events
- time since last change
- approval status
- reason for change
- salesperson linked to the contract

Potential insights:

```text
Contracts with repeated end-date changes may need manager review.
Contracts with rollback events may indicate billing correction, customer negotiation,
renewal friction or commercial governance issues.
Contracts that are rolling after end date should be prioritised for health checks.
```

These insights should feed both portals:

## Sales Manager Portal

Shows practical action:

```text
Reach out to customer for contract health check.
Review service value before renewal.
Confirm renewal owner.
Check whether contract rollback indicates customer concern.
```

## Data Scientist Portal

Shows modelling behaviour:

```text
Do rollback counts improve renewal-risk prediction?
Do contract extensions correlate with lower win probability?
Does renewal proximity change salesperson performance expectations?
```

---

# 51. Updated Sales Manager Portal Requirements

The manager portal must include a renewal/contract health section.

Add page:

## Contract Renewals and Health Checks

Display:

- upcoming renewals
- expired/rolling contracts
- account owner/salesperson
- services under each contract
- contract MRR/ARR
- renewal risk
- end-date change count
- rollback count
- recommended next action

Manager actions:

- filter by salesperson
- filter by renewal priority
- filter by service
- filter by customer segment
- identify customers needing health checks
- identify contracts with repeated end-date changes
- identify contracts with rollback history
- view customer contract summary
- prioritise renewal outreach

---

# 52. Updated Data Scientist Portal Requirements

The data scientist portal must include contract-feature monitoring.

Add sections to Feature Lab and Model Monitoring:

- contract feature coverage
- missing contract links
- unmatched billing rows
- distribution of DaysToRenewal
- distribution of EndDateChangeCount
- distribution of RollbackCount
- model accuracy before/after contract features
- feature importance of contract features
- drift in renewal profile
- drift in contract service mix
- impact on recommendations

Important checks:

```text
Are billing rows linked to valid contracts?
Are all active contracts linked to customers?
Are services under contracts consistent with billed services?
Are rollback/change counts sensible?
Are expired rolling contracts treated correctly?
```

This ensures contract information improves the system without silently introducing data-quality issues.

---

# 53. Operational Meeting, Escalation, and Delivery Requirements

All operational records remain synthetic and local to the project workbook.

Required entities:

- Meetings linked to salesperson, customer, and optional opportunity
- Opportunity notes with response-required and waiting-response status
- Projects linked to opportunities
- Tickets linked to projects and opportunities
- Tasks linked to tickets, projects, and opportunities

Meeting records must support new-business discovery, opportunity reviews, account health checks, regular account reviews, support escalations, and renewal planning. They must include salesperson notes, summaries, next actions, due dates, and explicit critical-finding fields.

The manager portal must provide salesperson-level operational KPIs and opportunity-level delivery status. Opening a meeting displays its notes and highlights critical findings. Opening an opportunity displays related notes, project stage, tickets, and task statuses.

The deterministic question engine must calculate counts, rankings, response queues, and relationship joins with pandas. The local LLM may explain only the bounded records retrieved by that engine. It must not invent meeting, escalation, project, ticket, or task information.

Revenue experiments may use only lagged meeting and note features available before the target month. Current project and ticket snapshots must not be used to predict historical outcomes. Compare the same chronological holdout before and after operational features and report whether accuracy improved or worsened.

---

# 54. Current-Year Pipeline Forecasting Requirements

The manager portal must default to current-year performance while retaining prior-year records for local model training. Current opportunities require pipeline stage, expected close date, forecast category, stage probability, next action, time in stage, and pipeline risk.

Pipeline revenue must be probability adjusted rather than treating all open value as achievable. Compare YTD recognised revenue plus weighted pipeline with the annual target at team and salesperson level. Display open pipeline, weighted forecast, year-end forecast, forecast gap, coverage, and a transparent achievability score.

The local opportunity classifier must show accuracy, precision, recall, F1, and ROC-AUC. Exclude its probability from the blended forecast when holdout ROC-AUC is below the configured quality threshold. Stage probability, historical conversion, unanswered customer responses, critical findings, and pipeline risk remain auditable inputs.

The workbook engine and local LLM must answer pipeline forecast, target achievability, and coverage-action questions. Numeric values come from deterministic local calculations. Suggestions must cite measurable gaps such as insufficient pipeline, waiting responses, stalled opportunities, or high-risk value.

---

# 55. Compositional Local Question Engine

The question engine must build a structured query plan before reading records. Intent routing must not rely on one hard-coded sentence per question. Each plan should resolve independent slots where available:

- business domain, such as performance, meetings, opportunities, projects, contracts, customers, or data quality
- operation, such as list, rank, status, forecast, or explain
- salesperson and customer filters
- opportunity type and stage
- time period
- sort field and direction
- result limit

Vocabulary and synonyms must remain configurable in the local `config/query_intents.yaml` file. The planner and execution engine must use local Python and pandas only.

Supported meeting periods include:

- latest complete workbook week for "last week", "previous week", and "prior week"
- previous complete workbook month for "last month", "previous month", and "prior month"
- an explicit named calendar month and year, such as July 2026
- a specific day in named, ISO, or British numeric date format
- an inclusive date range, including shared-year forms such as 1 July to 15 July 2026
- a rolling historical period such as the last 14 or 30 days
- an upcoming rolling period such as the next 30 days
- all recorded upcoming meetings or upcoming meetings within a selected day, range, or month

Relative dates must use the latest valid meeting date in the workbook rather than the computer's current date. The answer must display the resolved date range so the manager can verify the interpretation.

The same period logic must support both rankings and detailed records. Examples include:

```text
Who held the most meetings last month?
Who held the most meetings in July 2026?
Who held the most meetings on 6 August 2026?
Who held the most meetings from 1 to 15 July 2026?
Who held the most meetings in the last 14 days?
What meetings did Alice have last week?
What meetings did Alice have in July 2026?
What meetings does Alice have in the next 30 days?
What are Alice's upcoming meetings in September 2026?
```

Ranking answers must include salesperson and meeting count. Detailed answers should include customer, meeting type, subject, notes, critical findings, next action, follow-up status, and linked opportunity where recorded.

Historical meeting queries include only records marked Held. Upcoming queries require a date after the workbook snapshot and a future status such as Scheduled, Planned, Confirmed, Tentative, or Booked. When the workbook has no such rows, the app must explain the missing scheduling data rather than fabricate an answer. A future scheduled date must not move the workbook snapshot forward.

Full salesperson names and unique first names may be recognised. When a first name matches more than one salesperson, the app must return the matching full-name options and ask the manager to clarify rather than choosing a record.

---

# 56. Deterministic Answer and Local LLM Boundary

Verified facts, filters, joins, counts, rankings, forecasts, and date ranges must be calculated by deterministic local code. The local LLM may explain or summarise the bounded result but must not calculate replacement figures or broaden the query silently.

Deterministic results must be retained in both Workbook Engine and Local LLM modes for:

- performance rankings
- opportunity lists and cross-sell filters
- meeting rankings and meeting detail
- project, ticket, and task status
- pipeline forecasts and coverage actions

Non-deterministic or underspecified questions must return a clarification request. The response should state which information is missing, explain why it matters, and provide a useful example. It must not return every record as a fallback.

The optional local explanation model is `qwen3:4b-instruct` through the local Ollama executable. It is used because its approximately 2.5 GB size is practical on the target M1 MacBook with 16 GB RAM while keeping inference on the machine. It is not fine-tuned or trained on the workbook, does not make the revenue prediction, and must not call a hosted endpoint.

All manager-facing interface text and generated explanations must use British English. Currency must use the pound symbol.

---

# 57. Model Analysis and Feature Re-evaluation

The second portal is named Model Analysis. It must explain why the revenue model exists, what it predicts, how it is validated, and what limitations apply.

Revenue modelling must predict next-month salesperson revenue from information available before the target month. The final six months form a chronological holdout. Dummy Regressor, Linear Regression, Random Forest, and Gradient Boosting must be trained and compared on the same rows using MAE, RMSE, and R-squared. Optional local libraries may add candidates without becoming required dependencies.

Every new feature group must trigger a fresh algorithm comparison on the same temporal holdout. A new feature set is promoted only when the selected validation metric improves and leakage checks pass. More features are not assumed to be better.

The portal must show:

- model purpose and unit of prediction
- candidate model leaderboard
- MAE, RMSE, and R-squared
- selected feature set and model
- feature importance where available
- before-and-after feature experiment results
- data-quality flags, including missing values, duplicate identifiers, invalid dates, and mismatching names or links
- classification metrics when a classification model is evaluated

The saved model artefact and its metadata must remain under the local `models/` directory.

---

# 58. Local and Container Runtime

Local execution remains the default boundary. The Dash server binds to `127.0.0.1`, reads the configured local workbook and YAML files, and stores model artefacts locally. No customer or salesperson data leaves the machine.

The project is structured for a future private Azure Container Apps deployment without changing the analytical contracts. Runtime paths, host, port, model directory, workbook path, metrics path, and query-intent path must be supplied through environment variables. The container must expose health and readiness endpoints and use a single Gunicorn worker by default to avoid loading workbook data and models into several processes.

The container image must not bundle Ollama or silently replace the local LLM with a hosted model. Any future private LLM deployment requires an explicit architecture, identity, networking, storage, retention, and data-residency review.
