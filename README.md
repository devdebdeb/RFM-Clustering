# RFM Customer Segmentation Engine (v1.0)

## Overview
This project performs unsupervised learning (K-Means) to segment customers based on:
* **Recency (R):** Days since last order.
* **Frequency (F):** Total number of orders.
* **Monetary (M):** Total Lifetime Value (LTV).
* **Tenure (T):** Days since first order.

**Selected Model:** K-Means (K=5)
**Identified Segments:** Champions, Loyalists, Potential, At Risk, Hibernating.

## Installation
1. Activate environment: `.\venv\Scripts\activate`
2. Install dependencies: `pip install -r requirements.txt`

## Usage
Run the full production pipeline (Ingest -> Process -> Tag):

```bash

python main.py