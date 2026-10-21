# ⚡ InsightAnalyst AI — Enterprise Data Diagnostics & Autonomous RAG Analytics Engine

**InsightAnalyst AI** is an enterprise-grade automated data diagnostic, cleaning, unsupervised machine learning, and conversational analytics platform rewritten as a high-performance **React + TypeScript + Tailwind CSS** application powered by **AlaSQL In-Memory Analytics** and **Google Gemini 2.0 AI**.

It enables business leaders, data engineers, and analysts to transform raw, messy tabular datasets into production-grade cleaned datasets, actionable statistical insights, interactive visual analytics dashboards, and executive dossiers in seconds.

---

## 🌟 Key Highlights & Capabilities

### 1. 📊 Automated Executive Diagnostic & RAG Synthesis
* **RAG-Powered Briefings**: Ingests raw tabular data, segments records into semantic vector chunks, and generates context-grounded executive summaries.
* **Dataset Health & Hygiene Scoring**: Real-time 0–100 quality telemetry evaluating missingness, duplicate rates, schema anomalies, and variance health.
* **Exportable Dossiers**: One-click download of rich HTML executive audit dossiers and Markdown reports.

### 2. 🛡️ Defensive Automated Cleaning Pipeline & DAG Telemetry
* **Step 1: Duplicate Scan & Purge**: Fast deduplication across high-dimensional feature spaces.
* **Step 2: Numeric Imputation**: Robust median-strategy imputation preserving distribution skewness.
* **Step 3: Categorical Normalization**: Mode imputation paired with explicit sentinel labeling (`"Unknown"`).
* **Step 4: String Hygiene**: Automated stripping, whitespace compaction, and uniform casing.
* **Step 5: Chrono UTC Datetime Coercion**: Multi-format datetime parsing with ISO-8601 schema enforcement.
* **Interactive DAG Telemetry**: Step-by-step latency tracking, cell modification counts, and full data audit metrics.

### 3. 📈 Intelligent Visual Dashboard & Custom Sandbox
* **Automated Visual Grid**: Auto-generated numeric distributions, categorical count distributions, time-series trends, and correlation heatmaps.
* **Natural Language Query Filtering**: Filter your dataset in plain English (e.g. *"Show records with revenue above 5000 in California"*).
* **Custom Visual Explorer Sandbox**: Interactive multi-axis sandbox supporting Bar, Scatter, Line, Pie/Donut, and Bubble plots with configurable groupings and aggregations.

### 4. 🧠 Unsupervised Machine Learning Engine
* **K-Means Clustering & Silhouette Optimization**: Automated customer/entity segmentation with 2D Principal Component Analysis (PCA) projection.
* **Classical Mamdani Fuzzy Inference System**: Continuous anomaly severity assessment evaluating Anomaly Intensity, Feature Deviation, and Business Impact into Low, Medium, High, and Critical risk tiers.
* **Covariance & Correlation Matrices**: Multi-feature Pearson correlation analysis.

### 5. 💬 Conversational AI Analyst with RAG & Gemini
* **Conversational Interface**: Query your data with natural language to generate instant answers, data slices, and charts.
* **Hybrid Intelligence Engine**: Works offline with a deterministic heuristic engine or connects to **Google Gemini 2.0** for deep multi-hop reasoning.
* **DuckDB / AlaSQL Query Studio**: Execute instant analytical SQL queries directly on your dataset in-memory with automatic chart visualizations.

---

## 🛠️ Tech Stack

* **Frontend Framework**: React 19, TypeScript, Vite
* **Styling**: Tailwind CSS v4, Inter font typography
* **Charting**: Chart.js & React-Chartjs-2
* **In-Memory Analytical SQL**: AlaSQL
* **Data Ingestion**: PapaParse (CSV/TSV), SheetJS (XLSX/XLS)
* **AI & LLM**: Google Gen AI SDK (`@google/genai`) with Gemini 2.0 Flash

---

## ⚡ Development & Scripts

```bash
# Start development server on port 3000
npm run dev

# Compile and type check
npm run build
npm run lint
```
