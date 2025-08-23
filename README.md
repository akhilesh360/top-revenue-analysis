# Time on Page vs Revenue Analysis

A comprehensive statistical analysis examining the relationship between website Time on Page and Revenue generation.

## 🎯 Project Overview

This project addresses two main tasks:

1. **Statistical Analysis**: Analyze the relationship between Time on Page (TOP) and Revenue using robust statistical methods
2. **Interactive Dashboard**: Demonstrate Central Limit Theorem through an interactive Streamlit application

## 📊 Key Findings

- **Strong positive relationship** between Time on Page and Revenue
- **Significant revenue increase** for each standard deviation increase in Time on Page (controlled)
- Relationship shows **diminishing returns** at higher time values
- Effect remains **consistent across browsers, platforms, and sites**

## 🗂️ Project Structure

```
top-revenue-analysis/
├── data/
│   └── testdata.csv                    # Source dataset
├── analysis/
│   ├── analysis.ipynb                  # Main Jupyter notebook analysis
│   └── generate_pdf_report.py          # PDF report generator
├── figs/                               # Generated visualizations
├── report/                             # Reports and results
├── streamlit/
│   └── app.py                         # Interactive CLT demonstration
├── code_appendix.html                 # Complete code documentation
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Run Analysis

```bash
# Interactive notebook (recommended)
cd analysis
jupyter notebook analysis.ipynb
```

### 3. Generate PDF Report

```bash
cd analysis
python generate_pdf_report.py
```

### 4. Launch Interactive Dashboard

```bash
cd streamlit
streamlit run app.py
```

## 📋 Requirements Satisfaction

### ✅ Task 1: Statistical Analysis
- [x] Relationship analysis between TOP and Revenue
- [x] Control for other variables (browser, platform, site)
- [x] Accessible PDF report for mixed audiences
- [x] Code appendix in HTML format
- [x] Professional visualizations
- [x] Robust statistical methodology

### ✅ Task 2: Interactive Dashboard
- [x] Streamlit application ready for deployment
- [x] Central Limit Theorem demonstration
- [x] Multiple probability distributions
- [x] Interactive parameter controls
- [x] Educational content and visualizations

## 🔧 Technical Stack

- **Python 3.8+** with statsmodels, pandas, matplotlib
- **Jupyter Notebooks** for interactive analysis
- **Streamlit** for web applications
- **Statistical Methods**: Robust regression, spline modeling

## 🌐 Live Deployment

### 📊 **Interactive Central Limit Theorem Demo**
- **Live App**: [Coming Soon - Deploy to Streamlit Cloud]
- **Repository**: [https://github.com/akhilesh360/top-revenue-analysis](https://github.com/akhilesh360/top-revenue-analysis)
- **Local Testing**: `streamlit run streamlit/app.py`

*After deployment, update this section with your live Streamlit Cloud URL*
