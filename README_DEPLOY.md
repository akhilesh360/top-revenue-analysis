# Revenue Analytics Dashboard 📊

A comprehensive statistical analysis demonstrating the relationship between user engagement (time on page) and revenue, showcasing key statistical concepts like Simpson's Paradox and the Central Limit Theorem.

## 🎯 Purpose

This project demonstrates:
- **Simpson's Paradox**: How overall correlations can be misleading without proper segmentation
- **Central Limit Theorem**: Why sample averages become normally distributed at scale
- **Business Analytics**: Translating statistical insights into actionable business recommendations
- **Mixed Audience Communication**: Accessible to both technical and non-technical stakeholders

## 📋 Project Structure

```
top-revenue-analysis/
├── streamlit_deploy.py          # Main Streamlit app (deployment-ready)
├── generate_brief_report.py     # PDF report generator
├── revenue_analysis_brief_report.pdf  # Generated report
├── requirements.txt             # Dependencies
├── README.md                   # This file
└── analysis/                   # Full analysis notebooks
    └── analysis.ipynb
```

## 🚀 Live Demo

### Streamlit App Deployment

**For Streamlit Cloud (share.streamlit.io):**

1. **Create GitHub Repository** (if not already done):
   ```bash
   git add .
   git commit -m "Add deployment-ready Streamlit app"
   git push origin main
   ```

2. **Deploy to Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Connect your GitHub repository
   - Set main file path: `streamlit_deploy.py`
   - Deploy!

3. **App Features**:
   - 📈 Executive Summary with business impact calculations
   - 🎭 Interactive Simpson's Paradox demonstration
   - 🎲 Central Limit Theorem educational tool
   - 📊 Distribution analysis with statistical tests
   - 🔍 Interactive data explorer with filtering

## 📄 Brief PDF Report

The project includes a **2-page accessible PDF report**:

### Page 1: Executive Summary (Non-Technical)
- Clear business insights without statistical jargon
- Revenue impact scenarios with dollar amounts
- Visual charts focusing on business outcomes
- Strategic recommendations for stakeholders

### Page 2: Statistical Methodology (Technical)
- Simpson's Paradox demonstration
- Model diagnostics and validation
- Distribution analysis and effect sizes
- Methodology notes for technical readers

**Generate the report:**
```bash
python generate_brief_report.py
```

## 🔧 Local Development

**Run locally:**
```bash
# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run streamlit_deploy.py

# Generate PDF report
python generate_brief_report.py
```

## 📊 Key Statistical Demonstrations

### 1. Simpson's Paradox
- Shows how overall correlation can be misleading
- Demonstrates importance of proper data segmentation
- Business insight: Mobile vs Desktop users behave differently

### 2. Central Limit Theorem
- Interactive demonstration with different distributions
- Shows why sample averages become normal at scale
- Business relevance: A/B testing and confidence intervals

### 3. Revenue Modeling
- Realistic business dataset with engagement metrics
- Statistical rigor with proper model diagnostics
- Translation of statistics into business impact

## 🎯 Target Audience

- **Business Stakeholders**: Executive summary with clear ROI calculations
- **Data Scientists**: Statistical methodology and model validation
- **Analysts**: Interactive tools for data exploration
- **Students**: Educational demonstrations of key statistical concepts

## 💡 Key Business Insights

- **Revenue Impact**: Each additional minute of engagement = $X.XXXX in revenue
- **Segment Differences**: Desktop users convert differently than mobile users
- **Statistical Confidence**: Analysis based on 3,000+ user sessions
- **Actionable Recommendations**: Focus on content quality over conversion optimization

## 🛠️ Technologies Used

- **Python**: Core analysis and modeling
- **Streamlit**: Interactive dashboard and deployment
- **Plotly**: Professional interactive visualizations  
- **Statsmodels**: Statistical modeling and diagnostics
- **Matplotlib/Seaborn**: PDF report visualizations
- **Pandas/NumPy**: Data manipulation and analysis

## 📈 Business Value

This project demonstrates how to:
1. **Communicate Complex Statistics** to mixed audiences
2. **Translate Data into Decisions** with clear business impact
3. **Avoid Statistical Pitfalls** like Simpson's Paradox
4. **Build Educational Tools** that scale knowledge across teams

Perfect for showcasing statistical expertise, business acumen, and communication skills to senior stakeholders like SVP Research roles.

---

*Built with ❤️ for demonstrating statistical rigor, business impact, and clear communication in data science.*
