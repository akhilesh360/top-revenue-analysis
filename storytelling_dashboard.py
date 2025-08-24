#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Statistical Distributions Storytelling Dashboard
Built for Patrick McCann, SVP Research @ Raptive

Educational dashboard that tells the story of how different distributions
behave and what this teaches us about averages and variability.
Designed for both technical and non-technical audiences.

Author: Sai Akhilesh Veldi
Date: August 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import warnings

# Import our enhanced analytics module
from core_analytics import (
    Config, DataLoader, StatisticalAnalyzer, DistributionGenerator,
    VisualizationEngine, SegmentAnalyzer, ReportGenerator
)

warnings.filterwarnings('ignore')

# Streamlit configuration
st.set_page_config(
    page_title="Statistical Distributions Story",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for storytelling
st.markdown("""
<style>
    .story-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .insight-callout {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-size: 1.1em;
        font-weight: 500;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
    }
    
    .technical-box {
        background: #f8f9fa;
        border: 2px solid #e9ecef;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .distribution-card {
        background: white;
        border: 1px solid #dee2e6;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .metric-highlight {
        background: linear-gradient(45deg, #FE6B8B 30%, #FF8E53 90%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        margin: 0.2rem;
        font-weight: bold;
    }
    
    .big-question {
        background: #e3f2fd;
        border-left: 5px solid #2196f3;
        padding: 1.5rem;
        margin: 2rem 0;
        font-size: 1.2em;
        border-radius: 0 10px 10px 0;
    }
    
    .takeaway-box {
        background: #f1f8e9;
        border: 2px solid #8bc34a;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: #fff3e0;
        border: 2px solid #ff9800;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def render_story_header():
    """Render the main story introduction"""
    st.markdown("""
    <div class="story-header">
        <h1>📊 The Statistical Distributions Story</h1>
        <h3>How Different Data Shapes Affect Business Decisions</h3>
        <p>Understanding why averages can mislead and when they don't</p>
        <p style="font-size: 0.9em; opacity: 0.9;">Built for Patrick McCann, SVP Research @ Raptive</p>
    </div>
    """, unsafe_allow_html=True)

def render_big_question():
    """Render the central question"""
    st.markdown("""
    <div class="big-question">
        <h3>🎯 The Big Question</h3>
        <p><strong>"How do different statistical distributions behave, and what does that teach us about averages and variability?"</strong></p>
        <p>Some data (like height) follows a Normal pattern. But revenue, wealth, and many business metrics are "heavy-tailed" - 
        meaning a few extreme values can dramatically change everything.</p>
    </div>
    """, unsafe_allow_html=True)

def render_introduction_tab():
    """Render introduction and explainer content"""
    st.markdown("## 📖 Understanding Data Shapes")
    
    render_big_question()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="distribution-card">
            <h4>🔔 Normal (Bell Curve) Data</h4>
            <ul>
                <li><strong>Shape:</strong> Symmetric, most values near center</li>
                <li><strong>Examples:</strong> Height, test scores, measurement errors</li>
                <li><strong>Business Impact:</strong> Traditional statistics work well</li>
                <li><strong>Key Insight:</strong> Mean ≈ Median, predictable patterns</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="distribution-card">
            <h4>📈 Heavy-Tailed Data</h4>
            <ul>
                <li><strong>Shape:</strong> Skewed, extreme values possible</li>
                <li><strong>Examples:</strong> Revenue, wealth, website traffic</li>
                <li><strong>Business Impact:</strong> A few "whales" dominate outcomes</li>
                <li><strong>Key Insight:</strong> Mean >> Median, be careful with averages!</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="insight-callout">
        💡 <strong>Why This Matters:</strong> In business, understanding your data's "shape" determines 
        whether you can trust averages, how to set targets, and where to focus your efforts.
    </div>
    """, unsafe_allow_html=True)

def render_distribution_viewer():
    """Render interactive distribution comparison"""
    st.markdown("## 🎛️ Interactive Distribution Explorer")
    
    col_control, col_viz = st.columns([1, 2])
    
    with col_control:
        st.markdown("### Controls")
        
        # Distribution selection
        dist_type = st.selectbox(
            "📊 Choose Distribution",
            ["Normal", "Lognormal", "Pareto", "Exponential"],
            help="Each represents different real-world patterns"
        )
        
        # Sample size
        n_samples = st.slider(
            "Sample Size (n)",
            min_value=100,
            max_value=10000,
            value=1000,
            step=100,
            help="More data = clearer pattern"
        )
        
        # Show metrics toggle
        show_metrics = st.checkbox("Show Mean/Median Lines", value=True)
        
        # Distribution info
        dist_info = DistributionGenerator.get_distribution_info(dist_type)
        
        st.markdown(f"""
        <div class="technical-box">
            <h4>📋 About {dist_type}</h4>
            <p><strong>Description:</strong> {dist_info['description']}</p>
            <p><strong>Real World:</strong> {dist_info['real_world']}</p>
            <p><strong>Business Insight:</strong> {dist_info['business_insight']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_viz:
        # Generate data
        data = DistributionGenerator.generate_distribution(dist_type, n_samples)
        
        # Create visualization
        fig = VisualizationEngine.create_distribution_comparison(data, dist_type, show_metrics)
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate metrics for display
        mean_val = np.mean(data)
        median_val = np.median(data)
        skewness = StatisticalAnalyzer.calculate_metrics(data)['skewness']
        
        # Key metrics
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.markdown(f'<div class="metric-highlight">Mean: {mean_val:.1f}</div>', unsafe_allow_html=True)
        with col_m2:
            st.markdown(f'<div class="metric-highlight">Median: {median_val:.1f}</div>', unsafe_allow_html=True)
        with col_m3:
            st.markdown(f'<div class="metric-highlight">Skewness: {skewness:.2f}</div>', unsafe_allow_html=True)
    
    # Educational callout
    if dist_type == "Normal":
        st.markdown("""
        <div class="takeaway-box">
            ✅ <strong>Normal Distribution Takeaway:</strong> Mean and median are nearly identical. 
            Traditional statistics work great here - you can trust the average!
        </div>
        """, unsafe_allow_html=True)
    elif dist_type in ["Lognormal", "Pareto"]:
        mean_median_ratio = mean_val / median_val
        st.markdown(f"""
        <div class="warning-box">
            ⚠️ <strong>Heavy-Tailed Takeaway:</strong> Mean ({mean_val:.1f}) is {mean_median_ratio:.1f}x higher than median ({median_val:.1f})! 
            A few extreme values are pulling the average way up. The "typical" person is much closer to the median.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="insight-callout">
            📊 <strong>Moderate Skew:</strong> Some concentration in higher values, but not as extreme as Pareto. 
            Mean ({mean_val:.1f}) > Median ({median_val:.1f}), so be cautious with averages.
        </div>
        """, unsafe_allow_html=True)

def render_metrics_stability():
    """Show how different metrics behave under resampling"""
    st.markdown("## 🎯 Stability of Different Metrics")
    
    st.markdown("""
    **The Question:** When your data has extreme values, which metric should you trust more - mean, median, or something else?
    
    **The Test:** We'll take many random samples and see how much each metric jumps around.
    """)
    
    col_control, col_viz = st.columns([1, 2])
    
    with col_control:
        # Distribution for stability test
        stability_dist = st.selectbox(
            "Distribution to Test",
            ["Lognormal", "Pareto", "Normal", "Exponential"],
            help="Heavy-tailed distributions show bigger differences"
        )
        
        n_bootstrap = st.slider(
            "Number of Resamples",
            min_value=50,
            max_value=500,
            value=100,
            step=50,
            help="More resamples = more reliable test"
        )
        
        # Generate base data
        base_data = DistributionGenerator.generate_distribution(stability_dist, 2000)
    
    with col_viz:
        # Create stability chart
        fig = VisualizationEngine.create_metrics_stability_chart(base_data, n_bootstrap)
        st.plotly_chart(fig, use_container_width=True)
    
    # Educational interpretation
    if stability_dist in ["Pareto", "Lognormal"]:
        st.markdown("""
        <div class="insight-callout">
            💡 <strong>Heavy-Tail Insight:</strong> Notice how the sample mean varies much more than the median! 
            When a few extreme values dominate, the median gives you a more stable estimate of the "typical" value.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="takeaway-box">
            ✅ <strong>Light-Tail Insight:</strong> Mean and median are both pretty stable. 
            When data is well-behaved (like Normal), you can trust either metric.
        </div>
        """, unsafe_allow_html=True)

def render_outlier_impact():
    """Demonstrate impact of outliers"""
    st.markdown("## 💥 Outlier Impact Test")
    
    st.markdown("""
    **The Demo:** Start with normal-ish data, then add a few extreme "whale" customers. 
    Watch what happens to your metrics!
    """)
    
    col_control, col_viz = st.columns([1, 2])
    
    with col_control:
        st.markdown("### Outlier Controls")
        
        outlier_strength = st.slider(
            "Outlier Multiplier",
            min_value=2.0,
            max_value=20.0,
            value=10.0,
            step=1.0,
            help="How extreme should the outliers be?"
        )
        
        st.markdown("""
        <div class="technical-box">
            <h4>What's Happening?</h4>
            <p>We're taking 1% of the data points and making them much larger 
            (simulating "whale" customers or viral content).</p>
            <p><strong>Watch:</strong> Mean vs Median response to extremes</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_viz:
        # Generate base data (somewhat normal)
        base_data = DistributionGenerator.generate_distribution("Normal", 1000)
        
        # Create outlier impact demo
        fig = VisualizationEngine.create_outlier_impact_demo(base_data, outlier_strength)
        st.plotly_chart(fig, use_container_width=True)
    
    # Calculate the actual impact
    original_mean = np.mean(base_data)
    original_median = np.median(base_data)
    
    n_outliers = max(1, len(base_data) // 100)
    outliers = np.random.choice(base_data, n_outliers) * outlier_strength
    contaminated_data = np.concatenate([base_data, outliers])
    
    new_mean = np.mean(contaminated_data)
    new_median = np.median(contaminated_data)
    
    mean_change = ((new_mean - original_mean) / original_mean) * 100
    median_change = ((new_median - original_median) / original_median) * 100
    
    st.markdown(f"""
    <div class="warning-box">
        📈 <strong>Impact Summary:</strong> Adding just {n_outliers} extreme values ({n_outliers/len(base_data)*100:.1f}% of data):
        <br>• Mean changed by <strong>{mean_change:.1f}%</strong>
        <br>• Median changed by <strong>{median_change:.1f}%</strong>
        <br><br><strong>Lesson:</strong> Outliers can dramatically distort the mean while barely affecting the median!
    </div>
    """, unsafe_allow_html=True)

def render_inequality_panel():
    """Show inequality visualization with Lorenz curve"""
    st.markdown("## 📊 Inequality Analysis - The 80/20 Rule")
    
    st.markdown("""
    **The Big Picture:** Heavy-tailed data often follows the famous "80/20 rule" - 
    a small fraction of observations drive most of the total outcome.
    """)
    
    col_control, col_viz = st.columns([1, 2])
    
    with col_control:
        inequality_dist = st.selectbox(
            "Distribution for Inequality Test",
            ["Pareto", "Lognormal", "Normal", "Exponential"],
            help="Pareto is famous for extreme inequality"
        )
        
        # Generate data
        inequality_data = DistributionGenerator.generate_distribution(inequality_dist, 2000)
        
        # Calculate some quick stats
        top_10_threshold = np.percentile(inequality_data, 90)
        top_10_share = np.sum(inequality_data[inequality_data >= top_10_threshold]) / np.sum(inequality_data) * 100
        
        st.markdown(f"""
        <div class="technical-box">
            <h4>📈 Quick Stats</h4>
            <p><strong>Top 10% threshold:</strong> {top_10_threshold:.1f}</p>
            <p><strong>Top 10% share of total:</strong> {top_10_share:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_viz:
        # Create Lorenz curve
        fig, gini = VisualizationEngine.create_lorenz_curve(inequality_data)
        st.plotly_chart(fig, use_container_width=True)
    
    # Interpretation based on Gini coefficient
    if gini > 0.6:
        st.markdown(f"""
        <div class="warning-box">
            ⚠️ <strong>High Inequality (Gini: {gini:.3f}):</strong> This is classic "80/20" territory! 
            A small fraction of customers/users/content drives most outcomes. 
            <strong>Business Strategy:</strong> Focus heavily on identifying and retaining your whales.
        </div>
        """, unsafe_allow_html=True)
    elif gini > 0.3:
        st.markdown(f"""
        <div class="insight-callout">
            📊 <strong>Moderate Inequality (Gini: {gini:.3f}):</strong> Some concentration but not extreme. 
            You have important customers, but the distribution is fairly balanced.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="takeaway-box">
            ✅ <strong>Low Inequality (Gini: {gini:.3f}):</strong> Very even distribution! 
            Most customers contribute similarly to your outcomes.
        </div>
        """, unsafe_allow_html=True)

def render_real_data_analysis():
    """Analyze the actual revenue data"""
    st.markdown("## 🎯 Real Data: Your Revenue Analysis")
    
    st.markdown("""
    **Now let's apply these concepts to your actual revenue data from testdata.csv**
    """)
    
    # Load real data
    df, revenue_data = DataLoader.load_revenue_data()
    
    if df is None:
        st.error("Could not load revenue data. Please check testdata.csv exists.")
        return
    
    # Analyze the real data
    metrics = StatisticalAnalyzer.calculate_metrics(revenue_data)
    pattern_analysis = StatisticalAnalyzer.analyze_business_pattern(metrics)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Real data distribution
        fig = VisualizationEngine.create_revenue_distribution_chart(revenue_data, "Your Actual Revenue Distribution")
        
        # Add mean/median lines
        mean_val = metrics['mean']
        median_val = metrics['median']
        
        fig.add_vline(x=mean_val, line_dash="dash", line_color="red", 
                     annotation_text=f"Mean: ${mean_val:.2f}")
        fig.add_vline(x=median_val, line_dash="dot", line_color="green", 
                     annotation_text=f"Median: ${median_val:.2f}")
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Real data Lorenz curve
        fig_lorenz, gini = VisualizationEngine.create_lorenz_curve(revenue_data)
        st.plotly_chart(fig_lorenz, use_container_width=True)
    
    # Key insights about the real data
    st.markdown(f"""
    <div class="insight-callout">
        🎯 <strong>Your Revenue Data Insights:</strong>
        <br>• Pattern: {pattern_analysis['pattern_type']}
        <br>• Skewness: {metrics['skewness']:.2f} (>2 = very heavy-tailed)
        <br>• Mean vs Median: ${mean_val:.2f} vs ${median_val:.2f}
        <br>• Inequality (Gini): {gini:.3f}
        <br>• Top 10% drive: {((revenue_data >= np.percentile(revenue_data, 90)).sum() / len(revenue_data)) * 100:.1f}% of records
    </div>
    """, unsafe_allow_html=True)
    
    # Business recommendations
    st.markdown(f"""
    <div class="takeaway-box">
        💼 <strong>Business Strategy Recommendations:</strong>
        <br>• {pattern_analysis['business_interpretation']}
        <br>• {pattern_analysis['strategic_recommendation']}
        <br>• Risk Level: {pattern_analysis['risk_level']}
    </div>
    """, unsafe_allow_html=True)

def main():
    """Main application"""
    render_story_header()
    
    # Create tabs for the story structure
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📖 Introduction", 
        "🎛️ Distribution Explorer", 
        "🎯 Metric Stability", 
        "💥 Outlier Impact", 
        "📊 Inequality Analysis",
        "🎯 Real Data"
    ])
    
    with tab1:
        render_introduction_tab()
    
    with tab2:
        render_distribution_viewer()
    
    with tab3:
        render_metrics_stability()
    
    with tab4:
        render_outlier_impact()
    
    with tab5:
        render_inequality_panel()
    
    with tab6:
        render_real_data_analysis()
    
    # Final summary
    st.markdown("---")
    st.markdown("""
    ## 🎓 Key Takeaways for Business
    
    **For Non-Technical Readers:**
    - 📊 Some data is "normal" (bell curve), some is "heavy-tailed" (few extreme values)
    - ⚠️ Heavy-tailed data: averages can mislead, medians are more trustworthy
    - 🎯 Business focus: identify and retain your "whale" customers/content
    
    **For Technical Readers:**
    - 📈 Use skewness and Gini coefficients to characterize distributions
    - 🔧 Bootstrap methods show metric stability under resampling
    - 📊 Lorenz curves visualize inequality patterns
    - ⚙️ Choose robust statistics (median, trimmed mean) for heavy-tailed data
    
    **For Patrick McCann @ Raptive:**
    This framework applies directly to ad revenue, user engagement, and content performance - 
    all domains where understanding distributional properties drives strategic decisions.
    """)

if __name__ == "__main__":
    main()
