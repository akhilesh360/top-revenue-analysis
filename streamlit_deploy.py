#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Revenue Analytics: Time on Page Impact Analysis
A demonstration of statistical methods for business decision-making

Showcasing Central Limit Theorem, Simpson's Paradox, and Revenue Modeling
Perfect for sharing insights with technical and non-technical stakeholders
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from statsmodels.api import add_constant, OLS

# Page configuration
st.set_page_config(
    page_title="Revenue Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .insight-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def generate_business_data():
    """Generate realistic business dataset demonstrating statistical phenomena"""
    np.random.seed(42)
    n = 3000
    
    # User segments
    segments = np.random.choice(['Mobile', 'Desktop', 'Tablet'], n, p=[0.6, 0.35, 0.05])
    browsers = np.random.choice(['Chrome', 'Safari', 'Firefox', 'Edge'], n, p=[0.7, 0.15, 0.1, 0.05])
    
    # Time on page (log-normal distribution - realistic for web analytics)
    base_time = np.random.lognormal(mean=3.5, sigma=0.8, size=n)
    
    # Add segment effects (Simpson's Paradox setup)
    segment_multiplier = np.where(segments == 'Desktop', 1.4,
                         np.where(segments == 'Mobile', 0.7, 1.1))
    
    time_on_page = base_time * segment_multiplier
    time_on_page = np.clip(time_on_page, 10, 800)  # Reasonable bounds
    
    # Revenue with complex relationship
    # Desktop: higher revenue per time unit but users spend less time
    # Mobile: lower revenue per time unit but more engagement time
    base_revenue = 0.001 + 0.0002 * np.log(time_on_page)
    
    revenue_multiplier = np.where(segments == 'Desktop', 2.5,  # High conversion
                         np.where(segments == 'Mobile', 0.8,   # Lower conversion
                                 1.5))  # Tablet: middle ground
    
    revenue = base_revenue * revenue_multiplier + np.random.normal(0, 0.002, n)
    revenue = np.clip(revenue, 0.0001, None)
    
    return pd.DataFrame({
        'time_on_page': time_on_page,
        'revenue': revenue,
        'segment': segments,
        'browser': browsers,
        'time_minutes': time_on_page / 60
    })

def calculate_simpsons_paradox(df):
    """Demonstrate Simpson's Paradox in the data"""
    overall_corr = df[['revenue', 'time_on_page']].corr().iloc[0, 1]
    
    segment_correlations = {}
    for segment in df['segment'].unique():
        subset = df[df['segment'] == segment]
        if len(subset) > 50:
            corr = subset[['revenue', 'time_on_page']].corr().iloc[0, 1]
            segment_correlations[segment] = {
                'correlation': corr,
                'n': len(subset),
                'avg_time': subset['time_minutes'].mean(),
                'avg_revenue': subset['revenue'].mean()
            }
    
    return overall_corr, segment_correlations

def main():
    st.markdown('<div class="main-header">📊 Revenue Analytics: Statistical Insights for Business</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    ### 🎯 Demonstrating Key Statistical Concepts Through Business Data
    This dashboard showcases important statistical phenomena using realistic revenue data,
    perfect for both technical teams and business stakeholders.
    """)
    
    # Generate data
    with st.spinner("Generating realistic business dataset..."):
        df = generate_business_data()
    
    # Sidebar navigation
    st.sidebar.header("📋 Analytics Sections")
    
    sections = [
        "📈 Executive Summary",
        "🎭 Simpson's Paradox Demo", 
        "🎲 Central Limit Theorem",
        "📊 Revenue Distribution Analysis",
        "🔍 Interactive Data Explorer"
    ]
    
    selected_section = st.sidebar.selectbox("Choose Analysis", sections)
    
    # Dataset overview in sidebar
    st.sidebar.markdown("### 📊 Dataset Overview")
    st.sidebar.metric("Total Users", f"{len(df):,}")
    st.sidebar.metric("Avg Time", f"{df['time_minutes'].mean():.1f} min")
    st.sidebar.metric("Avg Revenue", f"${df['revenue'].mean():.4f}")
    
    # Route to selected section
    if selected_section == "📈 Executive Summary":
        show_executive_summary(df)
    elif selected_section == "🎭 Simpson's Paradox Demo":
        show_simpsons_paradox(df)
    elif selected_section == "🎲 Central Limit Theorem":
        show_clt_demonstration(df)
    elif selected_section == "📊 Revenue Distribution Analysis":
        show_distribution_analysis(df)
    elif selected_section == "🔍 Interactive Data Explorer":
        show_interactive_explorer(df)

def show_executive_summary(df):
    """Executive-friendly summary of key insights"""
    
    st.header("📈 Executive Summary: Key Business Insights")
    
    # Calculate key metrics
    overall_corr = df[['revenue', 'time_on_page']].corr().iloc[0, 1]
    avg_revenue = df['revenue'].mean()
    avg_time = df['time_minutes'].mean()
    
    # Fit simple model for business impact
    X = add_constant(df[['time_on_page']])
    model = OLS(df['revenue'], X).fit()
    time_coeff = model.params['time_on_page']
    
    # Business scenarios
    revenue_per_minute = time_coeff * 60
    monthly_users = 100000
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>Overall Correlation</h3>
            <h2>{:.3f}</h2>
            <p>Time vs Revenue</p>
        </div>
        """.format(overall_corr), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>Revenue Impact</h3>
            <h2>${:.5f}</h2>
            <p>Per additional minute</p>
        </div>
        """.format(revenue_per_minute), unsafe_allow_html=True)
    
    with col3:
        annual_impact = revenue_per_minute * 0.5 * monthly_users * 12  # 30-sec improvement
        st.markdown("""
        <div class="metric-card">
            <h3>Annual Opportunity</h3>
            <h2>${:,.0f}</h2>
            <p>From 30-sec engagement boost</p>
        </div>
        """.format(annual_impact), unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>Model R²</h3>
            <h2>{:.3f}</h2>
            <p>Variance explained</p>
        </div>
        """.format(model.rsquared), unsafe_allow_html=True)
    
    # Key insights
    st.markdown("""
    <div class="insight-box">
        <h4>🔑 Key Business Insights</h4>
        <ul>
            <li><strong>Positive Engagement-Revenue Relationship:</strong> Each additional minute of engagement 
                is associated with ${:.5f} in additional revenue</li>
            <li><strong>Significant Business Impact:</strong> A modest 30-second improvement in average 
                session time could generate ${:,.0f} annually</li>
            <li><strong>Platform Differences:</strong> Mobile and desktop users show different 
                engagement-to-revenue conversion patterns</li>
            <li><strong>Strategic Recommendation:</strong> Focus on content quality and user experience 
                improvements rather than just conversion optimization</li>
        </ul>
    </div>
    """.format(revenue_per_minute, annual_impact), unsafe_allow_html=True)
    
    # Visualization
    fig = px.scatter(
        df.sample(1000), 
        x='time_minutes', 
        y='revenue',
        color='segment',
        title="Revenue vs Engagement Time by User Segment",
        labels={'time_minutes': 'Time on Page (minutes)', 'revenue': 'Revenue ($)'},
        opacity=0.7
    )
    
    # Add trend line
    z = np.polyfit(df['time_minutes'], df['revenue'], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(df['time_minutes'].min(), df['time_minutes'].max(), 100)
    fig.add_traces(go.Scatter(
        x=x_trend, y=p(x_trend), mode='lines', name='Overall Trend',
        line=dict(color='red', width=3, dash='dash')
    ))
    
    st.plotly_chart(fig, use_container_width=True)

def show_simpsons_paradox(df):
    """Demonstrate Simpson's Paradox with business data"""
    
    st.header("🎭 Simpson's Paradox: When Correlations Mislead")
    
    st.markdown("""
    <div class="insight-box">
        <h4>🎯 What is Simpson's Paradox?</h4>
        <p>A statistical phenomenon where a trend appears in different groups of data 
        but disappears or reverses when the groups are combined. This is crucial for 
        business analytics because it shows why simple correlations can be misleading.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Calculate paradox
    overall_corr, segment_corrs = calculate_simpsons_paradox(df)
    avg_within_corr = np.mean([data['correlation'] for data in segment_corrs.values()])
    
    # Display the paradox
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Overall Correlation", f"{overall_corr:.3f}", help="Naive analysis result")
    
    with col2:
        st.metric("Average Within-Group", f"{avg_within_corr:.3f}", help="After controlling for segments")
    
    with col3:
        paradox_strength = abs(overall_corr - avg_within_corr)
        st.metric("Paradox Strength", f"{paradox_strength:.3f}", help="Difference between analyses")
    
    # Detailed breakdown
    st.subheader("📊 Segment-Specific Analysis")
    
    segment_data = []
    for segment, data in segment_corrs.items():
        segment_data.append({
            'Segment': segment,
            'Correlation': f"{data['correlation']:.3f}",
            'Sample Size': f"{data['n']:,}",
            'Avg Time (min)': f"{data['avg_time']:.1f}",
            'Avg Revenue ($)': f"{data['avg_revenue']:.4f}"
        })
    
    st.dataframe(pd.DataFrame(segment_data), use_container_width=True)
    
    # Visualization
    fig = px.scatter(
        df, 
        x='time_minutes', 
        y='revenue', 
        color='segment',
        title="Simpson's Paradox Visualization: Different Trends Within Groups",
        labels={'time_minutes': 'Time on Page (minutes)', 'revenue': 'Revenue ($)'},
        opacity=0.6
    )
    
    # Add overall trend (misleading)
    z_overall = np.polyfit(df['time_minutes'], df['revenue'], 1)
    p_overall = np.poly1d(z_overall)
    x_trend = np.linspace(df['time_minutes'].min(), df['time_minutes'].max(), 100)
    
    fig.add_traces(go.Scatter(
        x=x_trend, y=p_overall(x_trend), mode='lines', 
        name='Overall Trend (Misleading)', line=dict(color='red', width=4, dash='dash')
    ))
    
    # Add segment-specific trends
    colors = ['blue', 'green', 'orange']
    for i, segment in enumerate(df['segment'].unique()):
        segment_data = df[df['segment'] == segment]
        if len(segment_data) > 50:
            z_seg = np.polyfit(segment_data['time_minutes'], segment_data['revenue'], 1)
            p_seg = np.poly1d(z_seg)
            x_seg = np.linspace(segment_data['time_minutes'].min(), segment_data['time_minutes'].max(), 50)
            
            fig.add_traces(go.Scatter(
                x=x_seg, y=p_seg(x_seg), mode='lines',
                name=f'{segment} Trend (True)', line=dict(color=colors[i], width=3)
            ))
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Business interpretation
    st.markdown("""
    <div class="insight-box">
        <h4>💼 Business Implications</h4>
        <p><strong>Why This Matters:</strong> The overall correlation suggests one relationship, 
        but when we control for device type, we see different patterns within each segment.</p>
        <p><strong>Action Item:</strong> Don't optimize all user segments the same way. 
        Mobile and desktop users have fundamentally different engagement-revenue patterns.</p>
        <p><strong>Strategic Insight:</strong> Always segment your analysis by key business dimensions 
        before making strategic decisions.</p>
    </div>
    """, unsafe_allow_html=True)

def show_clt_demonstration(df):
    """Interactive Central Limit Theorem demonstration"""
    
    st.header("🎲 Central Limit Theorem: Why Sample Averages Matter")
    
    st.markdown("""
    <div class="insight-box">
        <h4>🎯 Why This Matters for Business</h4>
        <p>The Central Limit Theorem explains why we can trust sample-based metrics (like A/B test results) 
        even when individual user behavior is highly variable. It's fundamental to statistical inference in business.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Interactive controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        distribution_type = st.selectbox(
            "Population Distribution", 
            ["Revenue (Exponential)", "Time on Page (Log-normal)", "Uniform", "Normal"]
        )
    
    with col2:
        sample_size = st.slider("Sample Size", 5, 200, 30)
    
    with col3:
        num_samples = st.slider("Number of Samples", 100, 1000, 500)
    
    # Generate population based on selection
    np.random.seed(42)
    pop_size = 10000
    
    if distribution_type == "Revenue (Exponential)":
        population = np.random.exponential(scale=0.01, size=pop_size)
        pop_name = "Revenue Distribution (Right-skewed)"
        business_context = "Most transactions are small, few are large - typical business pattern"
    elif distribution_type == "Time on Page (Log-normal)":
        population = np.random.lognormal(mean=4, sigma=1, size=pop_size)
        pop_name = "Time on Page Distribution"
        business_context = "Short sessions are common, long sessions are rare"
    elif distribution_type == "Uniform":
        population = np.random.uniform(0, 1, size=pop_size)
        pop_name = "Uniform Distribution"
        business_context = "Equal probability across all values - rare in business"
    else:  # Normal
        population = np.random.normal(0.5, 0.15, size=pop_size)
        pop_name = "Normal Distribution"
        business_context = "Bell curve - some business metrics follow this pattern"
    
    # Calculate sample means
    sample_means = []
    for _ in range(num_samples):
        sample = np.random.choice(population, size=sample_size, replace=True)
        sample_means.append(np.mean(sample))
    
    sample_means = np.array(sample_means)
    
    # Create visualization
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            f"Population: {pop_name}",
            f"Sample Means Distribution (n={sample_size})",
            "CLT in Action: Means Converge",
            "Standard Error Decreases with √n"
        ]
    )
    
    # Population distribution
    fig.add_trace(
        go.Histogram(x=population[:2000], nbinsx=50, name="Population", opacity=0.7),
        row=1, col=1
    )
    
    # Sample means distribution
    fig.add_trace(
        go.Histogram(x=sample_means, nbinsx=40, name="Sample Means", opacity=0.7),
        row=1, col=2
    )
    
    # Add normal overlay
    mean_of_means = np.mean(sample_means)
    std_of_means = np.std(sample_means)
    x_norm = np.linspace(sample_means.min(), sample_means.max(), 100)
    y_norm = stats.norm.pdf(x_norm, mean_of_means, std_of_means)
    y_norm_scaled = y_norm * len(sample_means) * (sample_means.max() - sample_means.min()) / 40
    
    fig.add_trace(
        go.Scatter(x=x_norm, y=y_norm_scaled, mode='lines', name='Normal Fit', 
                  line=dict(color='red', width=3)),
        row=1, col=2
    )
    
    # CLT convergence
    sample_sizes = range(5, 101, 5)
    means_convergence = []
    se_convergence = []
    theoretical_se = []
    
    pop_mean = np.mean(population)
    pop_std = np.std(population)
    
    for n in sample_sizes:
        temp_means = [np.mean(np.random.choice(population, size=n)) for _ in range(200)]
        means_convergence.append(np.mean(temp_means))
        se_convergence.append(np.std(temp_means))
        theoretical_se.append(pop_std / np.sqrt(n))
    
    # Mean convergence
    fig.add_trace(
        go.Scatter(x=list(sample_sizes), y=means_convergence, mode='lines+markers', 
                  name='Sample Means', line=dict(color='blue')),
        row=2, col=1
    )
    fig.add_hline(y=pop_mean, line_dash="dash", line_color="red", row=2, col=1)
    
    # SE convergence
    fig.add_trace(
        go.Scatter(x=list(sample_sizes), y=se_convergence, mode='lines+markers', 
                  name='Empirical SE', line=dict(color='blue')),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=list(sample_sizes), y=theoretical_se, mode='lines', 
                  name='Theoretical SE = σ/√n', line=dict(color='red', dash='dash')),
        row=2, col=2
    )
    
    fig.update_layout(height=700, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)
    
    # Key statistics and business relevance
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>📊 Statistical Summary</h4>
            <p><strong>Population Mean:</strong> {pop_mean:.4f}</p>
            <p><strong>Sample Means Average:</strong> {mean_of_means:.4f}</p>
            <p><strong>Population Std:</strong> {pop_std:.4f}</p>
            <p><strong>Sample Means Std:</strong> {std_of_means:.4f}</p>
            <p><strong>Theoretical SE:</strong> {pop_std/np.sqrt(sample_size):.4f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Test normality
        if len(sample_means) > 10:
            _, p_value = stats.shapiro(sample_means[:5000] if len(sample_means) > 5000 else sample_means)
            st.markdown(f"""
            <div class="metric-card">
                <h4>🧪 Normality Test</h4>
                <p><strong>Shapiro-Wilk p-value:</strong> {p_value:.4f}</p>
                <p><strong>Normal?</strong> {"✅ Yes" if p_value > 0.05 else "⚠️ Borderline"}</p>
                <p><strong>CLT Working:</strong> {"✅ Confirmed" if p_value > 0.05 else "📈 Need larger n"}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Business applications
    st.markdown(f"""
    <div class="insight-box">
        <h4>💼 Business Applications</h4>
        <p><strong>Context:</strong> {business_context}</p>
        <p><strong>A/B Testing:</strong> Even with highly variable individual behavior, sample means 
        become normally distributed, enabling reliable statistical tests.</p>
        <p><strong>Confidence Intervals:</strong> With n={sample_size}, our margin of error is ±{1.96 * std_of_means:.4f} 
        at 95% confidence. Larger samples = smaller margins of error.</p>
        <p><strong>Business Metrics:</strong> Daily averages, conversion rates, and revenue metrics 
        all become predictably normal at scale, regardless of individual user variability.</p>
        <p><strong>Sample Size Planning:</strong> To halve your margin of error, you need 4× the sample size.</p>
    </div>
    """, unsafe_allow_html=True)

def show_distribution_analysis(df):
    """Analyze revenue and time distributions"""
    
    st.header("📊 Revenue & Time Distribution Analysis")
    
    # Distribution comparison
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=["Revenue Distribution", "Time Distribution", 
                       "Log-Revenue Distribution", "Revenue by Segment"]
    )
    
    # Revenue distribution
    fig.add_trace(
        go.Histogram(x=df['revenue'], nbinsx=50, name="Revenue", opacity=0.7),
        row=1, col=1
    )
    
    # Time distribution
    fig.add_trace(
        go.Histogram(x=df['time_minutes'], nbinsx=50, name="Time", opacity=0.7),
        row=1, col=2
    )
    
    # Log-revenue distribution
    fig.add_trace(
        go.Histogram(x=np.log(df['revenue']), nbinsx=50, name="Log-Revenue", opacity=0.7),
        row=2, col=1
    )
    
    # Revenue by segment
    for segment in df['segment'].unique():
        segment_data = df[df['segment'] == segment]
        fig.add_trace(
            go.Histogram(x=segment_data['revenue'], nbinsx=30, name=f"{segment} Revenue", 
                        opacity=0.6, legendgroup="segments"),
            row=2, col=2
        )
    
    fig.update_layout(height=700)
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistical tests
    st.subheader("📈 Distribution Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Revenue stats
        revenue_skew = stats.skew(df['revenue'])
        revenue_kurt = stats.kurtosis(df['revenue'])
        
        st.markdown(f"""
        <div class="metric-card">
            <h4>Revenue Distribution</h4>
            <p><strong>Mean:</strong> ${df['revenue'].mean():.4f}</p>
            <p><strong>Median:</strong> ${df['revenue'].median():.4f}</p>
            <p><strong>Skewness:</strong> {revenue_skew:.3f}</p>
            <p><strong>Kurtosis:</strong> {revenue_kurt:.3f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Time stats
        time_skew = stats.skew(df['time_minutes'])
        time_kurt = stats.kurtosis(df['time_minutes'])
        
        st.markdown(f"""
        <div class="metric-card">
            <h4>Time Distribution</h4>
            <p><strong>Mean:</strong> {df['time_minutes'].mean():.2f} min</p>
            <p><strong>Median:</strong> {df['time_minutes'].median():.2f} min</p>
            <p><strong>Skewness:</strong> {time_skew:.3f}</p>
            <p><strong>Kurtosis:</strong> {time_kurt:.3f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # Normality tests
        _, rev_p = stats.normaltest(df['revenue'])
        _, time_p = stats.normaltest(df['time_minutes'])
        
        st.markdown(f"""
        <div class="metric-card">
            <h4>Normality Tests</h4>
            <p><strong>Revenue p-value:</strong> {rev_p:.2e}</p>
            <p><strong>Time p-value:</strong> {time_p:.2e}</p>
            <p><strong>Revenue Normal:</strong> {"❌" if rev_p < 0.05 else "✅"}</p>
            <p><strong>Time Normal:</strong> {"❌" if time_p < 0.05 else "✅"}</p>
        </div>
        """, unsafe_allow_html=True)

def show_interactive_explorer(df):
    """Interactive data exploration tool"""
    
    st.header("🔍 Interactive Data Explorer")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_segments = st.multiselect("Device Segments", df['segment'].unique(), 
                                         default=df['segment'].unique())
    
    with col2:
        time_range = st.slider("Time Range (minutes)", 
                              float(df['time_minutes'].min()), 
                              float(df['time_minutes'].max()),
                              (float(df['time_minutes'].quantile(0.1)), 
                               float(df['time_minutes'].quantile(0.9))))
    
    with col3:
        plot_type = st.selectbox("Visualization", 
                                ["Scatter Plot", "Box Plot", "Violin Plot", "Heatmap"])
    
    # Filter data
    filtered_df = df[
        (df['segment'].isin(selected_segments)) &
        (df['time_minutes'] >= time_range[0]) &
        (df['time_minutes'] <= time_range[1])
    ]
    
    # Display filtered metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Filtered Records", f"{len(filtered_df):,}")
    with col2:
        st.metric("Avg Revenue", f"${filtered_df['revenue'].mean():.4f}")
    with col3:
        corr = filtered_df[['revenue', 'time_minutes']].corr().iloc[0, 1]
        st.metric("Correlation", f"{corr:.3f}")
    
    # Create visualization based on selection
    if plot_type == "Scatter Plot":
        fig = px.scatter(filtered_df.sample(min(2000, len(filtered_df))), 
                        x='time_minutes', y='revenue', color='segment',
                        title="Revenue vs Time by Segment")
    elif plot_type == "Box Plot":
        fig = px.box(filtered_df, x='segment', y='revenue', 
                    title="Revenue Distribution by Segment")
    elif plot_type == "Violin Plot":
        fig = px.violin(filtered_df, x='segment', y='time_minutes', 
                       title="Time Distribution by Segment")
    else:  # Heatmap
        pivot_data = filtered_df.groupby(['segment', 
                                        pd.cut(filtered_df['time_minutes'], bins=10)])['revenue'].mean().unstack()
        fig = px.imshow(pivot_data, title="Revenue Heatmap: Segment vs Time Bins")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary statistics
    if st.checkbox("Show Detailed Statistics"):
        st.subheader("📊 Filtered Data Summary")
        st.dataframe(filtered_df.describe(), use_container_width=True)

if __name__ == "__main__":
    main()
