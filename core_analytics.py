#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core Analytics Module - Reusable Data Analysis Functions
Built for Patrick McCann, SVP Research @ Raptive

Consolidated, optimized analytics functions for revenue analysis.
Removes redundancy and provides reusable components.

Author: Sai Akhilesh Veldi
Date: August 2025
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from typing import Dict, Tuple, Optional, Any
import warnings

warnings.filterwarnings('ignore')

# Configuration Constants
class Config:
    """Centralized configuration"""
    DATA_FILE = "data/testdata.csv"
    DEFAULT_SEED = 42
    BOOTSTRAP_SAMPLES = 1000
    CONFIDENCE_LEVEL = 0.95
    
    # Color schemes
    COLORS = {
        'primary': '#1f77b4',
        'secondary': '#ff7f0e', 
        'success': '#2ca02c',
        'warning': '#ff7f0e',
        'danger': '#d62728',
        'platforms': ['#2ca02c', '#ff7f0e'],
        'browsers': ['#4285f4', '#ff6b35']
    }

class DataLoader:
    """Centralized data loading and validation"""
    
    @staticmethod
    def load_revenue_data() -> Tuple[Optional[pd.DataFrame], Optional[np.ndarray]]:
        """Load and validate revenue data from testdata.csv"""
        try:
            df = pd.read_csv(Config.DATA_FILE)
            
            # Validate required columns
            required_cols = ['revenue', 'platform', 'browser', 'site']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Clean and validate data
            df = df.dropna(subset=['revenue'])
            df = df[df['revenue'] > 0]  # Remove invalid revenue values
            
            revenue_data = df['revenue'].values
            
            return df, revenue_data
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None, None

class StatisticalAnalyzer:
    """Core statistical analysis functions"""
    
    @staticmethod
    def calculate_metrics(data: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive statistical metrics"""
        metrics = {
            # Basic statistics
            'count': len(data),
            'mean': np.mean(data),
            'median': np.median(data),
            'std': np.std(data),
            'min': np.min(data),
            'max': np.max(data),
            'total_revenue': np.sum(data),
            
            # Percentiles
            'q25': np.percentile(data, 25),
            'q75': np.percentile(data, 75),
            'q90': np.percentile(data, 90),
            'q95': np.percentile(data, 95),
            'q99': np.percentile(data, 99),
            
            # Advanced metrics
            'skewness': stats.skew(data),
            'kurtosis': stats.kurtosis(data),
            'cv': np.std(data) / np.mean(data),  # Coefficient of variation
        }
        
        # Derived metrics
        metrics['iqr'] = metrics['q75'] - metrics['q25']
        metrics['mean_vs_median_diff'] = ((metrics['mean'] - metrics['median']) / metrics['median']) * 100
        
        return metrics
    
    @staticmethod
    def create_confidence_intervals(data: np.ndarray, n_bootstrap: int = None, confidence: float = None) -> Dict[str, Any]:
        """Create bootstrap confidence intervals"""
        n_bootstrap = n_bootstrap or Config.BOOTSTRAP_SAMPLES
        confidence = confidence or Config.CONFIDENCE_LEVEL
        
        np.random.seed(Config.DEFAULT_SEED)
        
        # Bootstrap sampling
        bootstrap_means = []
        bootstrap_medians = []
        bootstrap_p90s = []
        
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, len(data), replace=True)
            bootstrap_means.append(np.mean(sample))
            bootstrap_medians.append(np.median(sample))
            bootstrap_p90s.append(np.percentile(sample, 90))
        
        # Calculate confidence intervals
        alpha = 1 - confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        return {
            'mean_ci': [np.percentile(bootstrap_means, lower_percentile), 
                       np.percentile(bootstrap_means, upper_percentile)],
            'median_ci': [np.percentile(bootstrap_medians, lower_percentile), 
                         np.percentile(bootstrap_medians, upper_percentile)],
            'p90_ci': [np.percentile(bootstrap_p90s, lower_percentile), 
                      np.percentile(bootstrap_p90s, upper_percentile)],
            'bootstrap_means': bootstrap_means,
            'bootstrap_medians': bootstrap_medians,
            'confidence_level': confidence
        }
    
    @staticmethod
    def analyze_business_pattern(metrics: Dict[str, float]) -> Dict[str, Any]:
        """Analyze business patterns with actionable insights"""
        skewness = metrics['skewness']
        mean_median_diff = metrics['mean_vs_median_diff']
        
        if skewness > 2:
            pattern_type = "🔥 High-Concentration Revenue"
            risk_level = "High Opportunity"
            color = Config.COLORS['danger']
            business_interpretation = "Top performers drive most revenue - focus on retention and replication"
            strategic_recommendation = "Develop premium customer success programs and analyze top performer characteristics"
        elif skewness > 1:
            pattern_type = "📊 Moderate Revenue Concentration"
            risk_level = "Medium Opportunity"
            color = Config.COLORS['warning']
            business_interpretation = "Clear revenue segments exist with balanced risk-reward profile"
            strategic_recommendation = "Implement tiered optimization strategies for different user segments"
        else:
            pattern_type = "✅ Balanced Revenue Distribution"
            risk_level = "Stable Pattern"
            color = Config.COLORS['success']
            business_interpretation = "Revenue is evenly distributed with predictable, stable patterns"
            strategic_recommendation = "Focus on overall user experience improvements and consistent optimization"
        
        return {
            'pattern_type': pattern_type,
            'risk_level': risk_level,
            'color': color,
            'business_interpretation': business_interpretation,
            'strategic_recommendation': strategic_recommendation,
            'skewness': skewness,
            'explanation': f"""
            **Business Impact:**
            - Mean revenue: ${metrics['mean']:.2f}
            - Typical revenue: ${metrics['median']:.2f}
            - Difference: {mean_median_diff:.1f}%
            - Top 5% threshold: ${metrics['q95']:.2f}
            """
        }

class DistributionGenerator:
    """Generate different statistical distributions for education"""
    
    @staticmethod
    def generate_distribution(dist_type: str, n_samples: int = 1000, seed: int = 42) -> np.ndarray:
        """Generate data from different distribution types"""
        np.random.seed(seed)
        
        if dist_type == "Normal":
            # Normal distribution - symmetric, light tails
            return np.random.normal(loc=100, scale=20, size=n_samples)
        
        elif dist_type == "Lognormal":
            # Lognormal - heavy right tail (common in revenue/wealth)
            return np.random.lognormal(mean=4.5, sigma=0.5, size=n_samples)
        
        elif dist_type == "Pareto":
            # Pareto - extremely heavy tail (80/20 rule)
            return np.random.pareto(a=1.5, size=n_samples) * 50 + 10
        
        elif dist_type == "Exponential":
            # Exponential - moderate heavy tail
            return np.random.exponential(scale=50, size=n_samples)
        
        else:
            # Default to normal
            return np.random.normal(loc=100, scale=20, size=n_samples)
    
    @staticmethod
    def get_distribution_info(dist_type: str) -> Dict[str, str]:
        """Get educational information about each distribution"""
        info = {
            "Normal": {
                "description": "Symmetric bell curve - most values cluster around the average",
                "real_world": "Height, test scores, measurement errors",
                "business_insight": "Predictable patterns, traditional statistics work well",
                "tail_behavior": "Light tails - extreme values are rare"
            },
            "Lognormal": {
                "description": "Right-skewed - a few large values pull the average up",
                "real_world": "Income, stock prices, city sizes",
                "business_insight": "Revenue often follows this pattern - some customers generate much more",
                "tail_behavior": "Heavy right tail - big winners exist"
            },
            "Pareto": {
                "description": "Extreme inequality - tiny fraction drives most outcomes",
                "real_world": "Wealth distribution, website traffic, book sales",
                "business_insight": "The famous 80/20 rule - 20% of customers = 80% of revenue",
                "tail_behavior": "Very heavy tail - extreme concentration"
            },
            "Exponential": {
                "description": "Moderate skew - decreasing probability of larger values",
                "real_world": "Time between events, customer lifetime",
                "business_insight": "Most customers are small, but some big ones matter a lot",
                "tail_behavior": "Moderate heavy tail"
            }
        }
        return info.get(dist_type, info["Normal"])

class VisualizationEngine:
    """Enhanced visualization functions for statistical education"""
    
    @staticmethod
    def create_distribution_comparison(data: np.ndarray, dist_type: str, show_metrics: bool = True) -> go.Figure:
        """Create educational distribution visualization with mean/median lines"""
        fig = go.Figure()
        
        # Calculate key metrics
        mean_val = np.mean(data)
        median_val = np.median(data)
        
        # Main histogram
        fig.add_trace(go.Histogram(
            x=data,
            nbinsx=50,
            name=f"{dist_type} Distribution",
            opacity=0.7,
            marker_color=Config.COLORS['primary'],
            hovertemplate="Value: %{x:.1f}<br>Count: %{y}<extra></extra>"
        ))
        
        if show_metrics:
            # Add mean line
            fig.add_vline(
                x=mean_val,
                line_dash="dash",
                line_color=Config.COLORS['danger'],
                annotation_text=f"Mean: {mean_val:.1f}",
                annotation_position="top"
            )
            
            # Add median line
            fig.add_vline(
                x=median_val,
                line_dash="dot",
                line_color=Config.COLORS['success'],
                annotation_text=f"Median: {median_val:.1f}",
                annotation_position="bottom"
            )
        
        fig.update_layout(
            title=f"{dist_type} Distribution - Notice the Shape!",
            xaxis_title="Value",
            yaxis_title="Frequency",
            template="plotly_white",
            showlegend=False,
            height=400
        )
        
        return fig
    
    @staticmethod
    def create_metrics_stability_chart(data: np.ndarray, n_bootstrap: int = 100) -> go.Figure:
        """Show how different metrics behave under resampling"""
        np.random.seed(Config.DEFAULT_SEED)
        
        means = []
        medians = []
        trimmed_means = []
        
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=len(data)//2, replace=True)
            means.append(np.mean(sample))
            medians.append(np.median(sample))
            # 10% trimmed mean (remove extreme 10% from each end)
            trimmed_means.append(stats.trim_mean(sample, 0.1))
        
        fig = go.Figure()
        
        # Box plots for each metric
        fig.add_trace(go.Box(
            y=means,
            name="Sample Mean",
            marker_color=Config.COLORS['danger'],
            boxmean=True
        ))
        
        fig.add_trace(go.Box(
            y=medians,
            name="Sample Median",
            marker_color=Config.COLORS['success'],
            boxmean=True
        ))
        
        fig.add_trace(go.Box(
            y=trimmed_means,
            name="Trimmed Mean",
            marker_color=Config.COLORS['warning'],
            boxmean=True
        ))
        
        fig.update_layout(
            title="Metric Stability Under Resampling",
            yaxis_title="Metric Value",
            template="plotly_white",
            height=400
        )
        
        return fig
    
    @staticmethod
    def create_outlier_impact_demo(base_data: np.ndarray, outlier_multiplier: float = 10) -> go.Figure:
        """Demonstrate impact of adding outliers"""
        # Original data
        original_mean = np.mean(base_data)
        original_median = np.median(base_data)
        
        # Add some outliers
        n_outliers = max(1, len(base_data) // 100)  # 1% outliers
        outliers = np.random.choice(base_data, n_outliers) * outlier_multiplier
        contaminated_data = np.concatenate([base_data, outliers])
        
        new_mean = np.mean(contaminated_data)
        new_median = np.median(contaminated_data)
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Original Data", f"With {n_outliers} Outliers Added"),
            shared_yaxes=True
        )
        
        # Original data
        fig.add_trace(
            go.Histogram(x=base_data, nbinsx=30, name="Original", marker_color=Config.COLORS['primary']),
            row=1, col=1
        )
        fig.add_vline(x=original_mean, line_dash="dash", line_color=Config.COLORS['danger'], row=1, col=1)
        fig.add_vline(x=original_median, line_dash="dot", line_color=Config.COLORS['success'], row=1, col=1)
        
        # Contaminated data
        fig.add_trace(
            go.Histogram(x=contaminated_data, nbinsx=30, name="With Outliers", marker_color=Config.COLORS['warning']),
            row=1, col=2
        )
        fig.add_vline(x=new_mean, line_dash="dash", line_color=Config.COLORS['danger'], row=1, col=2)
        fig.add_vline(x=new_median, line_dash="dot", line_color=Config.COLORS['success'], row=1, col=2)
        
        fig.update_layout(
            title="How Outliers Affect Different Metrics",
            template="plotly_white",
            showlegend=False,
            height=400
        )
        
        # Add annotations
        fig.add_annotation(
            text=f"Mean: {original_mean:.1f}<br>Median: {original_median:.1f}",
            x=0.25, y=0.95, xref="paper", yref="paper",
            showarrow=False, bgcolor="white", bordercolor="black"
        )
        
        fig.add_annotation(
            text=f"Mean: {new_mean:.1f} (+{((new_mean-original_mean)/original_mean)*100:.1f}%)<br>Median: {new_median:.1f} (+{((new_median-original_median)/original_median)*100:.1f}%)",
            x=0.75, y=0.95, xref="paper", yref="paper",
            showarrow=False, bgcolor="white", bordercolor="black"
        )
        
        return fig
    
    @staticmethod
    def create_lorenz_curve(data: np.ndarray) -> go.Figure:
        """Create Lorenz curve to show inequality"""
        # Sort data and calculate cumulative percentages
        sorted_data = np.sort(data)
        n = len(sorted_data)
        
        # Calculate cumulative proportions
        cum_population = np.arange(1, n + 1) / n
        cum_wealth = np.cumsum(sorted_data) / np.sum(sorted_data)
        
        # Calculate Gini coefficient
        gini = 1 - 2 * np.trapz(cum_wealth, cum_population)
        
        fig = go.Figure()
        
        # Perfect equality line
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Perfect Equality',
            line=dict(dash='dash', color='gray'),
            hovertemplate="Perfect equality line<extra></extra>"
        ))
        
        # Lorenz curve
        fig.add_trace(go.Scatter(
            x=np.concatenate([[0], cum_population]),
            y=np.concatenate([[0], cum_wealth]),
            mode='lines',
            name='Actual Distribution',
            line=dict(color=Config.COLORS['primary'], width=3),
            fill='tonexty',
            fillcolor='rgba(31, 119, 180, 0.2)',
            hovertemplate="Population: %{x:.1%}<br>Wealth: %{y:.1%}<extra></extra>"
        ))
        
        fig.update_layout(
            title=f"Lorenz Curve - Inequality Visualization (Gini: {gini:.3f})",
            xaxis_title="Cumulative Population Proportion",
            yaxis_title="Cumulative Wealth Proportion",
            template="plotly_white",
            height=400
        )
        
        return fig, gini
    
    @staticmethod
    def create_revenue_distribution_chart(data: np.ndarray, title: str = "Revenue Distribution") -> go.Figure:
        """Create revenue distribution histogram"""
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=data,
            nbinsx=50,
            name="Revenue",
            opacity=0.7,
            marker_color=Config.COLORS['primary']
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Revenue ($)",
            yaxis_title="Frequency",
            showlegend=False,
            template="plotly_white"
        )
        
        return fig
    
    @staticmethod
    def create_platform_comparison(df: pd.DataFrame) -> go.Figure:
        """Create platform comparison chart"""
        platform_stats = df.groupby('platform')['revenue'].agg(['mean', 'count']).reset_index()
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=platform_stats['platform'],
            y=platform_stats['mean'],
            name="Average Revenue",
            marker_color=Config.COLORS['platforms'],
            text=[f'${val:.2f}' for val in platform_stats['mean']],
            textposition='outside'
        ))
        
        fig.update_layout(
            title="Average Revenue by Platform",
            xaxis_title="Platform",
            yaxis_title="Average Revenue ($)",
            showlegend=False,
            template="plotly_white"
        )
        
        return fig
    
    @staticmethod
    def create_browser_comparison(df: pd.DataFrame) -> go.Figure:
        """Create browser comparison chart"""
        browser_stats = df.groupby('browser')['revenue'].agg(['mean', 'count']).reset_index()
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=browser_stats['browser'],
            y=browser_stats['mean'],
            name="Average Revenue",
            marker_color=Config.COLORS['browsers'],
            text=[f'${val:.2f}' for val in browser_stats['mean']],
            textposition='outside'
        ))
        
        fig.update_layout(
            title="Average Revenue by Browser",
            xaxis_title="Browser",
            yaxis_title="Average Revenue ($)",
            showlegend=False,
            template="plotly_white"
        )
        
        return fig
    
    @staticmethod
    def create_confidence_interval_chart(confidence_intervals: Dict[str, Any]) -> go.Figure:
        """Create confidence interval visualization"""
        metrics = ['Mean', 'Median']
        lower_bounds = [confidence_intervals['mean_ci'][0], confidence_intervals['median_ci'][0]]
        upper_bounds = [confidence_intervals['mean_ci'][1], confidence_intervals['median_ci'][1]]
        
        fig = go.Figure()
        
        # Add error bars
        fig.add_trace(go.Scatter(
            x=metrics,
            y=[(l + u) / 2 for l, u in zip(lower_bounds, upper_bounds)],
            error_y=dict(
                type='data',
                symmetric=False,
                array=[(u - l) / 2 for l, u in zip(lower_bounds, upper_bounds)],
                arrayminus=[(u - l) / 2 for l, u in zip(lower_bounds, upper_bounds)]
            ),
            mode='markers',
            marker=dict(size=10, color=Config.COLORS['primary']),
            name=f"{confidence_intervals['confidence_level']*100:.0f}% Confidence Interval"
        ))
        
        fig.update_layout(
            title="Statistical Confidence Intervals",
            yaxis_title="Revenue ($)",
            template="plotly_white"
        )
        
        return fig

class SegmentAnalyzer:
    """Analyze different segments of the data"""
    
    @staticmethod
    def analyze_platform_performance(df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze platform-specific performance"""
        platform_stats = df.groupby('platform')['revenue'].agg(['count', 'mean', 'median', 'std']).round(2)
        
        desktop_avg = platform_stats.loc['desktop']['mean'] if 'desktop' in platform_stats.index else 0
        mobile_avg = platform_stats.loc['mobile']['mean'] if 'mobile' in platform_stats.index else 0
        
        performance_diff = abs(desktop_avg - mobile_avg)
        better_platform = "Desktop" if desktop_avg > mobile_avg else "Mobile"
        
        return {
            'stats': platform_stats,
            'better_platform': better_platform,
            'performance_diff': performance_diff,
            'insight': f"{better_platform} generates ${performance_diff:.2f} more revenue per record on average"
        }
    
    @staticmethod
    def analyze_browser_performance(df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze browser-specific performance"""
        browser_stats = df.groupby('browser')['revenue'].agg(['count', 'mean', 'median', 'std']).round(2)
        
        chrome_avg = browser_stats.loc['chrome']['mean'] if 'chrome' in browser_stats.index else 0
        safari_avg = browser_stats.loc['safari']['mean'] if 'safari' in browser_stats.index else 0
        
        performance_diff = abs(chrome_avg - safari_avg)
        better_browser = "Chrome" if chrome_avg > safari_avg else "Safari"
        
        return {
            'stats': browser_stats,
            'better_browser': better_browser,
            'performance_diff': performance_diff,
            'insight': f"{better_browser} users generate ${performance_diff:.2f} more revenue per session"
        }
    
    @staticmethod
    def analyze_top_performers(df: pd.DataFrame, percentile: float = 90) -> Dict[str, Any]:
        """Analyze top performing segments"""
        threshold = np.percentile(df['revenue'], percentile)
        top_performers = df[df['revenue'] >= threshold]
        
        # Analyze characteristics
        platform_dist = top_performers['platform'].value_counts()
        browser_dist = top_performers['browser'].value_counts()
        
        total_revenue = df['revenue'].sum()
        top_revenue = top_performers['revenue'].sum()
        revenue_concentration = (top_revenue / total_revenue) * 100
        
        return {
            'threshold': threshold,
            'count': len(top_performers),
            'platform_preference': platform_dist.index[0] if len(platform_dist) > 0 else "N/A",
            'browser_preference': browser_dist.index[0] if len(browser_dist) > 0 else "N/A",
            'revenue_concentration': revenue_concentration,
            'insight': f"Top {100-percentile}% of records generate {revenue_concentration:.1f}% of total revenue"
        }

class ReportGenerator:
    """Generate insights and recommendations"""
    
    @staticmethod
    def generate_executive_summary(
        metrics: Dict[str, float], 
        pattern_analysis: Dict[str, Any],
        platform_analysis: Dict[str, Any],
        browser_analysis: Dict[str, Any],
        top_performers: Dict[str, Any]
    ) -> Dict[str, str]:
        """Generate comprehensive executive summary"""
        
        return {
            'revenue_overview': f"""
            **Revenue Portfolio Analysis:**
            - Total revenue analyzed: ${metrics['total_revenue']:,.2f}
            - Average deal size: ${metrics['mean']:.2f}
            - Revenue consistency: {pattern_analysis['risk_level']}
            - Pattern type: {pattern_analysis['pattern_type']}
            """,
            
            'platform_insights': f"""
            **Platform Performance:**
            - {platform_analysis['insight']}
            - Strategic focus: Optimize {platform_analysis['better_platform']} experience
            """,
            
            'browser_insights': f"""
            **Browser Analysis:**
            - {browser_analysis['insight']}
            - User behavior: {browser_analysis['better_browser']} users show higher engagement
            """,
            
            'concentration_analysis': f"""
            **Revenue Concentration:**
            - {top_performers['insight']}
            - Top performer preference: {top_performers['platform_preference']} platform, {top_performers['browser_preference']} browser
            """,
            
            'strategic_recommendations': f"""
            **Key Recommendations:**
            1. {pattern_analysis['strategic_recommendation']}
            2. Focus platform optimization on {platform_analysis['better_platform']}
            3. Enhance {browser_analysis['better_browser']} user experience
            4. Develop retention strategies for top {top_performers['revenue_concentration']:.1f}% revenue generators
            """
        }
