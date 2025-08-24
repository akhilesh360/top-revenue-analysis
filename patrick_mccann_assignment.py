#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Patrick McCann Assignment - Time on Page vs Revenue Analysis
Data Scientist Position - Raptive
Author: Sai Akhilesh Veldi
Date: August 24, 2025

This script creates a comprehensive analysis of the relationship between 
time on page (top) and revenue, designed for mixed technical/non-technical audiences.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import warnings
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore')

# Set style for professional plots
plt.style.use('default')
sns.set_palette("husl")

class PatrickMcCannAnalysis:
    """Complete analysis class for Patrick McCann assignment"""
    
    def __init__(self, data_path):
        """Initialize with data loading"""
        self.df = pd.read_csv(data_path)
        self.prepare_data()
    
    def prepare_data(self):
        """Clean and prepare data for analysis"""
        print("📊 Data Overview:")
        print(f"Dataset shape: {self.df.shape}")
        print(f"Variables: {list(self.df.columns)}")
        print(f"Missing values: {self.df.isnull().sum().sum()}")
        
        # Basic statistics
        print("\n📈 Key Statistics:")
        print(f"Revenue range: ${self.df['revenue'].min():.4f} - ${self.df['revenue'].max():.4f}")
        print(f"Time on page range: {self.df['top'].min():.1f} - {self.df['top'].max():.1f} seconds")
        
    def create_executive_summary_plot(self):
        """Create main relationship visualization"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Time on Page vs Revenue: Executive Summary', fontsize=20, fontweight='bold', y=0.98)
        
        # 1. Main scatter plot with trend line
        ax1.scatter(self.df['top'], self.df['revenue'], alpha=0.6, s=30, color='steelblue')
        
        # Add trend line
        z = np.polyfit(self.df['top'], self.df['revenue'], 1)
        p = np.poly1d(z)
        ax1.plot(self.df['top'], p(self.df['top']), "r--", linewidth=2, label=f'Trend: y={z[0]:.4f}x+{z[1]:.4f}')
        
        ax1.set_xlabel('Time on Page (seconds)', fontsize=12)
        ax1.set_ylabel('Revenue ($)', fontsize=12)
        ax1.set_title('A. Overall Relationship: Weak Positive Correlation', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Calculate correlation
        correlation = self.df['top'].corr(self.df['revenue'])
        ax1.text(0.05, 0.95, f'Correlation: r = {correlation:.3f}', transform=ax1.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7), fontsize=11)
        
        # 2. Revenue distribution
        ax2.hist(self.df['revenue'], bins=50, alpha=0.7, color='green', edgecolor='black')
        ax2.axvline(self.df['revenue'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: ${self.df["revenue"].mean():.4f}')
        ax2.axvline(self.df['revenue'].median(), color='orange', linestyle='--', linewidth=2, label=f'Median: ${self.df["revenue"].median():.4f}')
        ax2.set_xlabel('Revenue ($)', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('B. Revenue Distribution: Right-Skewed', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Time on page distribution
        ax3.hist(self.df['top'], bins=50, alpha=0.7, color='purple', edgecolor='black')
        ax3.axvline(self.df['top'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {self.df["top"].mean():.1f}s')
        ax3.axvline(self.df['top'].median(), color='orange', linestyle='--', linewidth=2, label=f'Median: {self.df["top"].median():.1f}s')
        ax3.set_xlabel('Time on Page (seconds)', fontsize=12)
        ax3.set_ylabel('Frequency', fontsize=12)
        ax3.set_title('C. Time Distribution: Normal-like', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Binned analysis for clarity
        bins = pd.cut(self.df['top'], bins=10)
        binned_revenue = self.df.groupby(bins)['revenue'].mean()
        bin_centers = [interval.mid for interval in binned_revenue.index]
        
        ax4.bar(range(len(binned_revenue)), binned_revenue.values, alpha=0.7, color='coral')
        ax4.set_xlabel('Time on Page Bins (seconds)', fontsize=12)
        ax4.set_ylabel('Average Revenue ($)', fontsize=12)
        ax4.set_title('D. Revenue by Time Bins: Clear Upward Trend', fontsize=14, fontweight='bold')
        ax4.set_xticks(range(len(binned_revenue)))
        ax4.set_xticklabels([f'{center:.1f}' for center in bin_centers], rotation=45)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def statistical_analysis(self):
        """Perform detailed statistical analysis"""
        results = {}
        
        # Basic correlation
        correlation = self.df['top'].corr(self.df['revenue'])
        results['correlation'] = correlation
        
        # Statistical significance test
        corr_coef, p_value = stats.pearsonr(self.df['top'], self.df['revenue'])
        results['p_value'] = p_value
        results['significant'] = p_value < 0.05
        
        # Linear regression
        X = self.df[['top']]
        y = self.df['revenue']
        
        model = LinearRegression()
        model.fit(X, y)
        
        results['slope'] = model.coef_[0]
        results['intercept'] = model.intercept_
        results['r_squared'] = r2_score(y, model.predict(X))
        
        # Effect size interpretation
        if abs(correlation) < 0.1:
            effect_size = "Negligible"
        elif abs(correlation) < 0.3:
            effect_size = "Small"
        elif abs(correlation) < 0.5:
            effect_size = "Medium"
        else:
            effect_size = "Large"
        
        results['effect_size'] = effect_size
        
        return results
    
    def controlled_analysis(self):
        """Analyze relationship controlling for other variables"""
        # Create dummy variables for categorical features
        df_encoded = pd.get_dummies(self.df, columns=['browser', 'platform', 'site'])
        
        # Multiple regression with controls
        feature_cols = [col for col in df_encoded.columns if col not in ['revenue']]
        X_controlled = df_encoded[feature_cols]
        y = df_encoded['revenue']
        
        model_controlled = LinearRegression()
        model_controlled.fit(X_controlled, y)
        
        # Get coefficient for 'top' variable
        top_index = feature_cols.index('top')
        controlled_coefficient = model_controlled.coef_[top_index]
        
        # R-squared comparison
        r2_controlled = r2_score(y, model_controlled.predict(X_controlled))
        
        # Simple regression for comparison
        X_simple = df_encoded[['top']]
        model_simple = LinearRegression()
        model_simple.fit(X_simple, y)
        r2_simple = r2_score(y, model_simple.predict(X_simple))
        
        return {
            'controlled_coefficient': controlled_coefficient,
            'simple_coefficient': model_simple.coef_[0],
            'r2_controlled': r2_controlled,
            'r2_simple': r2_simple,
            'feature_importance': dict(zip(feature_cols, model_controlled.coef_))
        }
    
    def create_controlled_analysis_plot(self, controlled_results):
        """Create visualization for controlled analysis"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Controlled Analysis: How Other Variables Affect the Relationship', fontsize=18, fontweight='bold')
        
        # 1. Coefficient comparison
        coefficients = [controlled_results['simple_coefficient'], controlled_results['controlled_coefficient']]
        labels = ['Simple\n(TOP only)', 'Controlled\n(All variables)']
        colors = ['lightblue', 'lightcoral']
        
        bars = ax1.bar(labels, coefficients, color=colors, alpha=0.8, edgecolor='black')
        ax1.set_ylabel('Coefficient Value', fontsize=12)
        ax1.set_title('A. Time-Revenue Relationship: Simple vs Controlled', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, coef in zip(bars, coefficients):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{coef:.5f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. R-squared comparison
        r2_values = [controlled_results['r2_simple'], controlled_results['r2_controlled']]
        bars2 = ax2.bar(labels, r2_values, color=['lightgreen', 'gold'], alpha=0.8, edgecolor='black')
        ax2.set_ylabel('R-squared', fontsize=12)
        ax2.set_title('B. Model Explanatory Power Comparison', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        for bar, r2 in zip(bars2, r2_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{r2:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Revenue by browser
        browser_revenue = self.df.groupby('browser')['revenue'].agg(['mean', 'std']).reset_index()
        x_pos = range(len(browser_revenue))
        ax3.bar(x_pos, browser_revenue['mean'], yerr=browser_revenue['std'], 
               capsize=5, alpha=0.7, color='skyblue', edgecolor='black')
        ax3.set_xlabel('Browser', fontsize=12)
        ax3.set_ylabel('Average Revenue ($)', fontsize=12)
        ax3.set_title('C. Revenue Varies by Browser', fontsize=14, fontweight='bold')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(browser_revenue['browser'])
        ax3.grid(True, alpha=0.3)
        
        # 4. Revenue by platform
        platform_revenue = self.df.groupby('platform')['revenue'].agg(['mean', 'std']).reset_index()
        x_pos = range(len(platform_revenue))
        ax4.bar(x_pos, platform_revenue['mean'], yerr=platform_revenue['std'], 
               capsize=5, alpha=0.7, color='lightcoral', edgecolor='black')
        ax4.set_xlabel('Platform', fontsize=12)
        ax4.set_ylabel('Average Revenue ($)', fontsize=12)
        ax4.set_title('D. Revenue Varies by Platform', fontsize=14, fontweight='bold')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(platform_revenue['platform'])
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_business_insights_plot(self, stats_results, controlled_results):
        """Create business-focused insights visualization"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Business Insights: What This Means for Revenue Strategy', fontsize=18, fontweight='bold')
        
        # 1. Revenue prediction based on time ranges
        time_ranges = [(0, 10), (10, 20), (20, 30), (30, 50)]
        range_labels = ['0-10s', '10-20s', '20-30s', '30s+']
        range_revenues = []
        range_counts = []
        
        for min_time, max_time in time_ranges:
            mask = (self.df['top'] >= min_time) & (self.df['top'] < max_time)
            if min_time == 30:  # Last range
                mask = self.df['top'] >= min_time
            
            range_revenues.append(self.df[mask]['revenue'].mean())
            range_counts.append(mask.sum())
        
        bars = ax1.bar(range_labels, range_revenues, alpha=0.8, color='gold', edgecolor='black')
        ax1.set_xlabel('Time on Page Range', fontsize=12)
        ax1.set_ylabel('Average Revenue ($)', fontsize=12)
        ax1.set_title('A. Revenue Optimization: Target Longer Sessions', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add count labels
        for bar, count in zip(bars, range_counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'n={count}', ha='center', va='bottom', fontsize=10)
        
        # 2. Platform strategy
        platform_stats = self.df.groupby('platform').agg({
            'revenue': ['mean', 'count'],
            'top': 'mean'
        }).round(4)
        platform_stats.columns = ['avg_revenue', 'count', 'avg_time']
        platform_stats = platform_stats.reset_index()
        
        width = 0.35
        x = np.arange(len(platform_stats))
        
        ax2_twin = ax2.twinx()
        bars1 = ax2.bar(x - width/2, platform_stats['avg_revenue'], width, 
                       label='Avg Revenue', alpha=0.8, color='lightblue')
        bars2 = ax2_twin.bar(x + width/2, platform_stats['avg_time'], width, 
                            label='Avg Time', alpha=0.8, color='lightcoral')
        
        ax2.set_xlabel('Platform', fontsize=12)
        ax2.set_ylabel('Average Revenue ($)', fontsize=12, color='blue')
        ax2_twin.set_ylabel('Average Time (seconds)', fontsize=12, color='red')
        ax2.set_title('B. Platform Performance: Mobile vs Desktop', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(platform_stats['platform'])
        
        # 3. Feature importance from controlled model
        importance_data = controlled_results['feature_importance']
        # Focus on most important features
        top_features = sorted(importance_data.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
        
        feature_names = [item[0] for item in top_features]
        feature_values = [item[1] for item in top_features]
        colors = ['red' if val < 0 else 'green' for val in feature_values]
        
        bars = ax3.barh(range(len(feature_names)), feature_values, color=colors, alpha=0.7)
        ax3.set_yticks(range(len(feature_names)))
        ax3.set_yticklabels(feature_names)
        ax3.set_xlabel('Coefficient (Impact on Revenue)', fontsize=12)
        ax3.set_title('C. Key Drivers: What Actually Affects Revenue', fontsize=14, fontweight='bold')
        ax3.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax3.grid(True, alpha=0.3)
        
        # 4. Key metrics summary
        ax4.axis('off')
        
        # Create text summary
        summary_text = f"""
        📊 KEY FINDINGS SUMMARY
        
        ✓ TIME-REVENUE RELATIONSHIP:
          • Correlation: {stats_results['correlation']:.3f} ({stats_results['effect_size']} effect)
          • Statistical significance: {'YES' if stats_results['significant'] else 'NO'} (p={stats_results['p_value']:.3f})
          • For every 1 second ↑ in time: ${stats_results['slope']:.5f} ↑ in revenue
        
        ✓ CONTROLLING FOR OTHER VARIABLES:
          • Simple model R²: {controlled_results['r2_simple']:.3f}
          • Full model R²: {controlled_results['r2_controlled']:.3f}
          • Improvement: {((controlled_results['r2_controlled'] - controlled_results['r2_simple']) / controlled_results['r2_simple'] * 100):.1f}%
        
        ✓ BUSINESS RECOMMENDATIONS:
          • Focus on engagement strategies for 30+ second sessions
          • Platform differences exist - optimize mobile experience
          • Time on page matters, but other factors are important too
        
        📈 REVENUE IMPACT:
          • Users with 30+ seconds: ${max(range_revenues):.4f} avg
          • Users with <10 seconds: ${range_revenues[0]:.4f} avg
          • Potential uplift: {((max(range_revenues) - range_revenues[0]) / range_revenues[0] * 100):.1f}%
        """
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def generate_report(self, output_file='patrick_mccann_revenue_analysis.pdf'):
        """Generate complete PDF report"""
        print("🎯 Generating comprehensive analysis for Patrick McCann...")
        
        # Perform analyses
        stats_results = self.statistical_analysis()
        controlled_results = self.controlled_analysis()
        
        with PdfPages(output_file) as pdf:
            # Page 1: Executive Summary
            fig1 = self.create_executive_summary_plot()
            pdf.savefig(fig1, bbox_inches='tight', dpi=300)
            plt.close(fig1)
            
            # Page 2: Controlled Analysis
            fig2 = self.create_controlled_analysis_plot(controlled_results)
            pdf.savefig(fig2, bbox_inches='tight', dpi=300)
            plt.close(fig2)
            
            # Page 3: Business Insights
            fig3 = self.create_business_insights_plot(stats_results, controlled_results)
            pdf.savefig(fig3, bbox_inches='tight', dpi=300)
            plt.close(fig3)
        
        print(f"✅ Report generated: {output_file}")
        
        # Print key findings
        print("\n🎯 KEY FINDINGS FOR PATRICK MCCANN:")
        print("="*50)
        print(f"📊 Time-Revenue Correlation: {stats_results['correlation']:.3f} ({stats_results['effect_size']} effect)")
        print(f"🔬 Statistical Significance: {'YES' if stats_results['significant'] else 'NO'} (p={stats_results['p_value']:.3f})")
        print(f"📈 Revenue Impact: ${stats_results['slope']:.5f} per additional second")
        print(f"🎛️ Model Performance: R² improves from {controlled_results['r2_simple']:.3f} to {controlled_results['r2_controlled']:.3f} with controls")
        print(f"💼 Business Impact: {((controlled_results['r2_controlled'] - controlled_results['r2_simple']) / controlled_results['r2_simple'] * 100):.1f}% improvement with full model")

def main():
    """Main execution"""
    print("🚀 Patrick McCann Assignment - Time on Page vs Revenue Analysis")
    print("="*60)
    
    # Load and analyze data
    analyzer = PatrickMcCannAnalysis('data/testdata.csv')
    
    # Generate comprehensive report
    analyzer.generate_report('patrick_mccann_revenue_analysis.pdf')
    
    print("\n✅ Analysis complete! Ready for Patrick McCann review.")

if __name__ == "__main__":
    main()
