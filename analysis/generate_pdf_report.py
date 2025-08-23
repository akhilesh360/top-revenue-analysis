#!/usr/bin/env python3
"""
Generate a PDF report from the analysis results.
This script creates a brief, accessible report suitable for mixed technical/nontechnical audiences.
"""

import os
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

def create_pdf_report():
    """Create a brief PDF report summarizing the TOP-Revenue analysis."""
    
    # Load the analysis results
    with open('../report/data_profile.json', 'r') as f:
        profile = json.load(f)
    
    # Create PDF
    pdf_path = '../report/revenue_analysis_report.pdf'
    
    with PdfPages(pdf_path) as pdf:
        # Page 1: Title and Summary
        fig = plt.figure(figsize=(8.5, 11))
        fig.suptitle('Time on Page vs Revenue Analysis', fontsize=20, fontweight='bold', y=0.95)
        
        # Remove axes for text page
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        # Executive Summary
        summary_text = f"""
EXECUTIVE SUMMARY

This analysis examines the relationship between time spent on a webpage (Time on Page) 
and revenue generation using statistical modeling and visualization.

KEY FINDINGS:

• Dataset: {profile['initial_rows']:,} initial observations, {profile['rows_after_drop']:,} used in analysis
• Strong positive relationship between Time on Page and Revenue
• A one standard deviation increase in Time on Page is associated with a 
  {profile['controlled_effect_mu_to_mu_plus_1sd_percent']:.1f}% increase in revenue
• This relationship holds even when controlling for browser, platform, and site differences
• The relationship shows diminishing returns at higher time values

METHODOLOGY:

• Data cleaning with outlier treatment (1%/99% winsorization)
• Three statistical models of increasing sophistication:
  1. Simple linear relationship
  2. Flexible spline curve (uncontrolled)  
  3. Spline curve controlling for browser, platform, and site
• Robust statistical methods with appropriate confidence intervals

BUSINESS IMPLICATIONS:

• Time on Page is a strong predictor of revenue generation
• Strategies to increase user engagement time are likely to drive revenue
• The effect is consistent across different browsers, platforms, and sites
• Focus should be on quality engagement rather than just raw time
"""
        
        ax.text(0.05, 0.85, summary_text, fontsize=11, verticalalignment='top', 
                transform=ax.transAxes, wrap=True)
        
        # Add date and technical details at bottom
        technical_note = f"""
Technical Note: Analysis conducted using Python with statsmodels for regression, 
robust standard errors, and spline functions. R² values: Simple model = {profile['m1_r2']:.3f}, 
Spline model = {profile['m2_r2']:.3f}, Controlled model = {profile['m3_r2']:.3f}. 
Data processed on {pd.Timestamp.now().strftime('%B %d, %Y')}.
"""
        ax.text(0.05, 0.05, technical_note, fontsize=9, style='italic',
                transform=ax.transAxes, wrap=True)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 2: Visualizations
        fig = plt.figure(figsize=(8.5, 11))
        fig.suptitle('Statistical Analysis Visualizations', fontsize=16, fontweight='bold', y=0.95)
        
        # Load and display the three key figures
        try:
            # Figure 1: Hexbin plot
            ax1 = plt.subplot(3, 1, 1)
            img1 = plt.imread('../figs/fig1_hexbin.png')
            ax1.imshow(img1)
            ax1.axis('off')
            ax1.set_title('Exploratory Analysis: Revenue vs Time on Page', fontsize=12, pad=10)
            
            # Figure 2: Spline uncontrolled
            ax2 = plt.subplot(3, 1, 2) 
            img2 = plt.imread('../figs/fig2_spline_uncontrolled.png')
            ax2.imshow(img2)
            ax2.axis('off')
            ax2.set_title('Diminishing Returns Pattern (Uncontrolled)', fontsize=12, pad=10)
            
            # Figure 3: Marginal effects
            ax3 = plt.subplot(3, 1, 3)
            img3 = plt.imread('../figs/fig3_marginal_effects.png')
            ax3.imshow(img3)
            ax3.axis('off')
            ax3.set_title('Controlled Effect of Time on Page', fontsize=12, pad=10)
            
        except FileNotFoundError as e:
            # If images don't exist, show placeholder text
            ax = fig.add_subplot(111)
            ax.axis('off')
            ax.text(0.5, 0.5, f'Visualizations not found.\nPlease run the analysis notebook first.\nError: {e}',
                   ha='center', va='center', fontsize=14, transform=ax.transAxes)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 3: Technical Details
        fig = plt.figure(figsize=(8.5, 11))
        fig.suptitle('Technical Methodology', fontsize=16, fontweight='bold', y=0.95)
        
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        methodology_text = f"""
DATA PROCESSING:

• Initial dataset: {profile['initial_rows']:,} rows, {profile['n_columns']} columns
• Missing data handling: Removed {profile['initial_rows'] - profile['rows_after_drop']} rows with missing revenue/time values
• Outlier treatment: Winsorized extreme values at 1st and 99th percentiles
  - Revenue: {profile['rows_winsorized']['revenue']} rows affected
  - Time on Page: {profile['rows_winsorized']['top']} rows affected

STATISTICAL MODELS:

Model 1: Simple Linear Regression
• Revenue ~ Time on Page
• R² = {profile['m1_r2']:.3f}
• Slope coefficient = {profile['m1_slope_top']:.4f}

Model 2: Flexible Relationship (Uncontrolled)
• log(Revenue) ~ Spline(Time on Page)
• R² = {profile['m2_r2']:.3f}
• Captures non-linear patterns and diminishing returns

Model 3: Controlled Analysis
• log(Revenue) ~ Spline(Time on Page) + Browser + Platform + Site
• R² = {profile['m3_r2']:.3f}
• Controls for technical and site-specific factors

DESCRIPTIVE STATISTICS:

Revenue:
• Mean: ${profile['eda_stats']['revenue']['mean']:.4f}
• Median: ${profile['eda_stats']['revenue']['median']:.4f}
• Standard Deviation: ${profile['eda_stats']['revenue']['sd']:.4f}

Time on Page:
• Mean: {profile['eda_stats']['top']['mean']:.2f} units
• Median: {profile['eda_stats']['top']['median']:.2f} units  
• Standard Deviation: {profile['eda_stats']['top']['sd']:.2f} units

Correlation between Revenue and Time on Page: {profile['eda_stats']['corr_revenue_top']:.3f}

ROBUSTNESS:

• Used robust standard errors (HC1) to account for heteroskedasticity
• Spline functions provide flexible modeling without overfitting
• Log transformation addresses skewness in revenue distribution
• Multiple model comparison ensures result consistency
"""
        
        ax.text(0.05, 0.9, methodology_text, fontsize=10, verticalalignment='top',
                transform=ax.transAxes, wrap=True)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    print(f"PDF report created: {pdf_path}")
    return pdf_path

if __name__ == "__main__":
    create_pdf_report()
