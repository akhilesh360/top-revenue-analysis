import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Page setup
st.set_page_config(
    page_title="CLT Demo", 
    page_icon="📊", 
    layout="centered",
    initial_sidebar_state="expanded"
)

# Title & description
st.title("Central Limit Theorem Demo 📊")
st.caption("Explore how the distribution of sample means approaches Normal as sample size grows.")

# Add educational intro
with st.expander("📚 What is the Central Limit Theorem?", expanded=False):
    st.markdown("""
    The **Central Limit Theorem (CLT)** is one of the most important concepts in statistics:
    
    🎯 **Key Insight**: No matter what the original population looks like, the distribution of sample means will approach a normal (bell-shaped) distribution as the sample size increases.
    
    🔍 **Why This Matters**:
    - Makes statistical inference possible
    - Foundation for confidence intervals and hypothesis testing
    - Explains why many natural phenomena follow normal distributions
    
    🧪 **Experiment**: Try different population distributions below and watch the magic happen!
    """)

# Sidebar controls
with st.sidebar:
    st.header("🎛️ Controls")
    
    st.markdown("**Population Distribution**")
    dist_name = st.selectbox(
        "Choose the underlying population:", 
        ["Normal", "Uniform", "Exponential", "t (df=2)", "Beta (skewed)", "Poisson"],
        help="This is the distribution we're sampling from"
    )
    
    st.markdown("**Sampling Parameters**")
    n = st.slider(
        "Sample size per replication (n)", 
        5, 500, 30, step=5,
        help="Number of observations in each sample"
    )
    R = st.slider(
        "Number of replications (R)", 
        100, 10000, 2000, step=100,
        help="How many samples to take"
    )
    
    st.markdown("**Visualization**")
    bins = st.slider("Histogram bins", 20, 120, 50, step=5)
    seed = st.number_input("Random seed (optional)", value=42, step=1)
    overlay_norm = st.checkbox("Overlay Normal curve on histogram", value=True)
    show_qq = st.checkbox("Show QQ-plot vs Normal", value=False)
    show_population = st.checkbox("Show original population distribution", value=True)

# RNG for reproducibility
rng = np.random.default_rng(int(seed) if seed else None)

# Generate data from chosen population
if dist_name == "Normal":
    data = rng.normal(0, 1, size=(R, n))
    pop_data = rng.normal(0, 1, size=10000)
elif dist_name == "Uniform":
    data = rng.uniform(-2, 2, size=(R, n))
    pop_data = rng.uniform(-2, 2, size=10000)
elif dist_name == "Exponential":
    data = rng.exponential(1, size=(R, n))
    pop_data = rng.exponential(1, size=10000)
elif dist_name == "t (df=2)":
    data = rng.standard_t(2, size=(R, n))
    pop_data = rng.standard_t(2, size=10000)
elif dist_name == "Beta (skewed)":
    data = rng.beta(2, 5, size=(R, n))
    pop_data = rng.beta(2, 5, size=10000)
else:  # Poisson
    data = rng.poisson(3, size=(R, n))
    pop_data = rng.poisson(3, size=10000)

# Compute sample means
means = data.mean(axis=1)
empirical_mean = means.mean()
empirical_se = means.std(ddof=1)

# Show population distribution if requested
if show_population:
    st.subheader("📊 Original Population Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        fig_pop, ax_pop = plt.subplots(figsize=(6, 4))
        ax_pop.hist(pop_data, bins=50, density=True, alpha=0.7, color="lightcoral", edgecolor="black")
        ax_pop.set_xlabel("Value")
        ax_pop.set_ylabel("Density")
        ax_pop.set_title(f"{dist_name} Population Distribution")
        st.pyplot(fig_pop)
    
    with col2:
        st.metric("Population Mean", f"{pop_data.mean():.3f}")
        st.metric("Population Std Dev", f"{pop_data.std():.3f}")
        st.metric("Population Skewness", f"{stats.skew(pop_data):.3f}")
        st.markdown(f"**Shape**: {dist_name}")

# Histogram of sample means
st.subheader("🎯 Distribution of Sample Means (The Magic!)")
fig, ax = plt.subplots(figsize=(8, 6))
ax.hist(means, bins=bins, density=True, alpha=0.7, color="skyblue", edgecolor="black")

if overlay_norm:
    # Fit a Normal curve with same mean & SE as the sample means
    x = np.linspace(means.min(), means.max(), 300)
    ax.plot(x, stats.norm.pdf(x, loc=empirical_mean, scale=empirical_se),
            "r-", lw=2, label="Normal PDF")
    ax.legend()

ax.set_xlabel("Sample Mean")
ax.set_ylabel("Density")
ax.set_title(f"{dist_name} population → Sample Means Distribution (n={n}, R={R:,})")
ax.grid(True, alpha=0.3)
st.pyplot(fig)

# Add metrics below the plot
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Mean of Sample Means", f"{empirical_mean:.4f}")
with col2:
    st.metric("SE of Sample Means", f"{empirical_se:.4f}")
with col3:
    theoretical_se = pop_data.std() / np.sqrt(n)
    st.metric("Theoretical SE", f"{theoretical_se:.4f}")
with col4:
    normality_test = stats.normaltest(means)
    st.metric("Normality p-value", f"{normality_test.pvalue:.6f}")

# Add explanation of what we're seeing
st.markdown(f"""
**🔍 What's happening?**
- Original population: {dist_name} distribution
- Sample means empirical SE: {empirical_se:.4f}
- Theoretical SE (σ/√n): {theoretical_se:.4f}
- Difference: {abs(empirical_se - theoretical_se):.4f}
""")

if normality_test.pvalue < 0.05:
    st.success("✅ The sample means are significantly normal-like (CLT working!)")
else:
    st.warning("⚠️ Sample means not yet fully normal - try increasing n or R")

# QQ-plot option
if show_qq:
    st.subheader("🎲 QQ-plot vs Normal Distribution")
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    stats.probplot(means, dist="norm", plot=ax2)
    ax2.set_title("QQ-plot: Sample Means vs Normal Distribution")
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)
    
    st.markdown("""
    **📈 Reading the QQ-plot:**
    - Points on the diagonal line = normally distributed
    - Curved pattern = deviation from normality
    - Better fit to the line = stronger evidence for CLT
    """)

# Educational summary
st.markdown("---")
st.subheader("📚 Central Limit Theorem Summary")

col1, col2 = st.columns(2)
with col1:
    st.markdown(f"""
    **📊 Your Results:**
    - **SE of sample means**: {empirical_se:.4f}
    - **Theoretical SE**: {pop_data.std() / np.sqrt(n):.4f}
    - **Sample size**: {n:,}
    - **Replications**: {R:,}
    """)

with col2:
    st.markdown("""
    **🧠 Key Insights:**
    - As **n** increases → SE decreases as **1/√n**
    - Shape becomes normal regardless of population
    - Larger samples = more precise estimates
    - Foundation for statistical inference
    """)

# Final call to action
st.info("""
🎯 **Try This**: 
1. Start with **Exponential** or **Beta (skewed)** distribution
2. Set **n=5** and see the non-normal shape
3. Gradually increase **n** to 50, 100, 200...
4. Watch the distribution become beautifully normal! ✨
""")

# Add footer
st.markdown("---")
st.markdown("*Built with Streamlit • Demonstrating the Central Limit Theorem*")
