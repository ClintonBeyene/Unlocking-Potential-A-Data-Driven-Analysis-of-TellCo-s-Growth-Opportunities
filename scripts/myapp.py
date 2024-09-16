import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from load_data import load_data_from_postgres
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import plotly.express as px

# Load data from PostgreSQL with caching for performance optimization
@st.cache_data
def load_data(query):
    return load_data_from_postgres(query)

# Function to create interactive bar chart
def create_bar_chart(data, x_col, y_col, title):
    fig = px.bar(data, x=x_col, y=y_col, 
                 labels={x_col: 'MSISDN/Number', y_col: 'Total Data Usage (Mb)'},
                 title=title)
    st.plotly_chart(fig)

# Page 1: Combined Subsections
def page_combined(df, data1):
    st.title("TellCo Data Analysis")
    st.write("This app provides a comprehensive analysis of TellCo's data, including handset usage, data consumption, and more.")

    # Top Handsets
    st.header("Top 10 Handsets")
    top_10_handsets = df["Handset Type"].value_counts().head(10)
    top_3_handset_manufacturers = df["Handset Manufacturer"].value_counts().head(3)
    st.write("Top 10 Handsets")
    st.table(top_10_handsets)
    st.write("Top 3 Handset Manufacturers")
    st.table(top_3_handset_manufacturers)

    # Dropdown for selecting a manufacturer
    st.header("Top 5 Handsets per Top 3 Handset Manufacturer")
    manufacturer = st.selectbox("Select a Manufacturer", top_3_handset_manufacturers.index)
    if manufacturer:
        top_5_handsets = df[df['Handset Manufacturer'] == manufacturer]['Handset Type'].value_counts().head(5)
        st.write(f"**Top 5 Handsets for {manufacturer}**")
        st.write(top_5_handsets)

    # Decile Data Analysis
    st.header("Decile Data Analysis")
    data1['decile'] = pd.qcut(data1['total_session_duration'], 10, labels=False)
    top_five_deciles = data1[data1['decile'] >= 5]
    decile_data = top_five_deciles.groupby('decile').agg(
        total_DL_data=('total_DL_data','sum'),
        total_UL_data=('total_UL_data','sum')
    )
    decile_data['total_data'] = decile_data['total_DL_data'] + decile_data['total_UL_data']
    st.write("Decile Data:")
    st.write(decile_data)
    st.write("\nData Summary Statistics:")
    st.write(data1.describe())

    # Dispersion Parameters
    st.header("Dispersion Parameters")
    std_dev = data1.std()
    variance = data1.var()
    dispersion_params = pd.DataFrame({'Standard Deviation': std_dev, 'Variance': variance})
    dispersion_params = dispersion_params.drop('IMSI', axis=0)
    st.write("Dispersion Parameters:")
    st.write(dispersion_params)

    # Distribution Plots
    st.header("Distribution of Total DL Data")
    fig, ax = plt.subplots()
    data1['total_DL_data'].hist(bins=50, edgecolor='black', ax=ax)
    ax.set_title('Distribution of Total DL Data')
    ax.set_xlabel('Total DL Data')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

    st.header("Box Plot of Total DL Data")
    fig, ax = plt.subplots()
    data1.boxplot(column='total_DL_data', ax=ax)
    ax.set_title('Box Plot of Total DL Data')
    ax.set_ylabel('Total DL Data')
    st.pyplot(fig)

    st.header("Density Plot of Total DL Data")
    fig, ax = plt.subplots()
    sns.kdeplot(data1['total_DL_data'], fill=True, ax=ax)
    ax.set_title('Density Plot of Total DL Data')
    ax.set_xlabel('Total DL Data')
    ax.set_ylabel('Density')
    st.pyplot(fig)

    # Bar Chart for xDR Sessions
    st.header("Bar Chart of Number of xDR Sessions")
    fig, ax = plt.subplots()
    data1['number_of_xDR_sessions'].value_counts().plot(kind='bar', edgecolor='black', ax=ax)
    ax.set_title('Bar Chart of Number of xDR Sessions')
    ax.set_xlabel('Number of xDR Sessions')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

    # Bivariate Analysis
    st.header("Bivariate Analysis")
    applications = ['total_social_media_data', 'total_youtube_data', 'total_netflix_data', 'total_google_data', 'total_email_data', 'total_gaming_data']
    cols = st.columns(3)
    for i, app in enumerate(applications):
        with cols[i % 3]:
            fig, ax = plt.subplots()
            ax.scatter(data1[app], data1['total_DL_data'] + data1['total_UL_data'])
            ax.set_title(f'Scatterplot for {app}')
            ax.set_xlabel(app)
            ax.set_ylabel('Total DL+UL Data')
            st.pyplot(fig)

    # Correlation Matrix
    st.header("Correlation Matrix")
    columns_of_interest = [
        'total_social_media_data', 'total_google_data', 'total_email_data',
        'total_youtube_data', 'total_netflix_data', 'total_gaming_data', 'total_other_data'
    ]
    correlation_matrix = data1[columns_of_interest].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix Heatmap')
    st.pyplot(plt)

    # PCA Results
    st.header("PCA Results")
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data1[columns_of_interest])
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(scaled_data)
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, alpha=0.5, align='center')
    ax.set_ylabel('Variance Explained')
    ax.set_xlabel('Principal Components')
    ax.set_title('Explained Variance by Principal Components')
    st.pyplot(fig)
    st.write(pca_df.head())

# Page 2: Placeholder
def engagement_analysis(data2):
    st.title("User Engagement Analysis")
    st.write("Telecom brands must prioritize user engagement and activity on their apps to ensure business success, which can be achieved by tracking user activities and optimizing network resources based on engagement levels.")
    
    agg_metrics = data2.groupby('MSISDN/Number').agg({
        'Bearer Id': 'nunique',
        'Dur. (hr)': 'sum',
        'Total Data (Mb)': 'sum'
    }).reset_index()

    agg_metrics.rename(columns={
        'Bearer Id': 'session_frequency',
        'Dur. (hr)': 'total_duration',
        'Total Data (Mb)': 'total_traffic'
    }, inplace=True)
    
    engagement_metircs = { 'top_10_frequency': agg_metrics.nlargest(10, 'session_frequency'),
    'top_10_duration': agg_metrics.nlargest(10, 'total_duration'),
    'top_10_traffic': agg_metrics.nlargest(10, 'total_traffic')
    }

    st.header('Top 10 Customer per Engagement Metrics') 
    engagement = st.selectbox("Select an Engagement Metrics", list(engagement_metircs.keys()))
    if engagement:
        top_10_customer = engagement_metircs[engagement]
        st.write(f"**Top 10 Customer for {engagement}**")
        st.write(top_10_customer)

    # Normalizing metrics
    scaler = MinMaxScaler()
    normalized_metrics = scaler.fit_transform(agg_metrics[['session_frequency', 'total_duration', 'total_traffic']])

    # Clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    agg_metrics['cluster'] = kmeans.fit_predict(normalized_metrics)

    # Compute metrics for each cluster
    cluster_metrics = agg_metrics.groupby('cluster').agg({
        'session_frequency': ['min', 'max', 'mean', 'sum'],
        'total_duration': ['min', 'max', 'mean', 'sum'],
        'total_traffic': ['min', 'max', 'mean', 'sum']
    }).reset_index()

    # Flatten the multi-level columns
    cluster_metrics.columns = ['_'.join(col).strip() if col[1] else col[0] for col in cluster_metrics.columns.values]

    st.subheader("Cluster Metrics Summary")
    st.write("Below is a summary of the metrics for each cluster, including session frequency, total duration, and total traffic.")

    # Selectbox for choosing the metric
    metric = st.selectbox("Select a Metric", ['session_frequency', 'total_duration', 'total_traffic'])

    # Display the selected metric's statistics
    selected_metrics = cluster_metrics[['cluster', f'{metric}_mean', f'{metric}_max', f'{metric}_sum']]
    selected_metrics.columns = ['Cluster', 'Mean', 'Max', 'Sum']  # Rename columns for better readability

    st.dataframe(selected_metrics)

    # Visualize total traffic per cluster
    st.write("Total Traffic per Cluster")
    fig, ax = plt.subplots()
    sns.barplot(x='cluster', y='total_traffic', data=agg_metrics, ax=ax)
    plt.title('Total Traffic per Cluster')
    st.pyplot(fig)      

    app_traffic = data2.groupby(['MSISDN/Number']).agg({
    'Gaming Data (Mb)': 'sum',
    'Netflix Data (Mb)': 'sum',
    'Email Data (Mb)': 'sum',
    'Google Data (Mb)': 'sum',
    'Youtube Data (Mb)': 'sum',
    'Social Media Data (Mb)': 'sum',
    'Other Data (Mb)': 'sum'
    }).reset_index() 

    # Selectbox for choosing the data category
    category = st.selectbox("Select Data Category", [
        'Gaming Data (Mb)', 'Netflix Data (Mb)', 'Email Data (Mb)', 
        'Google Data (Mb)', 'Youtube Data (Mb)', 'Social Media Data (Mb)', 
        'Other Data (Mb)'
    ])

    # Get top 10 users for the selected category
    top_10 = app_traffic.nlargest(10, category)

    # Create and display the bar chart
    create_bar_chart(top_10, 'MSISDN/Number', category, f'Top 10 Users by {category}')

    # Calculate total usage for each application
    app_usage = app_traffic.drop(columns=['MSISDN/Number']).sum()

    # Sort application by usage and select the top 3
    top_3_apps = app_usage.sort_values(ascending=False).iloc[:3]

    # Create a bar plot for the top 3 most used applications
    fig = px.bar(top_3_apps, x=top_3_apps.index, y=top_3_apps.values,
             labels={'x': 'Application', 'y': 'Total usage (mb)'},
             title = 'Top 3 Most Used Applications')
    st.plotly_chart(fig)

    # Elbow method to detrmine optimal k 
    sse = []
    for k in range(1, 11):
        Kmeans = KMeans(n_clusters=k, random_state=42)
        Kmeans.fit(normalized_metrics)
        sse.append(Kmeans.inertia_)

    # Create an interactive plot for the elbow method 
    fig = px.line(x=range(1, 11), y=sse, labels={'x': 'Number of Clusters', 'y': 'SSE'},
                  title='Elbow Method for Optimal k')
    st.plotly_chart(fig)
    
# Page 3: Placeholder
def experience_analysis():
    st.title("Experience Analytics")
    st.write("The telecommunication industry’s revolution over the last decade, driven by mobile devices, necessitates vendors to focus on consumer needs and perceptions, optimizing products and services through tracking and evaluating customer experiences, particularly in relation to network parameters and device characteristics, to meet evolving expectations.")

# Page 4: Placeholder
def satisfaction_analysis():
    st.title("Satisfaction Analysis")
    st.write("Assuming user satisfaction depends on engagement and experience, this section guides analyst through tasks to analyze customer satisfaction in depth.")

# Main function to run the Streamlit app
def main():
    query = "select * from cleaned_xdr_data;"
    try:
        df = load_data(query)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return
    
    try:
        data1 = pd.read_csv('./notebooks/capped_data.csv')
    except FileNotFoundError:
        st.error("Capped data file not found.")
        return
    
    try:
        data2 = pd.read_csv('./notebooks/user_engagement.csv')
    except FileNotFoundError:
        st.error("User engagement file not found.")
        return 
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["TellCo Data Analysis", "User Engagement Analysis",
                                       "Page 3", "Page 4"])

    if page == "TellCo Data Analysis":
        page_combined(df, data1)
    elif page == "User Engagement Analysis":
        engagement_analysis(data2)
    elif page == "Experience Analytics":
        experience_analysis()
    elif page == "Satisfaction Analysis":
        satisfaction_analysis()

if __name__ == "__main__":
    main()
