import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import io
import os

# Set the LOKY_MAX_CPU_COUNT environment variable
os.environ['LOKY_MAX_CPU_COUNT'] = '4'  # You can adjust this number based on your system

# Load and preprocess the data from the uploaded CSV file
def load_and_preprocess_data(uploaded_file):
    data = pd.read_csv(uploaded_file)
    return data

# Generate remarks for descriptive statistics
def generate_descriptive_stats_remarks(desc_stats):
    remarks = []
    avg_balance = desc_stats.loc['mean', 'Balance']
    if avg_balance > 2000:
        remarks.append(f"The average balance of ${avg_balance:.2f} is higher than expected. This could indicate a more affluent customer base or potential outliers influencing the average.")
    else:
        remarks.append(f"The average balance of ${avg_balance:.2f} is within the normal range, suggesting a typical customer base in terms of account balances.")
    
    max_balance = desc_stats.loc['max', 'Balance']
    min_balance = desc_stats.loc['min', 'Balance']
    remarks.append(f"The highest balance is ${max_balance:.2f}, while the lowest is ${min_balance:.2f}, showing a wide range of customer financial situations.")
    
    std_balance = desc_stats.loc['std', 'Balance']
    remarks.append(f"The standard deviation of balances is ${std_balance:.2f}, indicating the level of variability or dispersion in customer account balances.")
    
    return "\n".join(remarks)

# Generate remarks for balance distribution
def generate_balance_distribution_remarks(data):
    skewness = data['Balance'].skew()
    kurtosis = data['Balance'].kurtosis()
    median = data['Balance'].median()
    q1 = data['Balance'].quantile(0.25)
    q3 = data['Balance'].quantile(0.75)
    
    remarks = []
    if skewness > 0.5:
        remarks.append(f"The balance distribution is significantly positively skewed (skewness: {skewness:.2f}), indicating that most balances are lower than the average, but there are a few very high balances pulling the average up. This suggests a small number of high-value accounts.")
    elif skewness < -0.5:
        remarks.append(f"The balance distribution is significantly negatively skewed (skewness: {skewness:.2f}), indicating that most balances are higher than the average, but there are a few very low balances pulling the average down. This suggests a concentration of higher-value accounts with some outliers on the lower end.")
    else:
        remarks.append(f"The balance distribution is approximately symmetrical (skewness: {skewness:.2f}), meaning balances are relatively evenly distributed around the average. This suggests a balanced mix of account values.")
    
    if kurtosis > 3:
        remarks.append(f"The distribution has heavy tails (kurtosis: {kurtosis:.2f}), indicating more extreme values than would be expected in a normal distribution. This suggests the presence of outliers that may need further investigation.")
    elif kurtosis < 3:
        remarks.append(f"The distribution has light tails (kurtosis: {kurtosis:.2f}), indicating fewer extreme values than would be expected in a normal distribution. This suggests a more uniform spread of balance values.")
    
    remarks.append(f"The median balance is ${median:.2f}, with 25% of accounts below ${q1:.2f} and 75% below ${q3:.2f}. This interquartile range provides insight into the spread of typical account balances.")
    
    return "\n".join(remarks)

# Generate remarks for correlation heatmap
def generate_correlation_remarks(correlation_matrix):
    remarks = []
    income_balance_corr = correlation_matrix.loc['Income', 'Balance']
    if abs(income_balance_corr) > 0.7:
        remarks.append(f"There is a strong {'positive' if income_balance_corr > 0 else 'negative'} correlation (r = {income_balance_corr:.2f}) between Income and Balance. This suggests that {'as income increases, balance tends to increase significantly' if income_balance_corr > 0 else 'as income increases, balance tends to decrease significantly'}.")
    elif abs(income_balance_corr) > 0.3:
        remarks.append(f"There is a moderate {'positive' if income_balance_corr > 0 else 'negative'} correlation (r = {income_balance_corr:.2f}) between Income and Balance. This indicates a {'tendency for balance to increase with income' if income_balance_corr > 0 else 'tendency for balance to decrease with income'}, but other factors also play significant roles.")
    else:
        remarks.append(f"There is a weak or no significant correlation (r = {income_balance_corr:.2f}) between Income and Balance. This suggests that factors other than income may be more important in determining account balances.")
    
    # Find the strongest correlation (excluding self-correlations)
    corr_matrix_no_diag = correlation_matrix.where(~np.eye(correlation_matrix.shape[0], dtype=bool))
    strongest_corr = corr_matrix_no_diag.abs().max().max()
    strongest_pair = np.unravel_index(corr_matrix_no_diag.abs().values.argmax(), corr_matrix_no_diag.shape)
    var1, var2 = correlation_matrix.index[strongest_pair[0]], correlation_matrix.columns[strongest_pair[1]]
    
    remarks.append(f"The strongest correlation is between {var1} and {var2} (r = {strongest_corr:.2f}). This {'positive' if correlation_matrix.loc[var1, var2] > 0 else 'negative'} relationship warrants further investigation to understand its implications for the credit portfolio.")
    
    return "\n".join(remarks)

# Generate remarks for balance by student status
def generate_student_balance_remarks(data):
    remarks = []
    student_mean_balance = data.groupby('Student')['Balance'].mean()
    student_median_balance = data.groupby('Student')['Balance'].median()
    
    if student_mean_balance['Yes'] > student_mean_balance['No']:
        diff_percent = ((student_mean_balance['Yes'] - student_mean_balance['No']) / student_mean_balance['No']) * 100
        remarks.append(f"Students tend to have higher average balances (${student_mean_balance['Yes']:.2f}) compared to non-students (${student_mean_balance['No']:.2f}), a difference of {diff_percent:.1f}%. This could be due to student loans or different spending patterns among students.")
    else:
        diff_percent = ((student_mean_balance['No'] - student_mean_balance['Yes']) / student_mean_balance['Yes']) * 100
        remarks.append(f"Non-students tend to have higher average balances (${student_mean_balance['No']:.2f}) compared to students (${student_mean_balance['Yes']:.2f}), a difference of {diff_percent:.1f}%. This might reflect higher incomes among non-students or more conservative spending habits.")
    
    remarks.append(f"The median balance for students is ${student_median_balance['Yes']:.2f}, while for non-students it's ${student_median_balance['No']:.2f}. This {'supports' if (student_median_balance['Yes'] > student_median_balance['No']) == (student_mean_balance['Yes'] > student_mean_balance['No']) else 'contrasts with'} the trend seen in average balances.")
    
    student_balance_std = data[data['Student'] == 'Yes']['Balance'].std()
    non_student_balance_std = data[data['Student'] == 'No']['Balance'].std()
    remarks.append(f"The standard deviation of balances for students (${student_balance_std:.2f}) {'is higher than' if student_balance_std > non_student_balance_std else 'is lower than'} that of non-students (${non_student_balance_std:.2f}), indicating {'more' if student_balance_std > non_student_balance_std else 'less'} variability in student account balances.")
    
    return "\n".join(remarks)

# Function to display remarks in a structured format
def display_remarks(remarks):
    for remark in remarks:
        st.write("- " + remark)

# Generate overall insights
def generate_overall_insights(data, desc_stats, correlation_matrix):
    insights = []
    
    # Overall balance insights
    avg_balance = desc_stats.loc['mean', 'Balance']
    median_balance = desc_stats.loc['50%', 'Balance']
    insights.append(f"The average balance of ${avg_balance:.2f} {'is significantly higher than' if avg_balance > median_balance * 1.2 else 'is close to'} the median balance of ${median_balance:.2f}, {'suggesting some high-value accounts are influencing the average' if avg_balance > median_balance * 1.2 else 'indicating a relatively even distribution of balances'}.")
    
    # Income-Balance relationship
    income_balance_corr = correlation_matrix.loc['Income', 'Balance']
    insights.append(f"Income shows a {'strong' if abs(income_balance_corr) > 0.7 else 'moderate' if abs(income_balance_corr) > 0.3 else 'weak'} correlation (r = {income_balance_corr:.2f}) with balance, {'indicating that income is a significant factor in determining account balances' if abs(income_balance_corr) > 0.5 else 'suggesting that factors beyond income play important roles in determining account balances'}.")
    
    # Student vs Non-student insights
    if 'Student' in data.columns:
        student_prop = (data['Student'] == 'Yes').mean()
        insights.append(f"Students make up {student_prop:.1%} of the customer base. {'This significant student presence suggests tailored products or services might be beneficial.' if student_prop > 0.3 else 'The relatively small student population might not warrant specific student-focused products.'}")
    
    # Age insights if available
    if 'Age' in data.columns:
        avg_age = data['Age'].mean()
        insights.append(f"The average customer age is {avg_age:.1f} years. {'This suggests a younger customer base, potentially more open to digital banking solutions.' if avg_age < 35 else 'This indicates a more mature customer base, possibly valuing traditional banking services.'}")
    
    # Credit score insights if available
    if 'CreditScore' in data.columns:
        avg_credit_score = data['CreditScore'].mean()
        insights.append(f"The average credit score is {avg_credit_score:.0f}. {'This indicates a generally creditworthy customer base, potentially allowing for more competitive loan offerings.' if avg_credit_score > 700 else 'This suggests room for improvement in the overall creditworthiness of the customer base, possibly through financial education initiatives.'}")
    
    return insights

# Streamlit app
def main():
    st.title('Credit Data Analysis')

    # File uploader
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        # Load data
        data = load_and_preprocess_data(uploaded_file)

        # Dataset Overview
        st.header('Dataset Overview')
        st.write(data.head())
        st.write(f"Total records: {len(data)}")
        st.write(f"Columns: {', '.join(data.columns)}")

        # Descriptive Statistics
        st.header('Descriptive Statistics')
        desc_stats = data.describe()
        st.write(desc_stats)

        # Generate and display descriptive statistics remarks
        st.subheader('Key Remarks:')
        desc_stats_remarks = generate_descriptive_stats_remarks(desc_stats).split("\n")
        display_remarks(desc_stats_remarks)

        # Missing Values
        st.header('Missing Values')
        missing_values = data.isnull().sum()
        st.write(missing_values)
        if missing_values.sum() == 0:
            st.write("There are no missing values in the dataset, indicating complete data for all records.")
        else:
            st.write(f"There are {missing_values.sum()} missing values across {missing_values[missing_values > 0].count()} columns that need attention. This may affect the accuracy of our analysis.")

        # Balance Distribution
        st.header('Balance Distribution')
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data['Balance'], kde=True, ax=ax)
        ax.set_title('Distribution of Balance')
        st.pyplot(fig)

        # Generate and display balance distribution remarks
        st.subheader('Distribution Analysis Remarks:')
        balance_remarks = generate_balance_distribution_remarks(data).split("\n")
        display_remarks(balance_remarks)

        # Income vs Balance
        st.header('Income vs Balance')
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x='Income', y='Balance', data=data, ax=ax)
        ax.set_title('Income vs Balance')
        st.pyplot(fig)

        income_balance_corr = data['Income'].corr(data['Balance'])
        st.write(f"Income-Balance Correlation: {income_balance_corr:.2f}")

        # Correlation Heatmap
        st.header('Correlation Heatmap')

        # Remove non-numeric columns before calculating correlation matrix
        numeric_data = data.select_dtypes(include=[np.number])
        correlation_matrix = numeric_data.corr()

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        ax.set_title('Correlation Matrix')
        st.pyplot(fig)

        # Generate and display correlation remarks
        st.subheader('Correlation Analysis Remarks:')
        correlation_remarks = generate_correlation_remarks(correlation_matrix).split("\n")
        display_remarks(correlation_remarks)

        # Balance by Student Status (if 'Student' column exists)
        if 'Student' in data.columns:
            st.header('Balance Distribution by Student Status')
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(x='Student', y='Balance', data=data, ax=ax)
            ax.set_title('Balance Distribution by Student Status')
            st.pyplot(fig)

            # Generate and display student balance remarks
            st.subheader('Student vs Non-Student Balance Remarks:')
            student_balance_remarks = generate_student_balance_remarks(data).split("\n")
            display_remarks(student_balance_remarks)

        # Final Conclusion based on the analysis
        st.header('Overall Insights and Recommendations:')
        overall_insights = generate_overall_insights(data, desc_stats, correlation_matrix)
        display_remarks(overall_insights)

        # Additional automated analysis based on available data
        st.header('Additional Automated Analysis:')
        
        # Age distribution (if 'Age' column exists)
        if 'Age' in data.columns:
            st.subheader('Age Distribution')
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(data['Age'], kde=True, ax=ax)
            ax.set_title('Distribution of Customer Age')
            st.pyplot(fig)
            
            avg_age = data['Age'].mean()
            median_age = data['Age'].median()
            st.write(f"Average Age: {avg_age:.1f} years")
            st.write(f"Median Age: {median_age:.1f} years")
            
            age_remarks = []
            if avg_age < 30:
                age_remarks.append("The customer base is predominantly young, suggesting a focus on digital banking solutions and products tailored for younger demographics.")
            elif avg_age < 45:
                age_remarks.append("The customer base has a balanced age distribution, indicating a need for diverse product offerings to cater to various life stages.")
            else:
                age_remarks.append("The customer base skews older, suggesting potential emphasis on retirement planning and wealth management services.")
            
            if abs(avg_age - median_age) > 5:
                age_remarks.append(f"The significant difference between mean and median age ({abs(avg_age - median_age):.1f} years) indicates a skewed age distribution, possibly due to outliers or distinct customer segments.")
            
            display_remarks(age_remarks)

        # Credit Score analysis (if 'CreditScore' column exists)
        if 'CreditScore' in data.columns:
            st.subheader('Credit Score Analysis')
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(data['CreditScore'], kde=True, ax=ax)
            ax.set_title('Distribution of Credit Scores')
            st.pyplot(fig)
            
            avg_credit_score = data['CreditScore'].mean()
            median_credit_score = data['CreditScore'].median()
            low_credit_score_pct = (data['CreditScore'] < 600).mean() * 100
            high_credit_score_pct = (data['CreditScore'] > 750).mean() * 100
            
            st.write(f"Average Credit Score: {avg_credit_score:.0f}")
            st.write(f"Median Credit Score: {median_credit_score:.0f}")
            st.write(f"Percentage of customers with low credit scores (<600): {low_credit_score_pct:.1f}%")
            st.write(f"Percentage of customers with high credit scores (>750): {high_credit_score_pct:.1f}%")
            
            credit_score_remarks = []
            if avg_credit_score > 700:
                credit_score_remarks.append("The average credit score is good, indicating a generally creditworthy customer base. This may allow for more competitive loan offerings and lower default risks.")
            elif avg_credit_score > 650:
                credit_score_remarks.append("The average credit score is fair, suggesting a moderate level of credit risk. Targeted credit improvement programs could be beneficial.")
            else:
                credit_score_remarks.append("The average credit score is below average, indicating higher credit risk. Consider implementing credit education programs and secured credit products.")
            
            if high_credit_score_pct > 30:
                credit_score_remarks.append(f"A significant portion ({high_credit_score_pct:.1f}%) of customers have excellent credit scores, presenting opportunities for premium financial products and services.")
            if low_credit_score_pct > 20:
                credit_score_remarks.append(f"A considerable percentage ({low_credit_score_pct:.1f}%) of customers have low credit scores, suggesting a need for credit builder products and financial education initiatives.")
            
            display_remarks(credit_score_remarks)

        # Customer Segmentation (if relevant columns exist)
        if all(col in data.columns for col in ['Balance', 'Income', 'Age']):
            st.subheader('Customer Segmentation Analysis')
            from sklearn.preprocessing import StandardScaler
            from sklearn.cluster import KMeans
            
            # Prepare data for clustering
            features = ['Balance', 'Income', 'Age']
            X = data[features]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=3, random_state=42)
            data['Cluster'] = kmeans.fit_predict(X_scaled)
            
            # Visualize clusters
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(X['Balance'], X['Income'], X['Age'], c=data['Cluster'], cmap='viridis')
            ax.set_xlabel('Balance')
            ax.set_ylabel('Income')
            ax.set_zlabel('Age')
            plt.colorbar(scatter)
            plt.title('Customer Segments')
            st.pyplot(fig)
            
            # Generate insights for each cluster
            cluster_insights = []
            for cluster in range(3):
                cluster_data = data[data['Cluster'] == cluster]
                cluster_insights.append(f"Cluster {cluster}:")
                cluster_insights.append(f"- Size: {len(cluster_data)} customers ({len(cluster_data) / len(data) * 100:.1f}% of total)")
                cluster_insights.append(f"- Avg Balance: ${cluster_data['Balance'].mean():.2f}")
                cluster_insights.append(f"- Avg Income: ${cluster_data['Income'].mean():.2f}")
                cluster_insights.append(f"- Avg Age: {cluster_data['Age'].mean():.1f} years")
                cluster_insights.append("")
            
            st.write("Customer Segment Characteristics:")
            display_remarks(cluster_insights)
            
            segmentation_remarks = []
            segmentation_remarks.append("The customer base has been segmented into three distinct groups based on balance, income, and age.")
            segmentation_remarks.append("These segments can be used to tailor marketing strategies, product offerings, and customer service approaches.")
            segmentation_remarks.append("Consider developing targeted campaigns and products for each segment to improve customer satisfaction and increase revenue.")
            
            display_remarks(segmentation_remarks)

        # Time-based analysis (if 'Date' column exists)
        if 'Date' in data.columns:
            st.subheader('Time-based Analysis')
            data['Date'] = pd.to_datetime(data['Date'])
            data.set_index('Date', inplace=True)
            
            # Monthly average balance trend
            monthly_avg_balance = data['Balance'].resample('M').mean()
            fig, ax = plt.subplots(figsize=(12, 6))
            monthly_avg_balance.plot(ax=ax)
            ax.set_title('Monthly Average Balance Trend')
            ax.set_xlabel('Date')
            ax.set_ylabel('Average Balance')
            st.pyplot(fig)
            
            # Calculate growth rate
            total_growth = (monthly_avg_balance.iloc[-1] - monthly_avg_balance.iloc[0]) / monthly_avg_balance.iloc[0] * 100
            annualized_growth = total_growth / (len(monthly_avg_balance) / 12)
            
            time_remarks = []
            time_remarks.append(f"The overall growth in average balance over the observed period is {total_growth:.1f}%.")
            time_remarks.append(f"The annualized growth rate of average balance is approximately {annualized_growth:.1f}%.")
            
            if annualized_growth > 5:
                time_remarks.append("The positive growth trend indicates successful strategies in attracting and retaining high-value customers.")
            elif annualized_growth < 0:
                time_remarks.append("The negative growth trend suggests a need to review customer retention strategies and investigate potential causes of decreasing balances.")
            else:
                time_remarks.append("The relatively stable average balance indicates consistent performance but may also suggest opportunities for growth strategies.")
            
            display_remarks(time_remarks)

        # Recommendations
        st.header('Recommendations')
        recommendations = [
            "Develop tailored product offerings based on the identified customer segments to improve customer satisfaction and increase cross-selling opportunities.",
            "Implement targeted marketing campaigns for each age group, focusing on their specific financial needs and preferences.",
            "Enhance credit education programs to help customers with lower credit scores improve their financial health, potentially leading to increased product eligibility and customer loyalty.",
            "Leverage the correlation between income and balance to design tiered product offerings that cater to different income levels.",
            "Monitor and analyze the time-based trends in average balances to proactively address any declining trends and capitalize on growth opportunities.",
            "Consider introducing or promoting student-specific products if the analysis shows a significant student customer base with unique financial behaviors.",
            "Regularly update this analysis to track the impact of implemented strategies and identify emerging trends in customer behavior."
        ]
        display_remarks(recommendations)

        # Export results
        if st.button('Export Analysis Results'):
            export_results(data, desc_stats, correlation_matrix, overall_insights, recommendations)
            st.success('Analysis results have been exported successfully!')

def export_results(data, desc_stats, correlation_matrix, overall_insights, recommendations):
    # Create a BytesIO object to store the CSV data
    output = io.StringIO()
    
    # Write descriptive statistics
    output.write("Descriptive Statistics\n")
    desc_stats.to_csv(output)
    output.write("\n\n")
    
    # Write correlation matrix
    output.write("Correlation Matrix\n")
    correlation_matrix.to_csv(output)
    output.write("\n\n")
    
    # Write insights and recommendations
    output.write("Overall Insights\n")
    for insight in overall_insights:
        output.write(f"{insight}\n")
    output.write("\n")
    
    output.write("Recommendations\n")
    for recommendation in recommendations:
        output.write(f"{recommendation}\n")
    output.write("\n\n")
    
    # Add summary statistics
    output.write("Summary Statistics\n")
    output.write(f"Total Customers: {len(data)}\n")
    output.write(f"Average Balance: ${data['Balance'].mean():.2f}\n")
    output.write(f"Average Age: {data['Age'].mean():.1f}\n")
    output.write(f"Average Credit Score: {data['CreditScore'].mean():.0f}\n")
    
    # Get the value of the StringIO buffer
    csv_data = output.getvalue()
    
    # Provide download button
    st.download_button(
        label="Download CSV Report",
        data=csv_data,
        file_name="credit_data_analysis_report.csv",
        mime="text/csv"
    )
if __name__ == '__main__':
    main()