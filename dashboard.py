import pandas as pd 
import matplotlib.pyplot as plt
import streamlit as st 
import plotly.express as px
import seaborn as sns
from sklearn.metrics import confusion_matrix

df = pd.read_csv("/content/drive/MyDrive/healthcare-dataset-stroke-data.csv") 

# Function for confusion matrix
def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot(plt)

# Imputing missing bmi values with the median
df['bmi'].fillna(df['bmi'].median(), inplace=True)

# Streamlit App code
st.subheader("Stroke Data")
st.dataframe(df.head())

# Creating tabs for different types of graphs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Scatter Plot", "Histogram", "Pie Chart", "Bar Chart", "Line Chart", "Tree Map", "Heatmap & Confusion Matrices"])

with tab1:
    st.sidebar.header("Scatter Plot Options")
    scatter_x = st.sidebar.selectbox("X-Axis for Scatter Plot", ["age", "avg_glucose_level", "bmi"])
    scatter_y = st.sidebar.selectbox("Y-Axis for Scatter Plot", ["heart_disease", "hypertension", "stroke"])

    scatter_fig = px.scatter(df, x=scatter_x, y=scatter_y, 
                             title=f"Scatter Plot of {scatter_x} vs {scatter_y}", 
                             labels={scatter_x: scatter_x.capitalize(), scatter_y: scatter_y.capitalize()})
    st.plotly_chart(scatter_fig)

with tab2:
    st.sidebar.header("Histogram Options")
    hist_column = st.sidebar.selectbox("Select Column for Histogram", 
                                       ["hypertension", "heart_disease", "avg_glucose_level", "bmi", "stroke"])

    hist_fig = px.histogram(df, x=hist_column, 
                            title=f"Histogram of {hist_column}", 
                            labels={hist_column: hist_column.capitalize()}, 
                            nbins=50)
    st.plotly_chart(hist_fig)

with tab3:
    st.sidebar.header("Pie Chart Options")
    pie_column = st.sidebar.selectbox("Select Column for Pie Chart", 
                                      ["gender", "ever_married", "work_type", "Residence_type", "smoking_status", "stroke"])

    pie_fig = px.pie(df, names=pie_column, 
                     title=f"Pie Chart of {pie_column}", 
                     labels={pie_column: pie_column.capitalize()})
    st.plotly_chart(pie_fig)

with tab4:
    st.sidebar.header("Bar Chart Options")
    bar_x = st.sidebar.selectbox("X-Axis for Bar Chart", ["gender", "ever_married", "work_type", "Residence_type", "smoking_status", "stroke"])
    bar_y = st.sidebar.selectbox("Y-Axis for Bar Chart", ["age", "avg_glucose_level", "bmi", "heart_disease", "hypertension", "stroke"])

    bar_fig = px.bar(df, x=bar_x, y=bar_y, 
                     title=f"Bar Chart of {bar_x} vs {bar_y}", 
                     labels={bar_x: bar_x.capitalize(), bar_y: bar_y.capitalize()})
    st.plotly_chart(bar_fig)

with tab5:
    st.sidebar.header("Line Chart Options")
    line_x = st.sidebar.selectbox("Line Chart X-Axis", ["age", "avg_glucose_level", "bmi"])
    line_y = st.sidebar.selectbox("Line Chart Y-Axis", ["heart_disease", "hypertension", "stroke"])

    line_fig = px.line(df, x=line_x, y=line_y, 
                       title=f"Line Chart of {line_x} vs {line_y}", 
                       labels={line_x: line_x.capitalize(), line_y: line_y.capitalize()})
    st.plotly_chart(line_fig)

with tab6:
    st.sidebar.header("Tree Map Options")
    tree_path = st.sidebar.multiselect("Select Hierarchical Columns for Tree Map", ["gender", "ever_married", "work_type", "Residence_type", "smoking_status", "stroke"], default=["gender", "ever_married"])
    tree_values = st.sidebar.selectbox("Select Values for Tree Map", ["age", "avg_glucose_level", "bmi", "heart_disease", "hypertension", "stroke"])

    if tree_path:
        tree_fig = px.treemap(df, path=tree_path, values=tree_values, 
                              title=f"Tree Map of {tree_values} by {', '.join(tree_path)}", 
                              labels={tree_values: tree_values.capitalize()})
        st.plotly_chart(tree_fig)

with tab7:
    st.sidebar.header("Heatmap & Confusion Matrices Options")
    if st.sidebar.checkbox("Show Correlation Heatmap"):
        st.header("Correlation Heatmap")
        numerical_df = df.select_dtypes(include=['float64', 'int64'])
        correlation_matrix = numerical_df.corr()
        fig, ax = plt.subplots()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    if st.sidebar.checkbox("Show Confusion Matrices"):
        st.header("Confusion Matrices")
        y_test = st.sidebar.selectbox("Select True Labels Column", ["stroke"])  # Change this as needed
        y_pred_knn = st.sidebar.selectbox("Select KNN Predictions Column", ["stroke"])  # Change this as needed
        y_pred_svm = st.sidebar.selectbox("Select SVM Predictions Column", ["stroke"])  # Change this as needed
        y_pred_rf = st.sidebar.selectbox("Select RF Predictions Column", ["stroke"])  # Change this as needed

        # Generate confusion matrices
        cm_knn = confusion_matrix(df[y_test], df[y_pred_knn])
        cm_svm = confusion_matrix(df[y_test], df[y_pred_svm])
        cm_rf = confusion_matrix(df[y_test], df[y_pred_rf])

        # Plot confusion matrices
        plot_confusion_matrix(cm_knn, 'KNN Confusion Matrix')
        plot_confusion_matrix(cm_svm, 'SVM Confusion Matrix')
        plot_confusion_matrix(cm_rf, 'RF Confusion Matrix')
