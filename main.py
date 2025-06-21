import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# Load and preprocess the data
@st.cache_data
def load_data():
    data = pd.read_csv('data.csv')
    
    # Identify categorical columns (excluding the target)
    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
    categorical_cols.remove('Overall Category')  # Remove target from features
    
    # Identify numerical columns
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Define features and target category
    features = data.drop('Overall Category', axis=1)
    target = data['Overall Category']
    
    # Encode target variable
    le = LabelEncoder()
    target_encoded = le.fit_transform(target)
    
    # Create preprocessing pipeline for features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])
    
    # Split data into train and test sets, 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        features, target_encoded, test_size=0.2, random_state=11
    )
    
    return X_train, X_test, y_train, y_test, le, preprocessor, data

# Train and evaluate models
def evaluate_model(model, X_train, X_test, y_train, y_test, le, preprocessor):
    # Create pipeline with preprocessing and model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    pipeline.fit(X_train, y_train) # Model training
    y_pred = pipeline.predict(X_test) # Make predictions
    accuracy = accuracy_score(y_test, y_pred) # Calculate accuracy of previous predictions
    
    # Cross-validation
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)
    cv_mean = np.mean(cv_scores)
    
    # Confusion matrix (TT/FF/TF/FT)
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, 
                         index=le.classes_, 
                         columns=le.classes_)
    
    # Classification report
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    
    return {
        'model': pipeline,
        'accuracy': accuracy,
        'cv_mean': cv_mean,
        'confusion_matrix': cm_df,
        'classification_report': report_df,
        'predictions': y_pred,
        'true_values': y_test
    }

# Plot functions (same as before)
def plot_confusion_matrix(cm_df):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', ax=ax)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    st.pyplot(fig)

def plot_accuracy_comparison(results):
    models = list(results.keys())
    accuracies = [results[model]['accuracy'] for model in results]
    cv_means = [results[model]['cv_mean'] for model in results]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(models))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, accuracies, width, label='Test Accuracy')
    rects2 = ax.bar(x + width/2, cv_means, width, label='CV Mean Accuracy')
    
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Accuracy Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    
    st.pyplot(fig)

def plot_classification_report(report_df):
    # Drop support column and keep only precision, recall, f1-score
    report_to_plot = report_df.drop('support', axis=1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    report_to_plot.plot(kind='bar', ax=ax)
    plt.title('Classification Report Metrics')
    plt.ylabel('Score')
    plt.xlabel('Class')
    plt.xticks(rotation=45)
    plt.legend(loc='lower right')
    st.pyplot(fig)

# Main Streamlit app
def main():
    st.title("Student Performance Prediction")
    st.write("""
    This app evaluates four classification algorithms for predicting student performance categories.
    The dataset contains information about students' academic background, demographics, and study habits.
    """)
    
    # Load data
    X_train, X_test, y_train, y_test, le, preprocessor, data = load_data()

    # Verification of split
    st.subheader("Data Split Verification")
    st.write(f"Total samples: {len(data)}")
    st.write(f"Training samples: {len(X_train)} ({len(X_train)/len(data):.1%})")
    st.write(f"Testing samples: {len(X_test)} ({len(X_test)/len(data):.1%})")
    
    # Models init
    models = {
        'Decision Tree': DecisionTreeClassifier(random_state=66),
        'Support Vector Machine': SVC(random_state=18),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB()
    }
    
    # Evaluate models
    results = {}
    for name, model in models.items():
        results[name] = evaluate_model(model, X_train, X_test, y_train, y_test, le, preprocessor)
    
    # Create tabs for each model
    tabs = st.tabs(list(results.keys()) + ["Comparison"])
    
    # Display each model's results in its own tab
    for i, (model_name, result) in enumerate(results.items()):
        with tabs[i]:
            st.header(f"{model_name} Results")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Test Accuracy", f"{result['accuracy']:.2%}")
            with col2:
                st.metric("Cross-Validation Mean Accuracy", f"{result['cv_mean']:.2%}")
            
            st.subheader("Confusion Matrix")
            plot_confusion_matrix(result['confusion_matrix'])
            
            st.subheader("Classification Report")
            st.dataframe(result['classification_report'])
            
            st.subheader("Detailed Predictions")
            # Create a DataFrame with actual and predicted values
            predictions_df = pd.DataFrame({
                'Actual': le.inverse_transform(result['true_values']),
                'Predicted': le.inverse_transform(result['predictions']),
                'Correct': result['true_values'] == result['predictions']
            })
            
            # Count correct and incorrect predictions
            correct = predictions_df['Correct'].sum()
            incorrect = len(predictions_df) - correct
            
            st.write(f"Correct predictions: {correct} ({correct/len(predictions_df):.2%})")
            st.write(f"Incorrect predictions: {incorrect} ({incorrect/len(predictions_df):.2%})")
            
            # Show sample predictions
            st.write("Sample predictions:")
            st.dataframe(predictions_df.head(10))
    
    # Comparison tab
    with tabs[-1]:
        st.header("Model Comparison")
        
        st.subheader("Accuracy Comparison")
        plot_accuracy_comparison(results)
        
        st.subheader("Detailed Metrics Comparison")
        # Create a DataFrame comparing all models
        comparison_df = pd.DataFrame({
            'Model': list(results.keys()),
            'Test Accuracy': [results[model]['accuracy'] for model in results],
            'CV Mean Accuracy': [results[model]['cv_mean'] for model in results],
            'Precision (Weighted Avg)': [results[model]['classification_report'].loc['weighted avg', 'precision'] for model in results],
            'Recall (Weighted Avg)': [results[model]['classification_report'].loc['weighted avg', 'recall'] for model in results],
            'F1-Score (Weighted Avg)': [results[model]['classification_report'].loc['weighted avg', 'f1-score'] for model in results]
        })
        
        st.dataframe(comparison_df)
        
        st.subheader("Keywords")
        st.write("""
        - **Test Accuracy**: The percentage of correct predictions on the test set (20% of data)
        - **CV Mean Accuracy**: The average accuracy from 5-fold cross-validation on the training set
        - **Precision**: The ratio of correctly predicted positive observations to the total predicted positives
        - **Recall**: The ratio of correctly predicted positive observations to all observations in actual class
        - **F1-Score**: The weighted average of Precision and Recall
        """)

if __name__ == "__main__":
    main()