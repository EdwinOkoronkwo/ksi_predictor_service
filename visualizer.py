import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import style
style.use('ggplot')
from utils import setup_logger #Import logger

class DataVisualization:
    """
    Class for generating visualizations from a dataset.
    """

    def __init__(self, data):
        """
        Initializes the DataVisualization object.

        Args:
            data (pd.DataFrame): Pandas DataFrame containing the data.
        """
        self.data = data


    def plot_roc_curve(self, fpr, tpr, roc_auc, model_name):
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc='lower right')
        plt.show()

    def plot_roc_auc_comparison(self, model_results):
        """Plots the ROC AUC comparison."""
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Model', y='ROC AUC', data=model_results)
        plt.title('Model Comparison (ROC AUC)')
        plt.ylabel('ROC AUC')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        for index, value in enumerate(model_results['ROC AUC']):
            plt.text(index, value, round(value, 3), ha='center', va='bottom')
        plt.show()

    def plot_confusion_matrix(self, cm, model_name):
        """
        Plots a confusion matrix using seaborn heatmap.

        Args:
            cm (numpy.ndarray): The confusion matrix.
            model_name (str): The name of the model.
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Confusion Matrix - {model_name}")
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.show()

    def plot_knn_optimum_k(self, mean_test_scores, k_values):
        """Plots the optimum k for KNN and annotates the optimum k value."""
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, mean_test_scores, marker='o')
        plt.title("KNN Optimum k")
        plt.xlabel("Number of Neighbors (k)")
        plt.ylabel("Mean Test Score (Accuracy)")
        plt.xticks(k_values)
        plt.grid(True)
        # Find the optimum k and its corresponding score
        optimum_k = k_values[mean_test_scores.argmax()]
        optimum_score = max(mean_test_scores)
        # Annotate the optimum k value
        plt.annotate(
            f'Optimum k = {optimum_k}\nAccuracy = {optimum_score:.4f}',
            xy=(optimum_k, optimum_score),
            xytext=(optimum_k + 1, optimum_score - 0.02),  # Adjust text position as needed
            arrowprops=dict(facecolor='black', shrink=0.05),
        )
        plt.tight_layout()
        plt.show()

    def plot_train_test_metrics(self, train_accuracy, test_accuracy, model_name):
        """Plots the train vs test accuracy."""
        plt.figure(figsize=(10, 5))
        sns.barplot(x=['Train Accuracy', 'Test Accuracy'], y=[train_accuracy, test_accuracy])
        plt.title(f'Train vs Test Accuracy - {model_name}')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        for index, value in enumerate([train_accuracy, test_accuracy]):
            plt.text(index, value, round(value, 3), ha='center', va='bottom')
        plt.show()

    def plot_model_comparison(self, accuracy_scores):
        """Plots the model comparison."""
        plt.figure(figsize=(12, 6))
        sns.barplot(x=list(accuracy_scores.keys()), y=list(accuracy_scores.values()))
        plt.title('Model Comparison (Test Accuracy)')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        for model, accuracy in accuracy_scores.items():
            plt.text(list(accuracy_scores.keys()).index(model), accuracy, round(accuracy, 3), ha='center', va='bottom')
        plt.show()

    def plot_missing_data_heatmaps(self):
        """Plots separate heatmaps for missing data in four groups of columns."""
        print("\n--- Missing Data Heatmaps ---")

        # Split the columns into four groups
        n_cols = len(self.data.columns)
        group_size = n_cols // 4  # Calculate the size of each group
        column_groups = [
            self.data.columns[:group_size],
            self.data.columns[group_size:2 * group_size],
            self.data.columns[2 * group_size:3 * group_size],
            self.data.columns[3 * group_size:]
        ]

        # Generate heatmaps for each group
        for i, group in enumerate(column_groups):
            plt.figure(figsize=(10, 6))
            sns.heatmap(self.data[group].isnull(), cbar=True, cmap='viridis')
            plt.title(f"Missing Data Heatmap (Group {i + 1})")

            # Rotate x-axis labels
            plt.xticks(rotation=45, ha='right')  # Rotate 45 degrees and align right

            plt.show()

    def plot_histograms(self):
        """Plots histograms for selected numerical columns."""
        print("\n--- Histograms ---")
        numerical_cols = ['TIME', 'LATITUDE', 'LONGITUDE', 'ACCLASS', 'FATAL_NO']
        for col in numerical_cols:
            plt.figure()
            sns.histplot(self.data[col].dropna(), kde=True)
            plt.title(f"Histogram of {col}")
            plt.show()

    def plot_pairwise_scatter_plots(self):
        """Plots pairwise scatter plots for selected numerical columns."""
        print("\n--- Pairwise Scatter Plots ---")
        numeric_cols = ['TIME', 'LATITUDE', 'LONGITUDE', 'ACCLASS', 'FATAL_NO']
        sns.pairplot(self.data[numeric_cols])
        plt.show()


    def plot_bar_plots(self):
        """Plots bar plots for categorical columns."""
        print("\n--- Bar Plots ---")
        categorical_cols = ['ROAD_CLASS', 'DISTRICT', 'TRAFFCTL', 'VISIBILITY', 'LIGHT', 'RDSFCOND',
                            'ACCLASS', 'IMPACTYPE', 'INVTYPE', 'INJURY', 'DIVISION', 'HOOD_158',
                            'NEIGHBOURHOOD_158']
        for col in categorical_cols:
            plt.figure(figsize=(10, 6))
            sns.countplot(y=self.data[col].dropna(), order=self.data[col].dropna().value_counts().index)
            plt.title(f"Bar Plot of {col}")
            plt.show()

    def plot_pie_charts(self):
        """Plots pie charts for categorical columns."""
        print("\n--- Pie Charts ---")
        pie_cols = ['ACCLASS', 'ROAD_CLASS']
        for col in pie_cols:
            plt.figure(figsize=(8, 6))
            self.data[col].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
            plt.title(f"Pie Chart of {col}")
            plt.ylabel('')
            plt.show()

    def plot_scatter_plots(self):
        """Plots scatter plots for bivariate analysis."""
        print("\n--- Scatter Plots ---")

        # Scatter plot for numerical vs. numerical
        plt.figure()
        sns.scatterplot(x='LATITUDE', y='LONGITUDE', data=self.data)
        plt.title("LATITUDE vs LONGITUDE")
        plt.show()

        # Alternative plots for numerical vs. categorical
        plt.figure()
        sns.boxplot(x='ACCLASS', y='TIME', data=self.data)
        plt.title("TIME vs ACCLASS")
        plt.show()

        plt.figure()
        sns.boxplot(x='ROAD_CLASS', y='TIME', data=self.data)
        plt.title("TIME vs ROAD_CLASS")
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
        plt.show()

    def plot_box_plots(self):
        """Plots box plots for bivariate analysis."""
        print("\n--- Box Plots ---")
        box_pairs = [('TIME', 'DIVISION'), ('ACCLASS', 'INVAGE'), ('TIME', 'LIGHT')]
        for x, y in box_pairs:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=x, y=y, data=self.data)
            plt.title(f"{x} vs {y}")
            plt.xticks(rotation=45)
            plt.show()

    def plot_grouped_bar_plots(self):
        """Plots grouped bar plots for bivariate analysis."""
        print("\n--- Grouped Bar Plots ---")
        grouped_cols = [('ACCLASS', 'ROAD_CLASS'), ('ACCLASS', 'DISTRICT'), ('ACCLASS', 'VISIBILITY'),
                        ('ACCLASS', 'LIGHT'), ('ACCLASS', 'RDSFCOND'), ('DIVISION', 'ACCLASS')]
        for x, hue in grouped_cols:
            plt.figure(figsize=(12, 6))
            sns.countplot(x=x, hue=hue, data=self.data)
            plt.title(f"{x} vs {hue}")
            plt.xticks(rotation=45)
            plt.show()

    def plot_class_distribution(self, y, title="Class Distribution"):
        """Plots the class distribution of the target variable using seaborn."""
        plt.figure(figsize=(8, 6))

        # Replace 0 and 1 with "Non-fatal" and "Fatal"
        labels = ["Non-fatal" if val == 0 else "Fatal" for val in sorted(pd.Series(y).unique())]
        counts = pd.Series(y).value_counts().sort_index()

        # Use seaborn to plot the bar chart with different colors
        sns.barplot(x=labels, y=counts, palette=["skyblue", "salmon"])

        plt.title(title)
        plt.xlabel("Accident Severity")
        plt.ylabel("Count")
        plt.xticks(rotation=0)  # Ensure labels are not rotated
        plt.show()

    def plot_time_series(self):
        """Plots time series if DATE is available and converted."""
        print("\n--- Time Series Plots ---")
        if 'DATE' in self.data.columns:
            try:
                temp_df = self.data.copy()
                temp_df['DATE'] = pd.to_datetime(temp_df['DATE'])
                temp_df['DATE'] = temp_df['DATE'].dt.date
                daily_counts = temp_df['DATE'].value_counts().sort_index()
                plt.figure(figsize=(12, 6))
                daily_counts.plot()
                plt.title("Daily Accident Counts")
                plt.show()

                fatal_counts = temp_df[temp_df['ACCLASS'] == 'Fatal']['DATE'].value_counts().sort_index()
                plt.figure(figsize=(12, 6))
                fatal_counts.plot()
                plt.title("Daily Fatal Accident Counts")
                plt.show()

                temp_df['DAY_OF_WEEK'] = temp_df['DATE'].dt.day_name()
                plt.figure(figsize=(12, 6))
                sns.countplot(x='DAY_OF_WEEK', data=temp_df, order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
                plt.title("Accident Counts by Day of Week")
                plt.show()

            except Exception as e:
                print(f"Error in time series plotting: {e}")

    def plot_missing_percentages(self):
        """Plots bar plots for missing percentages."""
        print("\n--- Missing Percentages ---")
        missing_percentage = (self.data.isnull().sum() / len(self.data)) * 100
        missing_percentage = missing_percentage[missing_percentage > 0]
        plt.figure(figsize=(12, 6))
        missing_percentage.plot(kind='bar')
        plt.title("Missing Percentages")
        plt.show()

    def plot_feature_importance(self, feature_importance, title="Feature Importance", n=20):  # add n parameter.
        """Plots the top N feature importances."""
        top_n_features = feature_importance.nlargest(n)

        plt.figure(figsize=(12, 6))  # Increase width
        sns.barplot(x=top_n_features.values, y=top_n_features.index)
        plt.title(title)
        plt.xlabel('Feature Importance Score')
        plt.ylabel('Features')
        plt.subplots_adjust(left=0.3)  # Adjust left margin
        plt.show()

    def plot_training_times(self, training_times):
        """Creates a bar plot of training times using Seaborn."""
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(training_times.keys()), y=list(training_times.values()))
        plt.title("Model Training Times")
        plt.xlabel("Model")
        plt.ylabel("Training Time (seconds)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

    def plot_grouped_bar_plots(self):
        """Plots grouped bar plots for bivariate analysis."""
        print("\n--- Grouped Bar Plots ---")
        grouped_cols = [
            ('ACCLASS', 'ROAD_CLASS'),
            ('ACCLASS', 'DISTRICT'),
            ('ACCLASS', 'VISIBILITY'),  # Incident type vs. visibility
            ('ACCLASS', 'LIGHT'),  # Incident type vs. light
            ('ACCLASS', 'RDSFCOND'),  # Incident type vs. road surface condition
            ('DIVISION', 'ACCLASS')
        ]
        for x, hue in grouped_cols:
            plt.figure(figsize=(12, 6))
            sns.countplot(x=x, hue=hue, data=self.data)
            plt.title(f"{x} vs {hue}")
            plt.xticks(rotation=45)
            plt.show()

    def plot_heatmaps(self):
        """Plots heatmaps for bivariate analysis."""
        print("\n--- Heatmaps ---")

        # Function to create a single heatmap
        def create_heatmap(data, title):
            plt.figure(figsize=(8, 6))
            sns.heatmap(data, annot=True, cmap='viridis', fmt=".0f")
            plt.title(title)
            plt.show()

        # Heatmap of accident counts by VISIBILITY and ACCLASS
        heatmap_data_vis = self.data.groupby(['VISIBILITY', 'ACCLASS']).size().unstack()
        create_heatmap(heatmap_data_vis, "Accident Counts by VISIBILITY and ACCLASS")

        # Heatmap of accident counts by LIGHT and ACCLASS
        heatmap_data_light = self.data.groupby(['LIGHT', 'ACCLASS']).size().unstack()
        create_heatmap(heatmap_data_light, "Accident Counts by LIGHT and ACCLASS")

        # Heatmap of accident counts by RDSFCOND and ACCLASS
        heatmap_data_rdsfcond = self.data.groupby(['RDSFCOND', 'ACCLASS']).size().unstack()
        create_heatmap(heatmap_data_rdsfcond, "Accident Counts by RDSFCOND and ACCLASS")

        # Heatmap of accident counts by VISIBILITY and LIGHT
        heatmap_data_vis_light = self.data.groupby(['VISIBILITY', 'LIGHT']).size().unstack()
        create_heatmap(heatmap_data_vis_light, "Accident Counts by VISIBILITY and LIGHT")
