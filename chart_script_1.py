import plotly.graph_objects as go
import plotly.io as pio

# Data
features = ["Sentiment_Score", "Loan Account", "TechSupport Availed", "Interest Deposited", "Balance_Ratio", "Yearly Average Balance (USD)", "FDs", "Dependents"]
importance = [0.8045, 0.7182, 0.6036, 0.5912, 0.5893, 0.4378, 0.4364, 0.3429]

# Abbreviate feature names to meet 15 character limit while keeping as close to original as possible
abbreviated_features = []
for feature in features:
    if feature == "TechSupport Availed":
        abbreviated_features.append("TechSupport Ave")  # 15 chars
    elif feature == "Interest Deposited":
        abbreviated_features.append("Interest Dep")  # 12 chars
    elif feature == "Yearly Average Balance (USD)":
        abbreviated_features.append("Avg Balance USD")  # 15 chars
    else:
        abbreviated_features.append(feature)

# Reverse the order so highest importance is at the top
features_reversed = abbreviated_features[::-1]
importance_reversed = importance[::-1]

# Create horizontal bar chart
fig = go.Figure(go.Bar(
    x=importance_reversed,
    y=features_reversed,
    orientation='h',
    marker_color='#5D878F',  # Professional blue color from brand palette
    text=[f'{val:.3f}' for val in importance_reversed],  # Add importance values as labels
    textposition='outside',
    cliponaxis=False
))

# Update layout with shortened title to meet 40 character limit
fig.update_layout(
    title="XYZ Bank Churn Feature Importance",  # 33 characters
    xaxis_title="Importance",
    yaxis_title="Features"
)

# Update axes
fig.update_xaxes(range=[0, max(importance) * 1.1])  # Add some space for text labels
fig.update_yaxes(categoryorder='array', categoryarray=features_reversed)

# Save the chart
fig.write_image("churn_feature_importance.png")