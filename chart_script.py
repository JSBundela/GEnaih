import plotly.graph_objects as go

# Data for churn distribution
labels = ['No Churn', 'Churn']
values = [732, 251]
# Using blue and orange professional color scheme as requested
colors = ['#2E86AB', '#F18F01']  # Professional blue and orange

# Create pie chart
fig = go.Figure(data=[go.Pie(
    labels=labels, 
    values=values,
    marker_colors=colors,
    textinfo='label+percent',
    textposition='inside'
)])

# Update layout following the instructions
fig.update_layout(
    title="XYZ Bank Churn Prediction System",
    uniformtext_minsize=14, 
    uniformtext_mode='hide'
)

# Save the chart
fig.write_image("churn_dashboard.png")