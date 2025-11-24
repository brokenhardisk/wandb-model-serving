import wandb
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import os

class WandBVisualizer:
    def __init__(self, wandb_entity, wandb_project):
        self.entity = wandb_entity
        self.project = wandb_project
        self.api = wandb.Api()
    
    def get_project_runs(self):
        """Get all runs from the project"""
        try:
            runs = self.api.runs(f"{self.entity}/{self.project}")
            return runs
        except Exception as e:
            print(f"Error fetching runs from W&B: {e}")
            return []
    
    def get_run_metrics(self, run_id):
        """Get metrics for a specific run"""
        try:
            run = self.api.run(f"{self.entity}/{self.project}/{run_id}")
            history = run.history()
            return history, run.config, run.summary
        except Exception as e:
            print(f"Error fetching run metrics: {e}")
            return None, None, None
    
    def create_training_plots(self, history):
        """Create Plotly charts from training history"""
        if history is None or history.empty:
            return None
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Training Loss', 'Training Accuracy')
        )
        
        # Loss plot
        if 'loss' in history.columns and '_step' in history.columns:
            fig.add_trace(
                go.Scatter(x=history['_step'], y=history['loss'], 
                          name='Training Loss', line=dict(color='blue')),
                row=1, col=1
            )
        
        if 'val_loss' in history.columns:
            fig.add_trace(
                go.Scatter(x=history['_step'], y=history['val_loss'], 
                          name='Validation Loss', line=dict(color='red')),
                row=1, col=1
            )
        
        # Accuracy plot
        if 'accuracy' in history.columns:
            fig.add_trace(
                go.Scatter(x=history['_step'], y=history['accuracy'], 
                          name='Training Accuracy', line=dict(color='green')),
                row=1, col=2
            )
        
        if 'val_accuracy' in history.columns:
            fig.add_trace(
                go.Scatter(x=history['_step'], y=history['val_accuracy'], 
                          name='Validation Accuracy', line=dict(color='orange')),
                row=1, col=2
            )
        
        fig.update_layout(height=400, showlegend=True)
        return fig
    
    def get_latest_run_metrics(self):
        """Get metrics from the latest run"""
        runs = self.get_project_runs()
        if not runs:
            return None, None, None
        
        latest_run = sorted(runs, key=lambda x: x.created_at, reverse=True)[0]
        return self.get_run_metrics(latest_run.id)