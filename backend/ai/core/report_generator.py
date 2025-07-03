"""
Report Generator Module

This module generates comprehensive financial reports using Plotly and outputs HTML or PDF using Jinja2 templates. It can output both HTML and PDF reports
with monthly financial summaries and insights.
"""

import os
import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from jinja2 import Environment, FileSystemLoader, Template

# Try to import PDF generation libraries
try:
    import weasyprint
    WEASYPRINT_AVAILABLE = True
except ImportError:
    WEASYPRINT_AVAILABLE = False
    logging.warning("weasyprint not available, PDF generation disabled")

try:
    from pdfkit import from_string
    PDFKIT_AVAILABLE = True
except ImportError:
    PDFKIT_AVAILABLE = False
    logging.warning("pdfkit not available, PDF generation disabled")

logger = logging.getLogger(__name__)


@dataclass
class ReportData:
    """Represents financial data for report generation"""
    monthly_data: List[Dict[str, Any]]
    summary: Dict[str, Any]
    categories: List[Dict[str, Any]]
    trends: Dict[str, Any]
    insights: List[Dict[str, Any]]
    goals: List[Dict[str, Any]]
    anomalies: List[Dict[str, Any]]


@dataclass
class ReportConfig:
    """Configuration for report generation"""
    language: str = 'en'
    format: str = 'html'  # 'html' or 'pdf'
    include_charts: bool = True
    include_insights: bool = True
    include_goals: bool = True
    include_anomalies: bool = True
    theme: str = 'light'  # 'light' or 'dark'
    custom_css: Optional[str] = None


class ReportGenerator:
    """
    Comprehensive financial report generator with visualizations and insights.
    
    This module creates professional financial reports with interactive charts,
    insights, and recommendations using Plotly and Jinja2 templates.
    """
    
    def __init__(self, templates_dir: str = None, output_dir: str = None):
        """
        Initialize the report generator.
        
        Args:
            templates_dir: Directory containing Jinja2 templates
            output_dir: Directory for generated reports
        """
        self.templates_dir = templates_dir or "backend/ai/templates/reports"
        self.output_dir = output_dir or "backend/static/reports"
        
        # Ensure directories exist
        os.makedirs(self.templates_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize Jinja2 environment
        self.jinja_env = Environment(
            loader=FileSystemLoader(self.templates_dir),
            autoescape=True
        )
        
        # Create default templates if they don't exist
        self._create_default_templates()
        
        # Chart color schemes
        self.color_schemes = {
            'light': {
                'primary': '#1f77b4',
                'secondary': '#ff7f0e',
                'success': '#2ca02c',
                'danger': '#d62728',
                'warning': '#ff7f0e',
                'info': '#17a2b8',
                'background': '#ffffff',
                'text': '#333333'
            },
            'dark': {
                'primary': '#6366f1',
                'secondary': '#f59e0b',
                'success': '#10b981',
                'danger': '#ef4444',
                'warning': '#f59e0b',
                'info': '#06b6d4',
                'background': '#1f2937',
                'text': '#f9fafb'
            }
        }
    
    def _create_default_templates(self):
        """Create default Jinja2 templates for reports."""
        templates = {
            'monthly_report.html': self._get_monthly_report_template(),
            'financial_summary.html': self._get_financial_summary_template(),
            'base.html': self._get_base_template()
        }
        
        for template_name, template_content in templates.items():
            template_path = os.path.join(self.templates_dir, template_name)
            if not os.path.exists(template_path):
                with open(template_path, 'w', encoding='utf-8') as f:
                    f.write(template_content)
                logger.info(f"Created default template: {template_name}")
    
    def _get_base_template(self) -> str:
        """Get the base HTML template."""
        return """
<!DOCTYPE html>
<html lang="{{ language }}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: {{ colors.background }};
            color: {{ colors.text }};
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
            border-bottom: 2px solid {{ colors.primary }};
            padding-bottom: 20px;
        }
        .section {
            margin-bottom: 30px;
            padding: 20px;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
        }
        .section h2 {
            color: {{ colors.primary }};
            margin-top: 0;
        }
        .metric {
            display: inline-block;
            margin: 10px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
            text-align: center;
            min-width: 120px;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: {{ colors.primary }};
        }
        .metric-label {
            font-size: 14px;
            color: #666;
        }
        .chart-container {
            margin: 20px 0;
            text-align: center;
        }
        .insight {
            background: #e3f2fd;
            border-left: 4px solid {{ colors.info }};
            padding: 15px;
            margin: 10px 0;
            border-radius: 4px;
        }
        .goal {
            background: #e8f5e8;
            border-left: 4px solid {{ colors.success }};
            padding: 15px;
            margin: 10px 0;
            border-radius: 4px;
        }
        .anomaly {
            background: #ffebee;
            border-left: 4px solid {{ colors.danger }};
            padding: 15px;
            margin: 10px 0;
            border-radius: 4px;
        }
        .footer {
            text-align: center;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #e0e0e0;
            color: #666;
        }
        {% if custom_css %}
        {{ custom_css }}
        {% endif %}
    </style>
</head>
<body>
    <div class="container">
        {% block content %}{% endblock %}
    </div>
</body>
</html>
"""
    
    def _get_monthly_report_template(self) -> str:
        """Get the monthly report template."""
        return """
{% extends "base.html" %}

{% block content %}
<div class="header">
    <h1>{{ title }}</h1>
    <p>{{ subtitle }}</p>
    <p>Generated on {{ generated_date }}</p>
</div>

<div class="section">
    <h2>📊 Financial Summary</h2>
    <div class="metrics">
        <div class="metric">
            <div class="metric-value">${{ "%.2f"|format(summary.total_income) }}</div>
            <div class="metric-label">Total Income</div>
        </div>
        <div class="metric">
            <div class="metric-value">${{ "%.2f"|format(summary.total_expenses) }}</div>
            <div class="metric-label">Total Expenses</div>
        </div>
        <div class="metric">
            <div class="metric-value">${{ "%.2f"|format(summary.net_balance) }}</div>
            <div class="metric-label">Net Balance</div>
        </div>
        <div class="metric">
            <div class="metric-value">{{ summary.total_transactions }}</div>
            <div class="metric-label">Transactions</div>
        </div>
    </div>
</div>

{% if include_charts %}
<div class="section">
    <h2>📈 Monthly Trends</h2>
    <div class="chart-container">
        <div id="monthly-trends-chart"></div>
    </div>
</div>

<div class="section">
    <h2>💸 Expense Categories</h2>
    <div class="chart-container">
        <div id="expense-categories-chart"></div>
    </div>
</div>

<div class="section">
    <h2>💰 Income vs Expenses</h2>
    <div class="chart-container">
        <div id="income-expenses-chart"></div>
    </div>
</div>
{% endif %}

{% if include_insights and insights %}
<div class="section">
    <h2>💡 Key Insights</h2>
    {% for insight in insights %}
    <div class="insight">
        <strong>{{ insight.title }}</strong><br>
        {{ insight.description }}
        {% if insight.recommendation %}
        <br><em>Recommendation: {{ insight.recommendation }}</em>
        {% endif %}
    </div>
    {% endfor %}
</div>
{% endif %}

{% if include_goals and goals %}
<div class="section">
    <h2>🎯 Goal Progress</h2>
    {% for goal in goals %}
    <div class="goal">
        <strong>{{ goal.name }}</strong><br>
        Progress: {{ goal.progress }}% ({{ goal.current }}/{{ goal.target }})
        {% if goal.status == 'on_track' %}
        <span style="color: green;">✓ On Track</span>
        {% elif goal.status == 'at_risk' %}
        <span style="color: orange;">⚠ At Risk</span>
        {% else %}
        <span style="color: red;">✗ Behind</span>
        {% endif %}
    </div>
    {% endfor %}
</div>
{% endif %}

{% if include_anomalies and anomalies %}
<div class="section">
    <h2>⚠️ Anomalies Detected</h2>
    {% for anomaly in anomalies %}
    <div class="anomaly">
        <strong>{{ anomaly.title }}</strong><br>
        {{ anomaly.description }}
        {% if anomaly.severity == 'high' %}
        <span style="color: red;">High Priority</span>
        {% elif anomaly.severity == 'medium' %}
        <span style="color: orange;">Medium Priority</span>
        {% else %}
        <span style="color: blue;">Low Priority</span>
        {% endif %}
    </div>
    {% endfor %}
</div>
{% endif %}

<div class="footer">
    <p>This report was generated automatically by FinancialHub AI</p>
    <p>For questions or support, please contact our team</p>
</div>

{% if include_charts %}
<script>
    // Monthly trends chart
    var monthlyData = {{ monthly_chart_data | safe }};
    Plotly.newPlot('monthly-trends-chart', monthlyData.data, monthlyData.layout);
    
    // Expense categories chart
    var categoriesData = {{ categories_chart_data | safe }};
    Plotly.newPlot('expense-categories-chart', categoriesData.data, categoriesData.layout);
    
    // Income vs expenses chart
    var incomeExpensesData = {{ income_expenses_chart_data | safe }};
    Plotly.newPlot('income-expenses-chart', incomeExpensesData.data, incomeExpensesData.layout);
</script>
{% endif %}
{% endblock %}
"""
    
    def _get_financial_summary_template(self) -> str:
        """Get the financial summary template."""
        return """
{% extends "base.html" %}

{% block content %}
<div class="header">
    <h1>Financial Summary Report</h1>
    <p>Period: {{ period }}</p>
    <p>Generated on {{ generated_date }}</p>
</div>

<div class="section">
    <h2>Quick Overview</h2>
    <div class="metrics">
        <div class="metric">
            <div class="metric-value">${{ "%.2f"|format(summary.net_balance) }}</div>
            <div class="metric-label">Current Balance</div>
        </div>
        <div class="metric">
            <div class="metric-value">{{ summary.savings_rate }}%</div>
            <div class="metric-label">Savings Rate</div>
        </div>
        <div class="metric">
            <div class="metric-value">{{ summary.top_category }}</div>
            <div class="metric-label">Top Expense</div>
        </div>
    </div>
</div>

<div class="section">
    <h2>Key Metrics</h2>
    <ul>
        <li>Average monthly income: ${{ "%.2f"|format(summary.avg_monthly_income) }}</li>
        <li>Average monthly expenses: ${{ "%.2f"|format(summary.avg_monthly_expenses) }}</li>
        <li>Total transactions this period: {{ summary.total_transactions }}</li>
        <li>Most active spending day: {{ summary.most_active_day }}</li>
    </ul>
</div>

{% if insights %}
<div class="section">
    <h2>Insights</h2>
    {% for insight in insights %}
    <div class="insight">
        {{ insight }}
    </div>
    {% endfor %}
</div>
{% endif %}
{% endblock %}
"""
    
    def generate_monthly_report(self, 
                              report_data: ReportData,
                              config: ReportConfig = None) -> str:
        """
        Generate a comprehensive monthly financial report.
        
        Args:
            report_data: Financial data for the report
            config: Report configuration
            
        Returns:
            Generated report content (HTML or PDF)
        """
        if config is None:
            config = ReportConfig()
        
        logger.info(f"Generating monthly report in {config.format} format")
        
        # Prepare template context
        context = self._prepare_report_context(report_data, config)
        
        # Generate charts if requested
        if config.include_charts:
            context.update(self._generate_charts(report_data, config))
        
        # Render template
        template = self.jinja_env.get_template('monthly_report.html')
        html_content = template.render(**context)
        
        # Convert to PDF if requested
        if config.format == 'pdf':
            return self._convert_to_pdf(html_content, config)
        
        return html_content
    
    def _prepare_report_context(self, 
                               report_data: ReportData, 
                               config: ReportConfig) -> Dict[str, Any]:
        """Prepare context data for template rendering."""
        colors = self.color_schemes.get(config.theme, self.color_schemes['light'])
        
        # Calculate additional metrics
        summary = report_data.summary.copy()
        if 'savings_rate' not in summary:
            summary['savings_rate'] = self._calculate_savings_rate(summary)
        
        return {
            'title': 'Monthly Financial Report',
            'subtitle': f"Period: {datetime.now().strftime('%B %Y')}",
            'generated_date': datetime.now().strftime('%B %d, %Y at %I:%M %p'),
            'language': config.language,
            'colors': colors,
            'custom_css': config.custom_css,
            'summary': summary,
            'monthly_data': report_data.monthly_data,
            'categories': report_data.categories,
            'trends': report_data.trends,
            'insights': report_data.insights if config.include_insights else [],
            'goals': report_data.goals if config.include_goals else [],
            'anomalies': report_data.anomalies if config.include_anomalies else [],
            'include_charts': config.include_charts
        }
    
    def _generate_charts(self, 
                        report_data: ReportData, 
                        config: ReportConfig) -> Dict[str, Any]:
        """Generate Plotly charts for the report."""
        colors = self.color_schemes.get(config.theme, self.color_schemes['light'])
        
        charts = {}
        
        # Monthly trends chart
        if report_data.monthly_data:
            charts['monthly_chart_data'] = self._create_monthly_trends_chart(
                report_data.monthly_data, colors
            )
        
        # Expense categories chart
        if report_data.categories:
            charts['categories_chart_data'] = self._create_expense_categories_chart(
                report_data.categories, colors
            )
        
        # Income vs expenses chart
        if report_data.monthly_data:
            charts['income_expenses_chart_data'] = self._create_income_expenses_chart(
                report_data.monthly_data, colors
            )
        
        return charts
    
    def _create_monthly_trends_chart(self, 
                                   monthly_data: List[Dict[str, Any]], 
                                   colors: Dict[str, str]) -> Dict[str, Any]:
        """Create monthly trends chart."""
        df = pd.DataFrame(monthly_data)
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Monthly Income & Expenses', 'Monthly Balance'),
            vertical_spacing=0.1
        )
        
        # Income and expenses
        fig.add_trace(
            go.Scatter(
                x=df['month'],
                y=df['income'],
                name='Income',
                line=dict(color=colors['success'], width=3),
                fill='tonexty'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['month'],
                y=df['expenses'],
                name='Expenses',
                line=dict(color=colors['danger'], width=3),
                fill='tonexty'
            ),
            row=1, col=1
        )
        
        # Balance
        fig.add_trace(
            go.Scatter(
                x=df['month'],
                y=df['balance'],
                name='Balance',
                line=dict(color=colors['primary'], width=3),
                fill='tonexty'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=600,
            showlegend=True,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color=colors['text'])
        )
        
        return fig.to_dict()
    
    def _create_expense_categories_chart(self, 
                                       categories: List[Dict[str, Any]], 
                                       colors: Dict[str, str]) -> Dict[str, Any]:
        """Create expense categories pie chart."""
        labels = [cat['name'] for cat in categories]
        values = [cat['amount'] for cat in categories]
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.3,
            marker_colors=px.colors.qualitative.Set3
        )])
        
        fig.update_layout(
            title='Expense Distribution by Category',
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color=colors['text'])
        )
        
        return fig.to_dict()
    
    def _create_income_expenses_chart(self, 
                                    monthly_data: List[Dict[str, Any]], 
                                    colors: Dict[str, str]) -> Dict[str, Any]:
        """Create income vs expenses bar chart."""
        df = pd.DataFrame(monthly_data)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Income',
            x=df['month'],
            y=df['income'],
            marker_color=colors['success']
        ))
        
        fig.add_trace(go.Bar(
            name='Expenses',
            x=df['month'],
            y=df['expenses'],
            marker_color=colors['danger']
        ))
        
        fig.update_layout(
            title='Income vs Expenses by Month',
            barmode='group',
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color=colors['text'])
        )
        
        return fig.to_dict()
    
    def _calculate_savings_rate(self, summary: Dict[str, Any]) -> float:
        """Calculate savings rate percentage."""
        total_income = summary.get('total_income', 0)
        total_expenses = summary.get('total_expenses', 0)
        
        if total_income == 0:
            return 0.0
        
        savings = total_income - total_expenses
        return round((savings / total_income) * 100, 1)
    
    def _convert_to_pdf(self, html_content: str, config: ReportConfig) -> str:
        """Convert HTML content to PDF."""
        if not WEASYPRINT_AVAILABLE and not PDFKIT_AVAILABLE:
            logger.warning("PDF generation not available, returning HTML")
            return html_content
        
        try:
            if WEASYPRINT_AVAILABLE:
                # Use WeasyPrint for PDF generation
                pdf = weasyprint.HTML(string=html_content).write_pdf()
                return pdf
            elif PDFKIT_AVAILABLE:
                # Use pdfkit as fallback
                pdf = from_string(html_content, False)
                return pdf
        except Exception as e:
            logger.error(f"Error generating PDF: {e}")
            return html_content
    
    def save_report(self, 
                   content: str, 
                   filename: str, 
                   config: ReportConfig) -> str:
        """
        Save the generated report to disk.
        
        Args:
            content: Report content
            filename: Output filename
            config: Report configuration
            
        Returns:
            Path to saved file
        """
        # Determine file extension
        extension = 'pdf' if config.format == 'pdf' else 'html'
        full_filename = f"{filename}.{extension}"
        
        # Create output path
        output_path = os.path.join(self.output_dir, full_filename)
        
        # Save file
        mode = 'wb' if config.format == 'pdf' else 'w'
        encoding = None if config.format == 'pdf' else 'utf-8'
        
        with open(output_path, mode, encoding=encoding) as f:
            f.write(content)
        
        logger.info(f"Report saved to: {output_path}")
        return output_path
    
    def generate_quick_summary(self, 
                             report_data: ReportData,
                             config: ReportConfig = None) -> str:
        """
        Generate a quick financial summary report.
        
        Args:
            report_data: Financial data
            config: Report configuration
            
        Returns:
            Generated summary content
        """
        if config is None:
            config = ReportConfig()
        
        context = {
            'period': datetime.now().strftime('%B %Y'),
            'generated_date': datetime.now().strftime('%B %d, %Y'),
            'summary': report_data.summary,
            'insights': report_data.insights[:3] if report_data.insights else [],
            'colors': self.color_schemes.get(config.theme, self.color_schemes['light'])
        }
        
        template = self.jinja_env.get_template('financial_summary.html')
        return template.render(**context)
    
    def create_sample_data(self) -> ReportData:
        """Create sample data for testing and demonstration."""
        monthly_data = [
            {
                'month': '2024-01',
                'income': 5000.0,
                'expenses': 3500.0,
                'balance': 1500.0,
                'transactions': 45
            },
            {
                'month': '2024-02',
                'income': 5200.0,
                'expenses': 3800.0,
                'balance': 1400.0,
                'transactions': 52
            },
            {
                'month': '2024-03',
                'income': 4800.0,
                'expenses': 3200.0,
                'balance': 1600.0,
                'transactions': 38
            }
        ]
        
        summary = {
            'total_income': 15000.0,
            'total_expenses': 10500.0,
            'net_balance': 4500.0,
            'total_transactions': 135,
            'avg_monthly_income': 5000.0,
            'avg_monthly_expenses': 3500.0,
            'top_category': 'Food & Dining'
        }
        
        categories = [
            {'name': 'Food & Dining', 'amount': 2500.0},
            {'name': 'Transportation', 'amount': 1800.0},
            {'name': 'Entertainment', 'amount': 1200.0},
            {'name': 'Shopping', 'amount': 2000.0},
            {'name': 'Utilities', 'amount': 800.0}
        ]
        
        insights = [
            {
                'title': 'High Food Spending',
                'description': 'Your food spending is 25% above average for your income level.',
                'recommendation': 'Consider meal planning to reduce dining out expenses.'
            },
            {
                'title': 'Good Savings Rate',
                'description': 'You\'re saving 30% of your income, which is excellent!',
                'recommendation': 'Keep up the good work and consider increasing your emergency fund.'
            }
        ]
        
        goals = [
            {
                'name': 'Emergency Fund',
                'current': 8000.0,
                'target': 10000.0,
                'progress': 80.0,
                'status': 'on_track'
            },
            {
                'name': 'Vacation Fund',
                'current': 2000.0,
                'target': 5000.0,
                'progress': 40.0,
                'status': 'at_risk'
            }
        ]
        
        anomalies = [
            {
                'title': 'Unusual Large Transaction',
                'description': 'A $500 transaction on March 15th was 3x your average transaction size.',
                'severity': 'medium'
            }
        ]
        
        trends = {
            'income_trend': 'stable',
            'expense_trend': 'decreasing',
            'savings_trend': 'increasing'
        }
        
        return ReportData(
            monthly_data=monthly_data,
            summary=summary,
            categories=categories,
            trends=trends,
            insights=insights,
            goals=goals,
            anomalies=anomalies
        )
