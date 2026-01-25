"""
Report Generator

Creates visual reports from backtest results.
Generates interactive Plotly charts and summary statistics.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional
import json

from .backtest_engine import BacktestResult, Trade


class ReportGenerator:
    """Generate visual backtest reports"""
    
    # Color scheme - Modern dark theme
    COLORS = {
        "background": "#0d1117",
        "paper": "#161b22",
        "text": "#c9d1d9",
        "text_muted": "#8b949e",
        "grid": "#30363d",
        "profit": "#3fb950",
        "loss": "#f85149",
        "neutral": "#58a6ff",
        "accent": "#a371f7",
        "warning": "#d29922",
    }
    
    def __init__(self, result: BacktestResult):
        """
        Initialize report generator.
        
        Args:
            result: BacktestResult from backtest engine
        """
        self.result = result
    
    def generate_equity_curve(self) -> go.Figure:
        """Generate equity curve chart"""
        if self.result.equity_curve.empty:
            return self._empty_chart("No equity data")
        
        df = self.result.equity_curve
        
        # Calculate drawdown
        equity = df["equity"]
        rolling_max = equity.expanding().max()
        drawdown = (equity - rolling_max) / rolling_max * 100
        
        # Create figure with secondary y-axis
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
            subplot_titles=("资金曲线", "回撤")
        )
        
        # Equity curve
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["equity"],
                mode="lines",
                name="资金",
                line=dict(color=self.COLORS["neutral"], width=2),
                fill="tozeroy",
                fillcolor="rgba(88, 166, 255, 0.1)"
            ),
            row=1, col=1
        )
        
        # Initial capital line
        fig.add_hline(
            y=self.result.initial_capital,
            line_dash="dash",
            line_color=self.COLORS["text_muted"],
            annotation_text="初始资金",
            row=1, col=1
        )
        
        # Mark trades
        for trade in self.result.trades:
            if not trade.is_open:
                color = self.COLORS["profit"] if trade.pnl > 0 else self.COLORS["loss"]
                # Entry marker
                fig.add_trace(
                    go.Scatter(
                        x=[trade.entry_time],
                        y=[df[df["timestamp"] == trade.entry_time]["equity"].values[0] if len(df[df["timestamp"] == trade.entry_time]) > 0 else None],
                        mode="markers",
                        marker=dict(symbol="triangle-up", size=10, color=self.COLORS["profit"]),
                        name="买入",
                        showlegend=False,
                        hovertemplate=f"买入<br>价格: {trade.entry_price:.2f}<extra></extra>"
                    ),
                    row=1, col=1
                )
                # Exit marker
                fig.add_trace(
                    go.Scatter(
                        x=[trade.exit_time],
                        y=[df[df["timestamp"] == trade.exit_time]["equity"].values[0] if len(df[df["timestamp"] == trade.exit_time]) > 0 else None],
                        mode="markers",
                        marker=dict(symbol="triangle-down", size=10, color=color),
                        name="卖出",
                        showlegend=False,
                        hovertemplate=f"卖出 ({trade.exit_reason})<br>价格: {trade.exit_price:.2f}<br>盈亏: {trade.pnl_percent:+.2f}%<extra></extra>"
                    ),
                    row=1, col=1
                )
        
        # Drawdown chart
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=drawdown,
                mode="lines",
                name="回撤",
                line=dict(color=self.COLORS["loss"], width=1),
                fill="tozeroy",
                fillcolor="rgba(248, 81, 73, 0.2)"
            ),
            row=2, col=1
        )
        
        # Update layout
        self._apply_dark_theme(fig)
        fig.update_layout(
            title=dict(
                text=f"📈 {self.result.strategy_name} - 资金曲线",
                font=dict(size=20)
            ),
            height=600,
            showlegend=False
        )
        
        fig.update_yaxes(title_text="资金 (USDT)", row=1, col=1)
        fig.update_yaxes(title_text="回撤 (%)", row=2, col=1)
        
        return fig
    
    def generate_trade_analysis(self) -> go.Figure:
        """Generate trade analysis charts"""
        closed_trades = [t for t in self.result.trades if not t.is_open]
        
        if not closed_trades:
            return self._empty_chart("No trades to analyze")
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "交易盈亏分布", 
                "累计盈亏", 
                "持仓时间分布",
                "出场原因统计"
            ),
            specs=[
                [{"type": "bar"}, {"type": "scatter"}],
                [{"type": "histogram"}, {"type": "pie"}]
            ]
        )
        
        # 1. Trade PnL distribution
        pnls = [t.pnl_percent for t in closed_trades]
        colors = [self.COLORS["profit"] if p > 0 else self.COLORS["loss"] for p in pnls]
        
        fig.add_trace(
            go.Bar(
                x=list(range(1, len(pnls) + 1)),
                y=pnls,
                marker_color=colors,
                name="盈亏 %",
                hovertemplate="交易 #%{x}<br>盈亏: %{y:.2f}%<extra></extra>"
            ),
            row=1, col=1
        )
        
        # 2. Cumulative PnL
        cum_pnl = np.cumsum(pnls)
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(cum_pnl) + 1)),
                y=cum_pnl,
                mode="lines+markers",
                name="累计盈亏",
                line=dict(color=self.COLORS["accent"], width=2),
                marker=dict(size=6)
            ),
            row=1, col=2
        )
        fig.add_hline(y=0, line_dash="dash", line_color=self.COLORS["text_muted"], row=1, col=2)
        
        # 3. Duration distribution
        durations = [t.duration.total_seconds() / 3600 for t in closed_trades if t.duration]  # Hours
        if durations:
            fig.add_trace(
                go.Histogram(
                    x=durations,
                    nbinsx=20,
                    marker_color=self.COLORS["neutral"],
                    name="持仓时间"
                ),
                row=2, col=1
            )
        
        # 4. Exit reason pie chart
        exit_reasons = {}
        for t in closed_trades:
            reason = t.exit_reason or "未知"
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
        
        fig.add_trace(
            go.Pie(
                labels=list(exit_reasons.keys()),
                values=list(exit_reasons.values()),
                marker_colors=[self.COLORS["profit"], self.COLORS["loss"], self.COLORS["neutral"], self.COLORS["accent"]],
                textinfo="label+percent",
                hole=0.4
            ),
            row=2, col=2
        )
        
        # Update layout
        self._apply_dark_theme(fig)
        fig.update_layout(
            title=dict(
                text=f"📊 {self.result.strategy_name} - 交易分析",
                font=dict(size=20)
            ),
            height=700,
            showlegend=False
        )
        
        fig.update_xaxes(title_text="交易序号", row=1, col=1)
        fig.update_yaxes(title_text="盈亏 (%)", row=1, col=1)
        fig.update_xaxes(title_text="交易序号", row=1, col=2)
        fig.update_yaxes(title_text="累计盈亏 (%)", row=1, col=2)
        fig.update_xaxes(title_text="持仓时间 (小时)", row=2, col=1)
        fig.update_yaxes(title_text="交易次数", row=2, col=1)
        
        return fig
    
    def generate_metrics_card(self) -> go.Figure:
        """Generate a metrics summary card"""
        metrics = self.result.to_summary_dict()
        
        # Create a table figure
        fig = go.Figure()
        
        # Prepare data for display
        labels = list(metrics.keys())
        values = list(metrics.values())
        
        # Split into two columns
        mid = len(labels) // 2
        
        # Create indicator cards
        fig = make_subplots(
            rows=3, cols=4,
            specs=[[{"type": "indicator"}] * 4] * 3,
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        key_metrics = [
            ("总收益率", self.result.total_return_pct, "%", "profit" if self.result.total_return_pct > 0 else "loss"),
            ("年化收益率", self.result.annualized_return_pct, "%", "profit" if self.result.annualized_return_pct > 0 else "loss"),
            ("夏普比率", self.result.sharpe_ratio, "", "profit" if self.result.sharpe_ratio > 1 else "warning" if self.result.sharpe_ratio > 0 else "loss"),
            ("最大回撤", -self.result.max_drawdown_pct, "%", "loss"),
            ("胜率", self.result.win_rate, "%", "profit" if self.result.win_rate > 50 else "warning"),
            ("盈利因子", self.result.profit_factor if self.result.profit_factor != float('inf') else 99, "", "profit" if self.result.profit_factor > 1 else "loss"),
            ("总交易次数", self.result.total_trades, "", "neutral"),
            ("盈利交易", self.result.winning_trades, "", "profit"),
            ("亏损交易", self.result.losing_trades, "", "loss"),
            ("平均盈利", self.result.avg_win_pct, "%", "profit"),
            ("平均亏损", self.result.avg_loss_pct, "%", "loss"),
            ("初始资金", self.result.initial_capital, "USDT", "neutral"),
        ]
        
        for idx, (name, value, suffix, color_key) in enumerate(key_metrics):
            row = idx // 4 + 1
            col = idx % 4 + 1
            
            if suffix == "%":
                number_format = "+.2f" if name not in ["胜率", "最大回撤"] else ".1f"
                display_value = f"{value:{number_format}}%"
            elif suffix == "USDT":
                display_value = f"${value:,.0f}"
            else:
                display_value = f"{value:.2f}" if isinstance(value, float) else str(value)
            
            fig.add_trace(
                go.Indicator(
                    mode="number",
                    value=value,
                    number=dict(
                        font=dict(size=28, color=self.COLORS[color_key]),
                        suffix=suffix if suffix != "USDT" else "",
                        prefix="$" if suffix == "USDT" else "",
                        valueformat=".2f" if isinstance(value, float) else "d"
                    ),
                    title=dict(
                        text=name,
                        font=dict(size=14, color=self.COLORS["text_muted"])
                    ),
                    domain=dict(row=row-1, column=col-1)
                ),
                row=row, col=col
            )
        
        self._apply_dark_theme(fig)
        fig.update_layout(
            title=dict(
                text=f"🎯 {self.result.strategy_name} - 绩效指标",
                font=dict(size=20)
            ),
            height=450,
            grid=dict(rows=3, columns=4, pattern="independent")
        )
        
        return fig
    
    def generate_monthly_returns(self) -> go.Figure:
        """Generate monthly returns heatmap"""
        if self.result.equity_curve.empty:
            return self._empty_chart("No data for monthly returns")
        
        df = self.result.equity_curve.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["year"] = df["timestamp"].dt.year
        df["month"] = df["timestamp"].dt.month
        
        # Calculate monthly returns
        monthly = df.groupby(["year", "month"]).agg({
            "equity": ["first", "last"]
        })
        monthly.columns = ["start", "end"]
        monthly["return"] = (monthly["end"] - monthly["start"]) / monthly["start"] * 100
        monthly = monthly.reset_index()
        
        # Pivot for heatmap
        pivot = monthly.pivot(index="year", columns="month", values="return")
        
        # Month names
        month_names = ["一月", "二月", "三月", "四月", "五月", "六月", 
                       "七月", "八月", "九月", "十月", "十一月", "十二月"]
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=month_names[:pivot.shape[1]],
            y=pivot.index.astype(str),
            colorscale=[
                [0, self.COLORS["loss"]],
                [0.5, self.COLORS["paper"]],
                [1, self.COLORS["profit"]]
            ],
            zmid=0,
            text=[[f"{v:.1f}%" if not pd.isna(v) else "" for v in row] for row in pivot.values],
            texttemplate="%{text}",
            textfont={"size": 12, "color": self.COLORS["text"]},
            hoverongaps=False,
            hovertemplate="年份: %{y}<br>月份: %{x}<br>收益: %{z:.2f}%<extra></extra>"
        ))
        
        self._apply_dark_theme(fig)
        fig.update_layout(
            title=dict(
                text=f"📅 {self.result.strategy_name} - 月度收益",
                font=dict(size=20)
            ),
            height=300,
            xaxis_title="月份",
            yaxis_title="年份"
        )
        
        return fig
    
    def _apply_dark_theme(self, fig: go.Figure):
        """Apply dark theme to figure"""
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=self.COLORS["paper"],
            plot_bgcolor=self.COLORS["background"],
            font=dict(color=self.COLORS["text"], family="Inter, -apple-system, sans-serif"),
            title_font=dict(color=self.COLORS["text"]),
            margin=dict(l=60, r=40, t=80, b=60)
        )
        
        fig.update_xaxes(
            gridcolor=self.COLORS["grid"],
            linecolor=self.COLORS["grid"],
            zerolinecolor=self.COLORS["grid"]
        )
        fig.update_yaxes(
            gridcolor=self.COLORS["grid"],
            linecolor=self.COLORS["grid"],
            zerolinecolor=self.COLORS["grid"]
        )
    
    def _empty_chart(self, message: str) -> go.Figure:
        """Create an empty chart with message"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20, color=self.COLORS["text_muted"])
        )
        self._apply_dark_theme(fig)
        return fig
    
    def generate_full_report(self, output_dir: str = ".") -> dict:
        """
        Generate complete report with all charts.
        
        Args:
            output_dir: Directory to save HTML files
            
        Returns:
            Dictionary with file paths and summary
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate all charts
        charts = {
            "equity_curve": self.generate_equity_curve(),
            "trade_analysis": self.generate_trade_analysis(),
            "metrics": self.generate_metrics_card(),
            "monthly_returns": self.generate_monthly_returns()
        }
        
        # Save individual charts
        file_paths = {}
        for name, fig in charts.items():
            path = os.path.join(output_dir, f"{name}.html")
            fig.write_html(path, include_plotlyjs="cdn")
            file_paths[name] = path
        
        # Generate combined HTML report
        combined_html = self._generate_combined_html(charts)
        combined_path = os.path.join(output_dir, "full_report.html")
        with open(combined_path, "w", encoding="utf-8") as f:
            f.write(combined_html)
        file_paths["full_report"] = combined_path
        
        return {
            "files": file_paths,
            "summary": self.result.to_summary_dict()
        }
    
    def _generate_combined_html(self, charts: dict) -> str:
        """Generate a combined HTML report with all charts"""
        
        # Convert figures to HTML divs
        chart_divs = []
        for name, fig in charts.items():
            div = fig.to_html(full_html=False, include_plotlyjs=False)
            chart_divs.append(div)
        
        html = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.result.strategy_name} - 回测报告</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
            color: #c9d1d9;
            min-height: 100vh;
            padding: 2rem;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        header {{
            text-align: center;
            margin-bottom: 2rem;
            padding: 2rem;
            background: rgba(22, 27, 34, 0.8);
            border-radius: 16px;
            border: 1px solid #30363d;
        }}
        
        h1 {{
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, #58a6ff, #a371f7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }}
        
        .subtitle {{
            color: #8b949e;
            font-size: 1.1rem;
        }}
        
        .chart-container {{
            background: rgba(22, 27, 34, 0.8);
            border-radius: 16px;
            border: 1px solid #30363d;
            margin-bottom: 2rem;
            padding: 1rem;
            overflow: hidden;
        }}
        
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}
        
        .summary-card {{
            background: rgba(22, 27, 34, 0.8);
            border-radius: 12px;
            border: 1px solid #30363d;
            padding: 1.5rem;
            text-align: center;
        }}
        
        .summary-card .label {{
            color: #8b949e;
            font-size: 0.9rem;
            margin-bottom: 0.5rem;
        }}
        
        .summary-card .value {{
            font-size: 1.5rem;
            font-weight: 600;
        }}
        
        .positive {{ color: #3fb950; }}
        .negative {{ color: #f85149; }}
        .neutral {{ color: #58a6ff; }}
        
        footer {{
            text-align: center;
            padding: 2rem;
            color: #8b949e;
            font-size: 0.9rem;
        }}
        
        @media (max-width: 768px) {{
            body {{
                padding: 1rem;
            }}
            h1 {{
                font-size: 1.8rem;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📊 {self.result.strategy_name}</h1>
            <p class="subtitle">
                {self.result.symbol} | {self.result.timeframe} | 
                {self.result.start_date.strftime('%Y-%m-%d')} 至 {self.result.end_date.strftime('%Y-%m-%d')}
            </p>
        </header>
        
        <div class="summary-grid">
            <div class="summary-card">
                <div class="label">总收益率</div>
                <div class="value {'positive' if self.result.total_return_pct > 0 else 'negative'}">
                    {self.result.total_return_pct:+.2f}%
                </div>
            </div>
            <div class="summary-card">
                <div class="label">夏普比率</div>
                <div class="value {'positive' if self.result.sharpe_ratio > 1 else 'neutral'}">
                    {self.result.sharpe_ratio:.2f}
                </div>
            </div>
            <div class="summary-card">
                <div class="label">最大回撤</div>
                <div class="value negative">
                    {self.result.max_drawdown_pct:.2f}%
                </div>
            </div>
            <div class="summary-card">
                <div class="label">胜率</div>
                <div class="value {'positive' if self.result.win_rate > 50 else 'neutral'}">
                    {self.result.win_rate:.1f}%
                </div>
            </div>
            <div class="summary-card">
                <div class="label">总交易次数</div>
                <div class="value neutral">
                    {self.result.total_trades}
                </div>
            </div>
            <div class="summary-card">
                <div class="label">盈利因子</div>
                <div class="value {'positive' if self.result.profit_factor > 1 else 'negative'}">
                    {self.result.profit_factor:.2f}
                </div>
            </div>
        </div>
        
        {''.join(f'<div class="chart-container">{div}</div>' for div in chart_divs)}
        
        <footer>
            <p>Generated by Crypto Strategy Backtest Skill | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </footer>
    </div>
</body>
</html>
"""
        return html


def generate_markdown_report(result: BacktestResult) -> str:
    """
    Generate a markdown summary report.
    
    Args:
        result: BacktestResult from backtest
        
    Returns:
        Markdown formatted string
    """
    summary = result.to_summary_dict()
    
    md = f"""# 📊 {result.strategy_name} - 回测报告

## 基本信息

| 项目 | 值 |
|------|-----|
| 交易对 | {result.symbol} |
| 时间周期 | {result.timeframe} |
| 回测期间 | {result.start_date.strftime('%Y-%m-%d')} 至 {result.end_date.strftime('%Y-%m-%d')} |
| 初始资金 | ${result.initial_capital:,.2f} |
| 最终资金 | ${result.final_capital:,.2f} |

## 绩效指标

### 收益

| 指标 | 值 |
|------|-----|
| 总收益率 | {result.total_return_pct:+.2f}% |
| 年化收益率 | {result.annualized_return_pct:+.2f}% |

### 风险

| 指标 | 值 |
|------|-----|
| 夏普比率 | {result.sharpe_ratio:.2f} |
| 索提诺比率 | {result.sortino_ratio:.2f} |
| 最大回撤 | {result.max_drawdown_pct:.2f}% |

### 交易统计

| 指标 | 值 |
|------|-----|
| 总交易次数 | {result.total_trades} |
| 盈利交易 | {result.winning_trades} |
| 亏损交易 | {result.losing_trades} |
| 胜率 | {result.win_rate:.1f}% |
| 盈利因子 | {result.profit_factor:.2f} |
| 平均盈利 | {result.avg_win_pct:+.2f}% |
| 平均亏损 | {result.avg_loss_pct:+.2f}% |

## 交易记录 (最近 10 笔)

| 入场时间 | 入场价格 | 出场时间 | 出场价格 | 盈亏 | 原因 |
|----------|----------|----------|----------|------|------|
"""
    
    for trade in result.trades[-10:]:
        if not trade.is_open:
            md += f"| {trade.entry_time.strftime('%Y-%m-%d %H:%M')} | ${trade.entry_price:,.2f} | {trade.exit_time.strftime('%Y-%m-%d %H:%M')} | ${trade.exit_price:,.2f} | {trade.pnl_percent:+.2f}% | {trade.exit_reason} |\n"
    
    md += f"""
---
*Generated by Crypto Strategy Backtest Skill | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    return md
