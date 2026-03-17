import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")


# ---------------------- 1. 数据获取 ----------------------
def get_foreign_future_data(symbol: str, start_date: str = "20230101"):
    """
    获取外盘期货数据
    symbol: 品种代码 (如: '伦敦金'='XAU', '布伦特原油'='BRENT')
    """
    if symbol == "XAU":
        df = ak.futures_foreign_hist(symbol="XAU")
    elif symbol == "BRENT":
        df = ak.futures_foreign_hist(symbol="BRENT")
    else:
        raise ValueError("暂不支持该品种")

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').set_index('date')
    return df


def get_contract_info(symbol: str):
    """获取合约基础信息（示例）"""
    info = {
        "XAU": {
            "name": "伦敦金",
            "multiplier": 100,  # 合约乘数（盎司）
            "exchange": "伦敦金属交易所"
        },
        "BRENT": {
            "name": "布伦特原油",
            "multiplier": 1000,  # 合约乘数（桶）
            "exchange": "洲际交易所"
        }
    }
    return info.get(symbol, {})


# ---------------------- 2. 数据可视化 ----------------------
def plot_price_trend(df: pd.DataFrame, title: str):
    """绘制价格走势"""
    plt.figure(figsize=(12, 6))
    plt.plot(df['close'], label='收盘价', color='#1f77b4')
    plt.title(f"{title} 价格走势", fontsize=15)
    plt.xlabel("日期")
    plt.ylabel("价格 (USD)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_term_structure(df: pd.DataFrame, title: str):
    """绘制期限结构（示例：不同月份合约价格）"""
    plt.figure(figsize=(10, 5))
    plt.bar(['近月', '1月后', '3月后', '6月后'],
            [df['close'].iloc[-1], df['close'].iloc[-22], df['close'].iloc[-66], df['close'].iloc[-132]],
            color='#ff7f0e')
    plt.title(f"{title} 期限结构")
    plt.ylabel("价格 (USD)")
    plt.tight_layout()
    plt.show()


# ---------------------- 3. 简单预测（ARIMA） ----------------------
def simple_arima_forecast(df: pd.DataFrame, steps: int = 7):
    """ARIMA 短期预测"""
    model = ARIMA(df['close'], order=(1, 1, 1))
    result = model.fit()
    forecast = result.get_forecast(steps=steps)
    pred_mean = forecast.predicted_mean
    pred_ci = forecast.conf_int()

    plt.figure(figsize=(12, 6))
    plt.plot(df['close'], label='历史价格')
    pred_index = pd.date_range(df.index[-1], periods=steps + 1, freq='D')[1:]
    plt.plot(pred_index, pred_mean, label='预测价格', color='red')
    plt.fill_between(pred_index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='pink', alpha=0.3)
    plt.title("价格预测（未来7天）")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    return pred_mean


# ---------------------- 4. 主程序运行 ----------------------
if __name__ == "__main__":
    # 1. 获取数据
    symbol = "XAU"  # 切换为 "BRENT" 看原油
    df = get_foreign_future_data(symbol)
    contract_info = get_contract_info(symbol)

    # 2. 展示合约信息
    print("=" * 50)
    print(f"合约信息：{contract_info['name']}")
    print(f"交易所：{contract_info['exchange']}")
    print(f"合约乘数：{contract_info['multiplier']}")
    print(f"最新价：{df['close'].iloc[-1]:.2f} USD")
    print(f"今日增减仓：{df['volume'].iloc[-1] - df['volume'].iloc[-2]:.0f} 手")
    print("=" * 50)

    # 3. 可视化
    plot_price_trend(df, contract_info['name'])
    plot_term_structure(df, contract_info['name'])

    # 4. 预测
    forecast = simple_arima_forecast(df, steps=7)
    print("未来7天预测价格：")
    print(forecast)