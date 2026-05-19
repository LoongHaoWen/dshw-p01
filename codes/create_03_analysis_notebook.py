import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import statsmodels.api as sm


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output"
NB_PATH = PROJECT_DIR / "03_analysis.ipynb"


def md(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "id": uuid4().hex[:8],
        "source": text.splitlines(keepends=True),
    }


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "id": uuid4().hex[:8],
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def load_analysis_data():
    df = pd.read_csv(DATA_DIR / "combined" / "combined_data.csv", dtype={"code": str})
    df["code"] = df["code"].str.zfill(6)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["code", "date"])
    df["log_return"] = df.groupby("code")["close"].transform(lambda x: np.log(x / x.shift(1)))

    index_daily = df[["date", "hs300_close"]].drop_duplicates().sort_values("date")
    index_daily["hs300_log_return"] = np.log(index_daily["hs300_close"] / index_daily["hs300_close"].shift(1))
    return df, index_daily


def max_drawdown(close):
    close = close.dropna()
    if close.empty:
        return np.nan
    drawdown = close / close.cummax() - 1
    return drawdown.min()


def build_interpretations():
    df, index_daily = load_analysis_data()

    stock_meta = df[["code", "name", "industry"]].drop_duplicates("code")
    date_start = df["date"].min().strftime("%Y-%m-%d")
    date_end = df["date"].max().strftime("%Y-%m-%d")
    industry_text = "、".join(stock_meta.sort_values("industry")["industry"].drop_duplicates().tolist())
    stats = []
    for code_value, g in df.groupby("code"):
        ret = g["log_return"].dropna()
        stats.append(
            {
                "code": code_value,
                "name": g["name"].iloc[0],
                "industry": g["industry"].iloc[0],
                "annual_mean": ret.mean() * 252,
                "annual_volatility": ret.std(ddof=1) * np.sqrt(252),
                "skewness": ret.skew(),
                "kurtosis": ret.kurt(),
                "max_drawdown": max_drawdown(g["close"]),
            }
        )
    stats_df = pd.DataFrame(stats)

    top_stock = stats_df.sort_values("annual_mean", ascending=False).iloc[0]
    risk_stock = stats_df.sort_values("annual_volatility", ascending=False).iloc[0]
    dd_stock = stats_df.sort_values("max_drawdown").iloc[0]

    close_wide = df.pivot_table(index="date", columns="code", values="close", aggfunc="last").sort_index()
    norm = close_wide.divide(close_wide.iloc[0])
    hs300_norm = index_daily.set_index("date")["hs300_close"].divide(index_daily["hs300_close"].iloc[0])
    period_return = norm.iloc[-1] - 1
    best_norm_code = period_return.sort_values(ascending=False).index[0]
    worst_norm_code = period_return.sort_values().index[0]
    best_norm_meta = stock_meta.set_index("code").loc[best_norm_code]
    worst_norm_meta = stock_meta.set_index("code").loc[worst_norm_code]
    hs300_period_return = hs300_norm.iloc[-1] - 1

    ret_wide = df.pivot_table(index="date", columns="code", values="log_return", aggfunc="last").sort_index()
    dist = ret_wide.agg(["mean", "std", "skew"]).T
    highest_std = dist.sort_values("std", ascending=False).index[0]
    most_skewed = dist.reindex(dist["skew"].abs().sort_values(ascending=False).index).index[0]

    industry_order = stock_meta.sort_values(["industry", "code"])["code"].tolist()
    corr = ret_wide[industry_order].corr()
    pairs = []
    meta = stock_meta.set_index("code")
    for i, c1 in enumerate(industry_order):
        for c2 in industry_order[i + 1 :]:
            same = meta.loc[c1, "industry"] == meta.loc[c2, "industry"]
            pairs.append({"same_industry": same, "corr": corr.loc[c1, c2]})
    pair_df = pd.DataFrame(pairs).dropna()
    same_corr = pair_df.loc[pair_df["same_industry"], "corr"].mean()
    cross_corr = pair_df.loc[~pair_df["same_industry"], "corr"].mean()

    monthly = df[["date", "hs300_close", "m2_yoy"]].drop_duplicates("date").dropna(subset=["hs300_close"])
    monthly = monthly.set_index("date").resample("ME").last()
    monthly["hs300_monthly_return"] = np.log(monthly["hs300_close"] / monthly["hs300_close"].shift(1))
    macro_plot = monthly[["hs300_monthly_return", "m2_yoy"]].dropna()
    macro_corr = macro_plot["m2_yoy"].corr(macro_plot["hs300_monthly_return"])

    finance = pd.read_csv(DATA_DIR / "finance" / "finance_ratios.csv", dtype={"code": str})
    finance["code"] = finance["code"].str.zfill(6)
    roe = finance[finance["indicator"].eq("ROE")].merge(stock_meta, on="code", how="left")
    roe_latest = roe.sort_values("year").groupby("code").tail(1)
    roe_leader = roe_latest.sort_values("value", ascending=False).iloc[0]
    roe_industry = roe.groupby("industry")["value"].mean().sort_values(ascending=False)

    rf_daily = 0.02 / 252
    capm_source = df.merge(index_daily[["date", "hs300_log_return"]], on="date", how="left")
    capm_rows = []
    for code_value, g in capm_source.groupby("code"):
        reg = g[["log_return", "hs300_log_return"]].dropna().copy()
        reg["stock_excess"] = reg["log_return"] - rf_daily
        reg["market_excess"] = reg["hs300_log_return"] - rf_daily
        model = sm.OLS(reg["stock_excess"], sm.add_constant(reg["market_excess"])).fit()
        beta_ci = model.conf_int().loc["market_excess"]
        capm_rows.append(
            {
                "code": code_value,
                "name": g["name"].iloc[0],
                "industry": g["industry"].iloc[0],
                "alpha": model.params["const"],
                "alpha_p": model.pvalues["const"],
                "beta": model.params["market_excess"],
                "beta_p": model.pvalues["market_excess"],
                "beta_ci_low": beta_ci[0],
                "beta_ci_high": beta_ci[1],
                "r2": model.rsquared,
                "n_obs": int(model.nobs),
            }
        )
    capm = pd.DataFrame(capm_rows)
    beta_gt1 = capm[capm["beta"] > 1].sort_values("beta", ascending=False)
    beta_gt1_text = "、".join(
        f"{r['name']}（{r['industry']}，β={r['beta']:.2f}）" for _, r in beta_gt1.iterrows()
    )
    if not beta_gt1_text:
        beta_gt1_text = "没有股票的 beta 大于 1"
    alpha_sig = capm[capm["alpha_p"] < 0.05].sort_values("alpha_p")
    if alpha_sig.empty:
        alpha_sig_text = "没有股票的 alpha 在 5% 水平上显著异于 0"
    else:
        alpha_sig_text = "、".join(
            f"{r['name']}（alpha={r['alpha']:.5f}, p={r['alpha_p']:.3f}）" for _, r in alpha_sig.iterrows()
        )
    r2_high = capm.sort_values("r2", ascending=False).iloc[0]
    r2_low = capm.sort_values("r2", ascending=True).iloc[0]

    monthly_stock = df[["date", "code", "name", "industry", "close", "m2_yoy"]].copy()
    monthly_stock["month"] = monthly_stock["date"].dt.to_period("M").astype(str)
    monthly_stock = (
        monthly_stock.sort_values("date")
        .groupby(["code", "month"], as_index=False)
        .agg({"date": "last", "name": "last", "industry": "last", "close": "last", "m2_yoy": "last"})
        .sort_values(["code", "date"])
    )
    monthly_stock["monthly_return"] = monthly_stock.groupby("code")["close"].transform(
        lambda x: np.log(x / x.shift(1))
    )
    macro_rows = []
    for code_value, g in monthly_stock.groupby("code"):
        reg = g[["monthly_return", "m2_yoy"]].dropna()
        model = sm.OLS(reg["monthly_return"], sm.add_constant(reg["m2_yoy"])).fit()
        gamma_ci = model.conf_int().loc["m2_yoy"]
        macro_rows.append(
            {
                "code": code_value,
                "name": g["name"].iloc[0],
                "industry": g["industry"].iloc[0],
                "gamma": model.params["m2_yoy"],
                "gamma_p": model.pvalues["m2_yoy"],
                "gamma_ci_low": gamma_ci[0],
                "gamma_ci_high": gamma_ci[1],
                "r2": model.rsquared,
                "n_obs": int(model.nobs),
            }
        )
    macro_reg = pd.DataFrame(macro_rows)
    most_positive_gamma = macro_reg.sort_values("gamma", ascending=False).iloc[0]
    most_negative_gamma = macro_reg.sort_values("gamma", ascending=True).iloc[0]
    gamma_sig = macro_reg[macro_reg["gamma_p"] < 0.05].sort_values("gamma_p")
    if gamma_sig.empty:
        gamma_sig_text = "没有股票的 M2 系数在 5% 水平上显著"
    else:
        gamma_sig_text = "、".join(
            f"{r['name']}（γ={r['gamma']:.4f}, p={r['gamma_p']:.3f}）" for _, r in gamma_sig.iterrows()
        )

    return {
        "stats_text": (
            f"从描述性统计看，{top_stock['name']}（{top_stock['code']}，{top_stock['industry']}）的年化均值最高，"
            f"约为 {top_stock['annual_mean']:.2%}；{risk_stock['name']} 的年化波动率最高，约为 {risk_stock['annual_volatility']:.2%}。"
            f"最大回撤最深的是 {dd_stock['name']}，约为 {dd_stock['max_drawdown']:.2%}，说明即便长期收益较好的股票也可能经历较大的阶段性下跌。"
        ),
        "data_text": (
            f"本报告使用 10 只 A 股股票的后复权日度行情，样本区间为 {date_start} 至 {date_end}，覆盖行业包括 {industry_text}。"
            "市场基准使用沪深 300，辅助市场指数使用上证综指；宏观指标包括 CPI 同比、人民币/美元汇率、M2 同比、1 年期 LPR 和工业增加值增速，财务指标包括 ROE、净利润率、资产负债率和营业收入增速。"
        ),
        "fig1_text": (
            f"归一化走势显示，样本期内表现最好的是 {best_norm_meta['name']}（{best_norm_code}，{best_norm_meta['industry']}），"
            f"累计涨幅约为 {period_return.loc[best_norm_code]:.2%}；表现最弱的是 {worst_norm_meta['name']}，累计变化约为 {period_return.loc[worst_norm_code]:.2%}。"
            f"沪深 300 同期累计变化约为 {hs300_period_return:.2%}，个股之间的分化明显大于市场基准，说明行业景气度和公司基本面对收益路径影响很大。"
        ),
        "fig2_text": (
            f"日收益率分布整体集中在 0 附近，但多个股票存在尖峰厚尾特征，说明极端涨跌的概率高于正态分布假设。"
            f"其中 {meta.loc[highest_std, 'name']} 的日收益率标准差最高，{meta.loc[most_skewed, 'name']} 的偏度绝对值较大，提示后续回归或风险分析不宜只依赖均值和方差。"
        ),
        "fig3_text": (
            f"相关系数热力图按行业排序后，可以比较同行业与跨行业股票的联动程度。"
            f"本样本中同行业股票平均相关系数约为 {same_corr:.2f}，跨行业平均相关系数约为 {cross_corr:.2f}；如果同行业均值更高，通常反映共同基本面、政策和景气周期带来的同步波动。"
        ),
        "fig4_text": (
            f"本图选择 M2 同比增速作为宏观指标，Pearson 相关系数约为 {macro_corr:.2f}。"
            f"若拟合线向上，表示流动性扩张通常与权益市场收益改善相关；若斜率较弱或相关性接近 0，则说明月度市场收益还受到盈利预期、风险偏好和外部冲击等因素共同影响。"
        ),
        "fig5_text": (
            f"ROE 对比显示，最近 5 年不同公司的盈利能力差异明显，最新年度 ROE 最高的是 {roe_leader['name']}（{roe_leader['industry']}），约为 {roe_leader['value']:.2f}%。"
            f"按行业均值看，{roe_industry.index[0]} 行业 ROE 水平相对较高；ROE 趋势的持续上行通常意味着资本使用效率改善，而持续下行可能反映周期压力或盈利能力走弱。"
        ),
        "capm_text": (
            f"CAPM 结果中 beta 大于 1 的股票包括：{beta_gt1_text}。"
            f"这些股票对市场波动更敏感，更接近周期性或高弹性资产特征；beta 小于 1 的股票相对防御，但行业属性并不是唯一解释，公司自身经营周期和估值波动也会影响 beta。"
        ),
        "capm_alpha_text": (
            f"Alpha 显著性检验结果显示，{alpha_sig_text}。"
            f"显著 alpha 通常意味着样本期内该股票存在 CAPM 无法解释的超额收益或亏损，但也可能来自遗漏风险因子、行业冲击或样本期特殊事件。"
        ),
        "capm_r2_text": (
            f"R² 最高的是 {r2_high['name']}（R²={r2_high['r2']:.2f}），最低的是 {r2_low['name']}（R²={r2_low['r2']:.2f}）。"
            f"R² 高说明该股票日收益中较大比例可由沪深 300 解释；R² 低则说明个股特质、行业事件或非市场因子占比更高。"
        ),
        "macro_reg_text": (
            f"M2 月度回归中，γ 最大的是 {most_positive_gamma['name']}（{most_positive_gamma['industry']}，γ={most_positive_gamma['gamma']:.4f}），"
            f"γ 最小的是 {most_negative_gamma['name']}（{most_negative_gamma['industry']}，γ={most_negative_gamma['gamma']:.4f}）。"
            f"显著性方面，{gamma_sig_text}；如果系数不显著，说明在本样本中 M2 同比增速对单个股票月收益的解释力有限。"
        ),
        "macro_reg_econ_text": (
            "从经济逻辑看，M2 增速代表流动性环境，理论上更充裕的流动性可能提高风险偏好并支持估值。"
            "不同行业敏感性差异可能来自融资需求、久期属性、景气周期和政策传导速度不同，因此不能把 γ 简单解释为因果效应。"
        ),
        "conclusion_text": (
            f"综合来看，样本股票在 2020 年以来的表现分化明显：{top_stock['name']} 的年化收益表现突出，"
            f"{risk_stock['name']} 的波动更高，{dd_stock['name']} 的阶段性回撤压力最大。CAPM 结果表明，部分汽车、通讯和白酒股票 beta 高于 1，市场弹性较强；"
            "而 beta 较低或 R² 较低的股票更受个股和行业特质影响。宏观回归中 M2 对单只股票月收益的解释力整体有限，因此宏观变量更适合作为背景变量，而不是单独预测股票收益的充分依据。"
        ),
    }


interp = build_interpretations()

cells = [
    md(
        """# P01 金融数据分析报告

本报告基于 `01_download.ipynb` 和 `02_clean.ipynb` 生成的数据，完成数据说明、清洗说明、描述统计、图表分析、CAPM 回归和宏观变量回归。报告可独立阅读；即使不打开 Notebook，也能理解数据来源、处理方法和主要发现。"""
    ),
    md(
        """## 1. 数据说明

""" + interp["data_text"] + """

股票行情、指数数据、宏观指标和财务指标均保存为 CSV，清洗后的核心分析表为 `data/combined/combined_data.csv`。股票日收益率和 CAPM 均使用日对数收益率，市场基准为沪深 300；宏观分析使用月度数据以匹配宏观指标频率。"""
    ),
    md(
        """## 2. 清洗说明

清洗步骤在 `02_clean.ipynb` 中完成。每只股票的原始表先检测缺失值和重复值，再统一日期格式为 `datetime64[ns]`，并将价格、成交量和成交额字段转换为数值型。缺失的数值型字段采用向前填充，仍无法补齐的关键记录删除；重复记录按 `date + code` 删除；单日涨跌幅超过 +/-20% 的记录标注为 `is_extreme=True`，但不删除。

随后将 10 只股票的收盘价转换为宽表和长表，并将个股日度数据与沪深 300、上证综指按日期左连接。月度宏观指标通过 `YYYY-MM` 月份键映射到对应月份的每日交易数据，保证日度股票样本行数不因合并宏观数据而改变。"""
    ),
    md(
        """## 3. 分析环境和股票列表

下面代码读取清洗后的综合数据，统一股票代码格式，计算日对数收益率，并展示本报告使用的股票列表。所有图形输出到 `output/` 文件夹，PNG 分辨率不低于 150 dpi。"""
    ),
    code(
        """from pathlib import Path
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", font="Microsoft YaHei")

df = pd.read_csv(DATA_DIR / "combined" / "combined_data.csv", dtype={"code": str})
df["code"] = df["code"].str.zfill(6)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["code", "date"])
df["log_return"] = df.groupby("code")["close"].transform(lambda x: np.log(x / x.shift(1)))

stock_meta = df[["code", "name", "industry"]].drop_duplicates("code").sort_values(["industry", "code"])
stock_order = stock_meta["code"].tolist()
code_label = dict(zip(stock_meta["code"], stock_meta["code"] + " " + stock_meta["name"]))
industry_map = dict(zip(stock_meta["code"], stock_meta["industry"]))

index_daily = df[["date", "hs300_close"]].drop_duplicates().sort_values("date")
index_daily["hs300_log_return"] = np.log(index_daily["hs300_close"] / index_daily["hs300_close"].shift(1))

display(stock_meta)
"""
    ),
    md(
        """## 4.1 基本统计量

日收益率使用公式 $r_t=\\ln(P_t/P_{t-1})$ 计算。年化均值按日均收益率乘以 252，年化波动率按日收益率标准差乘以 $\\sqrt{252}$，最大回撤基于收盘价相对历史高点的跌幅计算。"""
    ),
    code(
        """def max_drawdown(close):
    close = close.dropna()
    drawdown = close / close.cummax() - 1
    return drawdown.min()

stats_rows = []
for code_value, g in df.groupby("code"):
    ret = g["log_return"].dropna()
    stats_rows.append({
        "股票": f"{code_value} {g['name'].iloc[0]}",
        "行业": g["industry"].iloc[0],
        "年化均值": ret.mean() * 252,
        "年化波动率": ret.std(ddof=1) * np.sqrt(252),
        "偏度": ret.skew(),
        "峰度": ret.kurt(),
        "最大回撤": max_drawdown(g["close"]),
    })

stats_df = pd.DataFrame(stats_rows).sort_values(["行业", "股票"])
stats_display = stats_df.copy()
for col in ["年化均值", "年化波动率", "最大回撤"]:
    stats_display[col] = stats_display[col].map(lambda x: f"{x:.2%}")
for col in ["偏度", "峰度"]:
    stats_display[col] = stats_display[col].map(lambda x: f"{x:.2f}")

stats_df.to_csv(OUTPUT_DIR / "return_descriptive_stats.csv", index=False, encoding="utf-8-sig")
display(stats_display)
"""
    ),
    md(interp["stats_text"]),
    md(
        """## 4.2 图 1：归一化收盘价走势图

将每只股票和沪深 300 的起点统一为 1，可以比较不同价格水平资产的累计表现。股票线条按行业着色，沪深 300 使用黑色虚线作为市场基准。"""
    ),
    code(
        """close_wide = df.pivot_table(index="date", columns="code", values="close", aggfunc="last").sort_index()
norm_close = close_wide.divide(close_wide.iloc[0])
hs300_norm = index_daily.set_index("date")["hs300_close"].divide(index_daily["hs300_close"].iloc[0])

industries = stock_meta["industry"].drop_duplicates().tolist()
palette = dict(zip(industries, sns.color_palette("tab10", n_colors=len(industries))))

fig, ax = plt.subplots(figsize=(13, 7))
for code_value in stock_order:
    ax.plot(
        norm_close.index,
        norm_close[code_value],
        color=palette[industry_map[code_value]],
        linewidth=1.4,
        alpha=0.9,
        label=f"{industry_map[code_value]} | {code_label[code_value]}",
    )
ax.plot(hs300_norm.index, hs300_norm, color="black", linestyle="--", linewidth=2.2, label="市场基准 | 沪深300")
ax.set_title("图1：10只股票与沪深300归一化收盘价走势（起点=1）")
ax.set_xlabel("日期")
ax.set_ylabel("归一化收盘价")
ax.legend(ncol=2, fontsize=8, frameon=True)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "fig1_normalized_close.png", dpi=180, bbox_inches="tight")
plt.show()
"""
    ),
    md(interp["fig1_text"]),
    md(
        """## 图 2：日收益率分布图

下图为 10 只股票日对数收益率的分面直方图，并叠加同均值、同标准差的正态分布曲线。每个子图标注该股票样本期内的日均收益率和日标准差。"""
    ),
    code(
        """ret_wide = df.pivot_table(index="date", columns="code", values="log_return", aggfunc="last").sort_index()

fig, axes = plt.subplots(2, 5, figsize=(16, 7), sharex=False, sharey=False)
axes = axes.ravel()
for ax, code_value in zip(axes, stock_order):
    ret = ret_wide[code_value].dropna()
    mu = ret.mean()
    sigma = ret.std(ddof=1)
    sns.histplot(ret, bins=45, stat="density", color=palette[industry_map[code_value]], alpha=0.45, ax=ax)
    if sigma > 0:
        x = np.linspace(ret.quantile(0.005), ret.quantile(0.995), 200)
        y = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        ax.plot(x, y, color="black", linewidth=1.3)
    ax.axvline(mu, color="red", linestyle="--", linewidth=1)
    ax.set_title(code_label[code_value], fontsize=10)
    ax.text(0.03, 0.95, f"均值={mu:.4f}\\n标准差={sigma:.4f}", transform=ax.transAxes, va="top", fontsize=8)
    ax.set_xlabel("日对数收益率")
    ax.set_ylabel("密度")

fig.suptitle("图2：10只股票日收益率分布与正态曲线", y=1.02, fontsize=15)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "fig2_return_distribution.png", dpi=180, bbox_inches="tight")
plt.show()
"""
    ),
    md(interp["fig2_text"]),
    md(
        """## 图 3：收益率相关系数热力图

热力图使用 10 只股票的日对数收益率计算 Pearson 相关系数。股票按行业排序，便于观察同行业股票是否比跨行业股票更同步。"""
    ),
    code(
        """corr = ret_wide[stock_order].corr()
label_order = [code_label[c] for c in stock_order]

fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(
    corr,
    ax=ax,
    cmap="RdBu_r",
    vmin=-1,
    vmax=1,
    annot=True,
    fmt=".2f",
    square=True,
    xticklabels=label_order,
    yticklabels=label_order,
    cbar_kws={"label": "相关系数"},
)
ax.set_title("图3：10只股票日收益率相关系数矩阵（按行业排序）")
ax.tick_params(axis="x", rotation=45)
ax.tick_params(axis="y", rotation=0)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "fig3_return_correlation_heatmap.png", dpi=180, bbox_inches="tight")
plt.show()
"""
    ),
    md(interp["fig3_text"]),
    md(
        """## 图 4：宏观指标与股市关系

选择 M2 同比增速作为宏观指标。M2 代表广义货币供给，通常用于观察市场流动性环境；这里将其与沪深 300 月度对数收益率进行散点和线性拟合分析。"""
    ),
    code(
        """monthly = df[["date", "hs300_close", "m2_yoy"]].drop_duplicates("date").dropna(subset=["hs300_close"])
monthly = monthly.set_index("date").resample("ME").last()
monthly["hs300_monthly_return"] = np.log(monthly["hs300_close"] / monthly["hs300_close"].shift(1))
macro_plot = monthly[["hs300_monthly_return", "m2_yoy"]].dropna()
pearson_corr = macro_plot["m2_yoy"].corr(macro_plot["hs300_monthly_return"])

fig, ax = plt.subplots(figsize=(9, 6))
sns.regplot(
    data=macro_plot,
    x="m2_yoy",
    y="hs300_monthly_return",
    ax=ax,
    scatter_kws={"s": 45, "alpha": 0.75},
    line_kws={"color": "red", "linewidth": 2},
)
ax.axhline(0, color="gray", linewidth=1, linestyle="--")
ax.set_title("图4：M2同比增速与沪深300月度收益率")
ax.set_xlabel("M2同比增速（%）")
ax.set_ylabel("沪深300月度对数收益率")
ax.text(0.03, 0.95, f"Pearson r = {pearson_corr:.2f}", transform=ax.transAxes, va="top", fontsize=12)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "fig4_m2_hs300_scatter.png", dpi=180, bbox_inches="tight")
plt.show()
"""
    ),
    md(interp["fig4_text"]),
    md(
        """## 图 5（选做）：财务指标跨公司对比

下图使用财务长表中的 ROE 指标，展示最近 5 年各公司的 ROE 变化，并按行业着色。ROE 是净资产收益率，能反映公司使用股东权益创造利润的能力。"""
    ),
    code(
        """finance = pd.read_csv(DATA_DIR / "finance" / "finance_ratios.csv", dtype={"code": str})
finance["code"] = finance["code"].str.zfill(6)
roe = finance[finance["indicator"].eq("ROE")].merge(stock_meta, on="code", how="left")
roe["label"] = roe["code"] + " " + roe["name"]
roe = roe.sort_values(["industry", "code", "year"])

fig, ax = plt.subplots(figsize=(12, 7))
for code_value, g in roe.groupby("code"):
    ax.plot(
        g["year"],
        g["value"],
        marker="o",
        linewidth=1.8,
        color=palette.get(g["industry"].iloc[0], "gray"),
        label=f"{g['industry'].iloc[0]} | {g['label'].iloc[0]}",
    )
ax.set_title("图5：最近5年10只股票ROE跨公司对比")
ax.set_xlabel("年度")
ax.set_ylabel("ROE（%）")
ax.legend(ncol=2, fontsize=8, frameon=True)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "fig5_roe_by_company.png", dpi=180, bbox_inches="tight")
plt.show()
"""
    ),
    md(interp["fig5_text"]),
    md(
        """# 第五部分：回归分析

本部分使用 `statsmodels.OLS` 完成 CAPM 模型估计，并使用月度数据分析宏观指标对个股收益率的影响。CAPM 的市场基准为沪深 300，年化无风险利率统一设为 2.0%，日频换算为 `0.02 / 252`。"""
    ),
    md(
        """## 5.1 CAPM 模型估计

对 10 只股票分别估计 CAPM 模型：

$$r_{i,t}-r_f=\\alpha_i+\\beta_i(r_{m,t}-r_f)+\\epsilon_{i,t}$$

其中，$r_{i,t}$ 为个股日对数收益率，$r_{m,t}$ 为沪深 300 日对数收益率，$r_f$ 为日频无风险利率。"""
    ),
    code(
        """rf_daily = 0.02 / 252
capm_source = df.merge(index_daily[["date", "hs300_log_return"]], on="date", how="left")

capm_rows = []
for code_value, g in capm_source.groupby("code"):
    reg = g[["log_return", "hs300_log_return"]].dropna().copy()
    reg["stock_excess"] = reg["log_return"] - rf_daily
    reg["market_excess"] = reg["hs300_log_return"] - rf_daily
    model = sm.OLS(reg["stock_excess"], sm.add_constant(reg["market_excess"])).fit()
    beta_ci = model.conf_int().loc["market_excess"]
    capm_rows.append({
        "code": code_value,
        "name": g["name"].iloc[0],
        "industry": g["industry"].iloc[0],
        "alpha": model.params["const"],
        "alpha_p_value": model.pvalues["const"],
        "beta": model.params["market_excess"],
        "beta_p_value": model.pvalues["market_excess"],
        "beta_ci_low": beta_ci[0],
        "beta_ci_high": beta_ci[1],
        "r_squared": model.rsquared,
        "n_obs": int(model.nobs),
    })

capm_results = pd.DataFrame(capm_rows).sort_values(["industry", "code"])
capm_results.to_csv(OUTPUT_DIR / "capm_results.csv", index=False, encoding="utf-8-sig")

capm_display = capm_results.copy()
capm_display["股票"] = capm_display["code"] + " " + capm_display["name"]
capm_display["95% CI"] = capm_display.apply(
    lambda r: f"[{r['beta_ci_low']:.2f}, {r['beta_ci_high']:.2f}]", axis=1
)
capm_display = capm_display[["股票", "industry", "alpha", "alpha_p_value", "beta", "95% CI", "r_squared"]]
capm_display = capm_display.rename(columns={
    "industry": "行业",
    "alpha": "alpha_hat",
    "alpha_p_value": "p值(alpha)",
    "beta": "beta_hat",
    "r_squared": "R²",
})
display(capm_display.style.format({
    "alpha_hat": "{:.5f}",
    "p值(alpha)": "{:.3f}",
    "beta_hat": "{:.2f}",
    "R²": "{:.2f}",
}))
"""
    ),
    md(
        """### CAPM 表格说明

表中 `alpha_hat` 是控制市场超额收益后的平均异常收益，`beta_hat` 是个股相对沪深 300 的系统性风险暴露。`95% CI` 是 beta 的 95% 置信区间，若区间整体高于 1，说明该股票在样本期内显著高于市场弹性；若区间包含 1，则 beta 与市场弹性的差异未必稳健。"""
    ),
    code(
        """capm_plot = capm_results.copy()
capm_plot["label"] = capm_plot["code"] + " " + capm_plot["name"]
capm_plot = capm_plot.sort_values(["industry", "beta"])

fig, ax = plt.subplots(figsize=(10, 7))
for idx, row in capm_plot.reset_index(drop=True).iterrows():
    ax.errorbar(
        x=row["beta"],
        y=idx,
        xerr=[[row["beta"] - row["beta_ci_low"]], [row["beta_ci_high"] - row["beta"]]],
        fmt="o",
        color=palette.get(row["industry"], "gray"),
        ecolor=palette.get(row["industry"], "gray"),
        capsize=4,
        markersize=6,
    )
ax.axvline(1, color="black", linestyle="--", linewidth=1.5, label="beta=1")
ax.set_yticks(range(len(capm_plot)))
ax.set_yticklabels(capm_plot["label"])
ax.set_xlabel("Beta 估计值")
ax.set_ylabel("股票")
ax.set_title("图6：CAPM Beta 系数与 95% 置信区间")

handles = [
    plt.Line2D([0], [0], marker="o", linestyle="", color=color, label=industry)
    for industry, color in palette.items()
]
handles.append(plt.Line2D([0], [0], color="black", linestyle="--", label="beta=1"))
ax.legend(handles=handles, title="行业", bbox_to_anchor=(1.02, 1), loc="upper left")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "fig6_capm_beta.png", dpi=180, bbox_inches="tight")
plt.show()
"""
    ),
    md(
        interp["capm_text"]
        + "\n\n"
        + interp["capm_alpha_text"]
        + "\n\n"
        + interp["capm_r2_text"]
    ),
    md(
        """## 5.2 宏观指标对股票收益率的影响（选做）

选择 M2 同比增速作为宏观变量 $X_t$，以月度数据估计：

$$r^{月}_{i,t}=\\alpha_i+\\gamma_i X_t+\\epsilon_{i,t}$$

这里的月度收益率由每只股票月末收盘价计算，M2 同比增速使用 `02_clean.ipynb` 中合并后的月度宏观指标。"""
    ),
    code(
        """monthly_stock = df[["date", "code", "name", "industry", "close", "m2_yoy"]].copy()
monthly_stock["month"] = monthly_stock["date"].dt.to_period("M").astype(str)
monthly_stock = (
    monthly_stock.sort_values("date")
    .groupby(["code", "month"], as_index=False)
    .agg({"date": "last", "name": "last", "industry": "last", "close": "last", "m2_yoy": "last"})
    .sort_values(["code", "date"])
)
monthly_stock["monthly_return"] = monthly_stock.groupby("code")["close"].transform(
    lambda x: np.log(x / x.shift(1))
)

macro_rows = []
for code_value, g in monthly_stock.groupby("code"):
    reg = g[["monthly_return", "m2_yoy"]].dropna()
    model = sm.OLS(reg["monthly_return"], sm.add_constant(reg["m2_yoy"])).fit()
    gamma_ci = model.conf_int().loc["m2_yoy"]
    macro_rows.append({
        "code": code_value,
        "name": g["name"].iloc[0],
        "industry": g["industry"].iloc[0],
        "gamma": model.params["m2_yoy"],
        "gamma_p_value": model.pvalues["m2_yoy"],
        "gamma_ci_low": gamma_ci[0],
        "gamma_ci_high": gamma_ci[1],
        "r_squared": model.rsquared,
        "n_obs": int(model.nobs),
    })

macro_reg_results = pd.DataFrame(macro_rows).sort_values(["industry", "code"])
macro_reg_results.to_csv(OUTPUT_DIR / "macro_m2_regression_results.csv", index=False, encoding="utf-8-sig")

macro_display = macro_reg_results.copy()
macro_display["股票"] = macro_display["code"] + " " + macro_display["name"]
macro_display["95% CI"] = macro_display.apply(
    lambda r: f"[{r['gamma_ci_low']:.4f}, {r['gamma_ci_high']:.4f}]", axis=1
)
macro_display = macro_display[["股票", "industry", "gamma", "gamma_p_value", "95% CI", "r_squared"]]
macro_display = macro_display.rename(columns={
    "industry": "行业",
    "gamma": "gamma_hat",
    "gamma_p_value": "p值(gamma)",
    "r_squared": "R²",
})
display(macro_display.style.format({
    "gamma_hat": "{:.4f}",
    "p值(gamma)": "{:.3f}",
    "R²": "{:.2f}",
}))
"""
    ),
    code(
        """macro_plot_reg = macro_reg_results.copy()
macro_plot_reg["label"] = macro_plot_reg["code"] + " " + macro_plot_reg["name"]
macro_plot_reg = macro_plot_reg.sort_values(["industry", "gamma"])

fig, ax = plt.subplots(figsize=(10, 7))
for idx, row in macro_plot_reg.reset_index(drop=True).iterrows():
    ax.errorbar(
        x=row["gamma"],
        y=idx,
        xerr=[[row["gamma"] - row["gamma_ci_low"]], [row["gamma_ci_high"] - row["gamma"]]],
        fmt="o",
        color=palette.get(row["industry"], "gray"),
        ecolor=palette.get(row["industry"], "gray"),
        capsize=4,
        markersize=6,
    )
ax.axvline(0, color="black", linestyle="--", linewidth=1.5, label="gamma=0")
ax.set_yticks(range(len(macro_plot_reg)))
ax.set_yticklabels(macro_plot_reg["label"])
ax.set_xlabel("M2 同比增速系数 gamma")
ax.set_ylabel("股票")
ax.set_title("图7：M2 同比增速对月度股票收益率的回归系数")

handles = [
    plt.Line2D([0], [0], marker="o", linestyle="", color=color, label=industry)
    for industry, color in palette.items()
]
handles.append(plt.Line2D([0], [0], color="black", linestyle="--", label="gamma=0"))
ax.legend(handles=handles, title="行业", bbox_to_anchor=(1.02, 1), loc="upper left")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "fig7_macro_m2_gamma.png", dpi=180, bbox_inches="tight")
plt.show()
"""
    ),
    md(interp["macro_reg_text"] + "\n\n" + interp["macro_reg_econ_text"]),
    md(
        """## 6. 结论

""" + interp["conclusion_text"] + """

方法上，本项目先用 CSV 保留原始和合并数据，再用 Parquet 展示列式存储优势；分析阶段以清洗后的 CSV 为主数据源，保证结果可复现。需要注意的是，CAPM 和宏观回归都是相关性分析，不能单独证明因果关系；后续若要提高解释力，可以加入行业指数、规模因子、价值因子、动量因子和更多公司基本面变量。"""
    ),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"created {NB_PATH}")
