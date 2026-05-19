# P01：金融数据获取、管理与初步分析

本项目围绕 A 股市场构建一个可复现的数据获取、管理与初步分析流程，覆盖个股行情、市场指数、宏观指标和上市公司财务指标。后续清洗和分析以 CSV 文件为主要数据源，同时使用 Parquet 展示进阶列式存储能力。

## 股票列表

| 代码 | 名称 | 行业 | 选股理由 |
|------|------|------|---------|
| 601398 | 工商银行 | 银行 | 国有大行代表，市值和流动性高，适合作为银行板块稳健型样本。 |
| 600036 | 招商银行 | 银行 | 股份制银行龙头，零售金融和财富管理能力突出，市场关注度高。 |
| 002594 | 比亚迪 | 汽车 | 新能源汽车龙头，产业链垂直整合和全球化扩张带来较高关注度。 |
| 601127 | 赛力斯 | 汽车 | 华为汽车链代表标的，智能汽车主题弹性较强。 |
| 000002 | 万科A | 房地产 | 房地产行业代表公司，适合观察地产周期和政策变化对股票表现的影响。 |
| 600519 | 贵州茅台 | 白酒 | 高端白酒龙头，盈利能力和品牌壁垒突出，是消费蓝筹代表。 |
| 000858 | 五粮液 | 白酒 | 浓香型白酒龙头，和贵州茅台共同代表白酒板块核心资产。 |
| 601088 | 中国神华 | 能源 | 煤炭能源龙头，高股息和周期属性明显，适合作为能源板块样本。 |
| 300308 | 中际旭创 | 通讯 | 光模块和 AI 算力链代表公司，成长属性和主题热度较高。 |
| 002352 | 顺丰控股 | 物流 | 快递物流龙头，能反映消费、供应链和综合物流景气度。 |

## 数据来源

- 股票行情：AkShare，A 股个股日度行情，后复权，保存于 `data/stock/`。
- 市场指数：沪深 300（`000300`）作为 CAPM 市场基准；上证综指（`000001`）作为 A 股整体市场走势参考，保存于 `data/index/`。
- 宏观指标：
  - CPI 同比增速，来源：AkShare `ak.macro_china_cpi_yearly()`，选择理由：反映通胀压力，影响货币政策预期、实际利率和股票估值。
  - 人民币/美元汇率，来源：Frankfurter API、GMD、World Bank API，选择理由：影响外资流动、出口企业收入折算、进口成本和市场风险偏好。
  - M2 同比增速，来源：AkShare `ak.macro_china_money_supply()`、World Bank API，选择理由：反映流动性环境和信用扩张，对市场估值和风险偏好有影响。
  - 1 年期 LPR 利率，来源：AkShare `ak.macro_china_lpr()`，选择理由：代表贷款市场报价利率，影响企业融资成本和权益估值折现率。
  - 工业增加值增速，来源：AkShare `ak.macro_china_gyzjz()`、World Bank API，选择理由：反映实体经济生产景气度，与企业盈利周期相关。
- 财务数据：AkShare 同花顺财务摘要接口，提取最近 5 个年度的 ROE、净利润率、资产负债率、营业收入增速，保存于 `data/finance/finance_ratios.csv`。

## 存储方式

- 基础：CSV（方式 A）
  - 所有原始数据均以 CSV 格式存储在 `data/stock/`、`data/index/`、`data/macro/`、`data/finance/`。
  - 合并后的综合数据保存为 `data/combined/combined_data.csv`。
- 进阶：Parquet（方式 B）
  - 清洗后的股票数据同时保存为 `data/clean/stock_clean.csv` 和 `data/clean/stock_clean.parquet`。
- 选择进阶方式的理由：Parquet 是列式存储格式，支持只读取需要的列，并保留 Schema 类型信息。相比 CSV，Parquet 通常文件体积更小、读取速度更快，尤其适合列数较多、数据量较大、需要反复读取部分列的分析任务。

CSV 的优点是简单、透明、跨平台，几乎所有数据分析工具都能直接读取，适合课程作业、数据交换、版本管理和人工检查。CSV 的不足是缺少严格类型约束和索引，文件变大后读写效率较低，多表关系也需要依赖代码维护。

在当前约 1.5 万行的股票清洗数据上，Parquet 文件体积约为 CSV 的一半以内，读取速度也更快；但两者都能在很短时间内完成读取，因此差异不会成为当前分析流程的瓶颈。当数据达到百万行以上、列数很多、需要频繁读取部分列，或需要严格保留字段类型时，Parquet 的优势会更显著。

## 下载日志

所有下载函数会将结果追加记录到 `download_log.txt`，格式为：

```text
[YYYY-MM-DD HH:MM:SS] SUCCESS  dataset_id  shape=(rows, cols)
[YYYY-MM-DD HH:MM:SS] FAILED   dataset_id  Error: message
```

## GitHub 仓库

- GitHub 仓库：https://github.com/LoongHaoWen/dshw-p01
- GitHub Pages：https://loonghaowen.github.io/dshw-p01/
- Quarto Online Book：仓库包含 `_quarto.yml`、`*.qmd` 章节和 `.github/workflows/quarto-pages.yml`，推送到 `main` 后由 GitHub Actions 自动渲染并部署到 GitHub Pages。

## 如何运行

1. 安装依赖：`pip install -r requirements.txt`
2. 运行 `01_download.ipynb` 下载原始数据
3. 运行 `02_clean.ipynb` 清洗并存储数据
4. 运行 `03_analysis.ipynb` 查看分析结果
5. 打开 `report.html` 阅读完整报告
6. 如需本地渲染 Quarto Book：安装 Quarto 后运行 `quarto render`
