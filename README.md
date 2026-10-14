# 光伏功率预测 pv_forecast

基于历史光伏功率、历史气象和未来日历/太阳位置，预测未来 **24 小时、96 个 15 分钟功率点**。提供基线模型、双输入 GRU、经验预测区间，以及可选的点预测修正。

当前模型处于离线模拟验证阶段。仓库中的评估结果来自模拟数据，尚未完成真实场站验收，也未连接设备控制。

## 功能

- 使用过去3天、288个历史点，支持功率及历史辐照、温度、云量等特征。
- 提供持久性、前一日同刻、近几日同刻均值三种基线，以及双输入GRU。
- 默认完整执行5/7/9月三段滚动验证，每段比较三组GRU；支持学习率下降、早停、最佳权重保存和逐轮日志。
- 数据按时间切分，目标窗口不跨分区，缩放器仅使用训练数据；历史回放先截断未来行再校验和填充。
- 支持昼夜分组经验区间、夜间零下界，以及冻结权重再校准。
- 支持可选的太阳位置/历史气象Ridge点修正，保留原GRU权重，并在滚动验证中选择修正参数。
- 支持保存重载、明确的故障基线回退、全窗口冻结回测、JSON预测记录及功率参考图。
- 支持按发布时间归档的未来气象预报接口，要求发布时间不晚于预测起点，且同一批次覆盖全部96点。

## 目录

```text
pv_forecast/
├─ README.md
├─ requirements-verified.txt          # 历史验证环境中的依赖版本
├─ docs/
│  ├─ 评估结果.json                   # 模拟评估摘要
│  └─ 验证记录.json                   # 历史53项测试及发布检查
└─ 代码/
   ├─ 光伏/
   │  ├─ 运行.py                     # 默认完整GRU训练入口
   │  ├─ requirements.txt
   │  ├─ pv_forecast/                # 算法、CLI、记录Schema
   │  └─ tests/
   └─ 数据/光伏数据/
      ├─ generate_huanghua_pv.py     # 生成模拟数据
      └─ simulation_config.json
```

运行生成的数据、模型权重、虚拟环境和本地输出不纳入代码提交。完整历史模型与训练证据保留在开发工作区。

## 安装

历史验证使用Python 3.13.5、TensorFlow CPU 2.21.0、Keras 3.14.1。建议使用Python 3.13复现；TensorFlow仅在GRU训练及推理时必需。

以下命令使用PowerShell，在仓库根目录执行：

```powershell
git clone https://github.com/gj2696928705-dotcom/pv_forecast.git
cd pv_forecast
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-verified.txt
```

只使用不含GRU的功能时，可安装 `代码/光伏/requirements.txt`。读取Excel还需要与文件格式对应的读取引擎，例如 `.xlsx` 使用 `openpyxl`。

## 生成模拟数据

```powershell
python .\代码\数据\光伏数据\generate_huanghua_pv.py
```

生成黄花站2025年模拟CSV、月度统计和校验报告：全年35,040行、15分钟粒度、Asia/Shanghai。模拟容量为5,000 kW，太阳位置使用近似坐标30.90°N、111.35°E；这些参数未经真实站点核验。

生成器与模型兼容性检查将同步天气用于历史特征。同步观测不属于提前发布的未来预报。

## 完整训练

在仓库根目录执行：

```powershell
python .\代码\光伏\运行.py
```

默认读取上述模拟CSV，执行三段×三候选的完整GRU对照，按合并验证RMSE选模，再训练最终GRU并生成留出测试报告。即使基线胜出，也保存GRU权重、最佳轮次和独立指标。

输出写入新的 `代码/光伏/运行结果/<时间戳>/`，包含模型、缩放器、分区/特征配置、逐轮CSV/JSON、日志、loss曲线、测试指标、预测与回测图。RMSE/MAE/R²保存为数值，不另外生成指标曲线或柱状图。

`--max-gru-samples` 为正数的运行属于显式限样调试，不能当作完整训练。训练与在线预测是分开的操作。

## 命令行

以下命令在 `代码/光伏` 目录执行：

```powershell
Set-Location .\代码\光伏
$pvData = "..\数据\光伏数据\黄花站_2025年_15分钟光伏模拟数据.csv"

# 完整训练；输出目录请使用新的名称
python -m pv_forecast.cli train `
  --input $pvData --output-dir "运行结果\新训练目录" `
  --history-weather-columns "SWGDN,temperature_c,cloud_cover" `
  --latitude 30.90 --longitude 111.35 --rolling-validation --require-gru

# 用实际生成的模型目录替换“已有模型目录”
python -m pv_forecast.cli predict `
  --input $pvData --bundle-dir "运行结果\已有模型目录" `
  --output "本地输出\forecast.json"

# 最后一个完整24小时的回放；完整全年评分使用backtest
python -m pv_forecast.cli evaluate `
  --input $pvData --bundle-dir "运行结果\已有模型目录" `
  --output "本地输出\evaluation.json"
```

输入最少包含 `timestamp,pv_power_kw`，功率单位为kW、口径为瞬时采样功率。历史气象通过 `--history-weather-columns` 指定；列名映射需要显式配置。能量列不能直接作为功率，转换需要明确时间区间。

### 冻结权重再校准

```powershell
python -m pv_forecast.cli recalibrate `
  --input $pvData --bundle-dir "运行结果\已有模型目录" `
  --output-dir "运行结果\新校准目录" --night-lower-bound zero
```

按父包原验证分区校准昼夜区间，不拟合GRU权重。夜间零下界只将区间下端点设为0，上端点和点预测保留。输出目录必须不存在，且位于父包之外。

### 可选点预测修正

```powershell
python -m pv_forecast.cli correct-point `
  --input $pvData --bundle-dir "运行结果\完整滚动训练模型目录" `
  --output-dir "运行结果\新点修正目录" --batch-windows 1024
```

要求父包是未修正的candidate GRU、已配置太阳坐标，并包含三段×三候选的完整训练证据。修正使用原GRU预测、前日/同刻均值、近期功率和历史气象摘要、未来日历/太阳位置，不读取未来实测天气标签。

正则系数默认比较0.001/0.01/0.1，按合并验证RMSE选择，同时检查白天误差与最差分区。最终修正只拟合训练窗口，先冻结修正和区间参数，再评价测试分区。夜间点预测设为零，这一假设需要实测验证。

### 全窗口回测

```powershell
python -m pv_forecast.cli backtest `
  --input "待评估数据.csv" `
  --bundle-dir "运行结果\原模型目录" "运行结果\修正模型目录" `
  --output "本地输出\backtest.json" --batch-windows 1024
```

评价每个完整15分钟起点，批大小只控制内存，不限制样本。缺失目标行不作为补值真值计分。报告全天/白天/夜间、月份、提前步数组及事后天气标签分组指标；不训练、不选模、不再校准。

### 未来天气预报

预报归档需包含 `issued_at,timestamp` 及指定气象列。相关命令可添加：

```powershell
--future-weather-input "预报归档.csv" --future-weather-columns "SWGDN,temperature_c,cloud_cover"
```

每个预测起点只选择不晚于该起点发布的最新完整批次，不混用发布批次或起点之后的修订值。历史训练需要历史预报归档，不能将同步气象观测当作未来预报。当前已报告的模拟指标未使用真实未来预报。

## 最新模拟评估结果

以下是2026-10-09完成的“GRU＋Ridge修正”候选评估，并非本次上传时重新训练得到的结果。

| 数据 | 范围 | RMSE | MAE | R² |
| --- | --- | ---: | ---: | ---: |
| 2025回顾测试 | 全天 | 394.27 kW | 208.54 kW | 0.6651 |
| 2025回顾测试 | 白天 | 605.17 kW | 491.31 kW | 0.4396 |
| 2026新随机模拟轨迹 | 全天 | 467.35 kW | 241.48 kW | 0.7078 |
| 2026新随机模拟轨迹 | 白天 | 659.47 kW | 480.82 kW | 0.5081 |

2026轨迹使用seed=20261010，先冻结模型再生成数据，完整评价34,657个预测窗口、3,327,072个重叠标签。同一轨迹原GRU白天RMSE为710.58 kW，修正后下降7.19%；12个月白天RMSE均下降。修正后白天区间覆盖82.41%，平均宽度1515.16 kW。

按天气类型分组，晴天RMSE1014.44→1057.40 kW、多云395.33→440.62 kW变差，阴天和雨天改善。晴天区间覆盖仅49.36%；8/9/10月白天覆盖低于名义80%。整体误差下降不能等同于所有天气都改善。

2025测试数据此前已查看；验证参与过GRU早停且部分与修正选择重叠。2026轨迹来自已知模拟器，重叠标签不独立。夜间大量零功率会改善全天指标，因此需同时关注白天结果。经验区间不保证未来覆盖率。

## 测试与故障处理

```powershell
Set-Location .\代码\光伏
python -m unittest discover -s tests -v
```

历史完整回归53项通过，覆盖窗口边界、训练专属缩放、未来信息隔离、预报发布时间、GRU训练/早停/保存重载、独立基线区间、点修正及测试标签不影响拟合参数等行为。详细摘要见 [验证记录](docs/验证记录.json)。

缺少权重、修正文件或必要天气输入时，使用已验证的基线及该基线独立校准的区间，状态标记为 `degraded`；没有可用基线、历史不足或不能衔接起点时明确返回 `unavailable`。JSON中的 `available` 表示本次输入下能生成预测，接入调度前还应核验站点、时间窗及模型版本。

真实应用仍需实测功率、按发布时间保存的气象预报、坐标及功率口径核验、真实回测和生产发布/监控流程。预测不能替代并网容量、保护阈值或设备硬约束。
