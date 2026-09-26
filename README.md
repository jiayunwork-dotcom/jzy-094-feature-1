# Streeter-Phelps 氧垂分析计算服务

沿排污口下游计算河流溶解氧（DO）亏氧曲线、定位亏氧最大（溶解氧最低）的临界点，
并支持单参数区间批量扫参与趋势汇报；**多排污口工况**可同时接收一串河程位置、
负荷各不相同的排口（工业排口、市政污水口、支流汇入口），叠加各口对下游同一
点的贡献，并沿全程找出全部局部亏氧峰、标注全局最深的临界点。仅提供 HTTP
JSON 接口，无页面。

- 运行时：Python 3.12 + FastAPI
- 求根、黄金分割、区间扫描全部用标准库手写（`app/numeric.py`），不引入 NumPy/SciPy
- 时间一律以「天」为单位，河程一律以「公里」为单位，二者通过 `t = x / U` 换算，
  绝不能把公里数直接代进以天为单位的公式
- 流速 `U` 取 **km/day**。换算：`1 m/s = 86.4 km/day`（例如 0.35 m/s ≈ 30.24 km/day）

## 模型（教科书闭式解）

碳质 BOD：

```
L(t) = L0 · exp(-k1 · t)
```

亏氧方程 `dD/dt = k1·L - k2·D`，一般解（k1 ≠ k2）：

```
D(t) = (k1·L0)/(k2-k1) · (e^{-k1 t} - e^{-k2 t}) + D0 · e^{-k2 t}
DO(t) = Csat - D(t)
```

k1 = k2 特解（分母为零，禁止套一般公式）：

```
D(t) = (D0 + k1·L0·t) · e^{-k1 t}
```

临界点由 `dD/dt = 0` 得到，一般公式：

```
t_c = ln[ (k2/k1) · (1 - D0·(k2-k1)/(k1·L0)) ] / (k2 - k1)
x_c = U · t_c
```

k1 = k2 特解：

```
t_c = 1/k1 - D0/(k1·L0)
```

当 `k1·L0 <= k2·D0`（初始时刻亏氧不再增大，曲线单调复氧）或特解给出 `t_c <= 0` 时，
下游不存在氧垂临界点，服务明确返回 `exists=false`，绝不返回负河程。

> 数值策略：当 `|k2-k1|` 相对偏差不超过 1e-12 时按 k2=k1 走特解，避免分母趋零造成
> 数值爆炸。该策略集中在 `app/closed_form.py` 的 `rates_equal()`，临界点与闭式求值共用。

## 多排污口叠加模型

在给定 k1、k2 下亏氧方程对负荷是**线性**的。设排口 i 位于河程 x_i
（时间原点 `t_i = x_i/U`）、初始碳质 BOD 负荷 L_i，河道最上游的来水本底亏氧为
D0（沿程按 `D0·e^{-k2 t}` 传递）。下游任一时刻 t 的总亏氧为本底残余与所有
「已经经过」（t_i ≤ t，含恰好同位置）排口各自闭式贡献之和：

```
D_total(t) = cf.deficit(t,   D0_bg, 0,   k1, k2)
           + Σ_{t_i ≤ t} cf.deficit(t - t_i, 0, L_i, k1, k2)
```

每个加项就是把单口闭式解的时间原点挪到该口子所在河程；排口对其上游点没有
贡献。每个加项直接调用 `closed_form.deficit`（k2=k1 时自动走同一份特解），
不另抄近似公式。BOD 同理线性叠加。

**同位策略（明确处理）**：位置完全相同的排口按「同位负荷相加」合并为一个
等效排口（线性叠加的直接推论），合并后按河程升序规范化，因此提交顺序对
结果没有任何影响。

**临界点不再有闭式根**：叠加后总亏氧是若干段起点不同的指数函数之和，
dD/dt=0 一般不再有闭式根（三口以上尤其如此）。`app/multi_critical.py` 改用
数值搜索，且利用分段结构保证不漏峰——

- 相邻排口之间活动排口集合固定，段内 D(t) 只是两个指数的组合
  （`A·e^{-k1 t} + B·e^{-k2 t}`，k1=k2 时为 `(P+Qt)·e^{-k t}`），导数在一个段内
  至多穿越零点一次，故每段至多一个局部极大；
- 每段独立做「端点 dD/dt 符号判定 → 二分找 +→− 穿越根 → 黄金分割细化峰点」，
  沿全程把所有局部极大都找出来；
- rank=1 为全局最深临界点（临界亏氧最大，并列取最上游），其余按河程顺序
  编号为 secondary；找不到任何上升段时如实返回 `exists=false`，不凑正数；
- 全部只用标准库手写（`app/numeric.py` 的二分/黄金分割），不引入 NumPy/SciPy。

扫描窗口可用 `t_max_day` 或 `x_max_km`（自河头起，二选一，内部以 `t=x/U`
折回时间）给出；都不给时自动倍增尾段长度，直到终点导数确实 ≤ 0，保证所有
峰都在窗口内（响应里 `window.auto=true`）。手动窗口若终点处亏氧仍在上升，
响应以 `critical.still_rising_at_end=true` 提示窗口外可能还有峰。

## 预置参考工况

`GET /reference` 返回可直接核对的参考工况（排污口下游明显氧垂）：

| 参数 | 值 | 含义 |
|---|---|---|
| D0 | 2.0 mg/L | 初始亏氧 |
| L0 | 20.0 mg/L | 初始碳质 BOD |
| k1 | 0.3 /day | 耗氧系数 |
| k2 | 0.6 /day | 复氧系数 |
| U  | 30.0 km/day | 流速（≈0.347 m/s） |
| Csat | 10.0 mg/L | 饱和溶解氧 |

解析结果：`t_c ≈ 1.959 day`，`x_c ≈ 58.8 km`，`D_c ≈ 5.56 mg/L`，`DO_c ≈ 4.44 mg/L`。

## 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET  | `/healthz` | 健康检查 |
| GET  | `/reference` | 预置参考工况与其临界点 |
| POST | `/critical` | 单一工况临界点定位（含特解、无临界点判定） |
| POST | `/profile` | 沿程扫描整条亏氧/DO 曲线，并给窗口内数值最优点 |
| POST | `/sweep` | 在 k1/k2/L0/D0/U 区间上批量扫参，汇报临界亏氧趋势 |
| POST | `/multi/critical` | 多排污口叠加：合成总亏氧曲线，数值搜索全部局部亏氧峰与全局最深临界点 |

非法输入（非数值、NaN/Inf、k1/k2/U/Csat 非正、L0 为负、D0 越界等）返回 HTTP 400；
结构错误（缺字段、多余字段、类型不符）返回 HTTP 422。

### 多排污口请求/响应要点

请求体：共用 `d0/k1/k2/u/csat`，外加 `outfalls`（非空，每个元素给 `x_km`、`l0`），
可选 `t_max_day` 或 `x_max_km`（二选一）与 `n_points`（默认 1001，2~10001）。
排口数量上限 10000。

响应：`input`（回显）、`merged_outfalls`（同位合并策略说明与规范化后的排口）、
`window`（实际扫描窗口，`auto` 标识是否自动选取）、`critical`（多峰搜索结果）、
`profile`（叠加后的沿程 BOD/亏氧/DO 曲线）。`critical` 里：

- `exists`、`n_local_maxima`、`peaks`（每个峰含 rank/kind/t_c_day/x_c_km/
  critical_deficit_mg_l/critical_do_mg_l）
- `global`（rank=1 的全局最深峰；无峰时为 null）
- `still_rising_at_end`（手动窗口终点仍在上升时为 true）

### 示例

```bash
curl -s localhost:8000/critical -H 'content-type: application/json' -d '{
  "d0": 2.0, "l0": 20.0, "k1": 0.3, "k2": 0.6,
  "u": 30.0, "csat": 10.0
}'

curl -s localhost:8000/profile -H 'content-type: application/json' -d '{
  "d0": 2.0, "l0": 20.0, "k1": 0.3, "k2": 0.6,
  "u": 30.0, "csat": 10.0,
  "x_max_km": 120.0, "n_points": 401
}'

curl -s localhost:8000/sweep -H 'content-type: application/json' -d '{
  "d0": 2.0, "l0": 20.0, "k1": 0.3, "k2": 0.6,
  "u": 30.0, "csat": 10.0,
  "parameter": "k2", "lo": 0.45, "hi": 1.2, "n_points": 6
}'

# 多排污口：两个相隔 80 km 的排口，下游会鼓两个亏氧峰
curl -s localhost:8000/multi/critical -H 'content-type: application/json' -d '{
  "d0": 2.0, "k1": 0.35, "k2": 0.6, "u": 30.0, "csat": 10.0,
  "outfalls": [
    {"x_km": 0.0,  "l0": 18.0},
    {"x_km": 80.0, "l0": 18.0}
  ]
}'
```

`/profile` 的扫描窗口二选一给 `t_max_day` 或 `x_max_km`（服务内部用 `t=x/U` 折回时间）；
都不给时默认扫 `t = 3/k1` 天。

## 模块划分

| 文件 | 职责 |
|---|---|
| `app/closed_form.py` | 亏氧/BOD/DO 闭式求值，时间-河程换算（单口与多口共享的唯一底层） |
| `app/critical.py` | 单口临界点解析定位（一般式、k2=k1 特解、无临界点判定）与数值核验 |
| `app/multi_source.py` | 多排污口曲线合成：本底衰减 + 各口闭式贡献叠加、同位合并、沿程扫描 |
| `app/multi_critical.py` | 多口工况全局/局部临界点数值搜索（分段 + 二分 + 黄金分割，独立于闭式路径） |
| `app/scanning.py` | 单口沿程扫描、网格最优点、黄金分割细化 |
| `app/sweep.py` | 区间批量扫参与临界亏氧趋势分类 |
| `app/numeric.py` | 手写 linspace / 二分求根 / 黄金分割 |
| `app/validation.py` | 全部输入校验（含多排口列表与逐口下标错误信息），非法即抛 `ParameterError` |
| `app/schemas.py` | FastAPI/Pydantic 请求模型 |
| `app/reference.py` | 预置参考工况 |
| `app/main.py` | HTTP 接口层 |

## 构建与运行

```bash
docker build -t sp-do-sag .
docker run --rm -p 8000:8000 sp-do-sag
# 服务监听固定端口 8000
```

在容器内执行测试：

```bash
docker run --rm sp-do-sag pytest -q
```

本地（已有 Python 3.12 与依赖时）：

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
pytest -q
```

## 锁定的规律（自动化测试）

单口工况：

- 只调大 k2：临界亏氧变小、氧垂变浅（`tests/test_critical.py`）
- k2 = k1：走特解而不是一般公式（`tests/test_critical.py`）
- 流速加倍：临界时刻不变、临界距离按比例加倍（`tests/test_critical.py`）
- L0 加倍：最大亏氧升高；单调复氧时报「无临界点」；非法系数报错

多排污口工况（`tests/test_multi_source.py` 内核、`tests/test_multi_api.py` 接口）：

- **单口退化一致**：只放一个排污口时，多口接口的曲线、临界点、临界亏氧
  与原单口接口数值完全对得上（临界亏氧到机器精度，位置到数值方法精度）
- **同位合并一致**：两个位置完全相同的排口 == 一个负荷相加（翻倍）的单口
- **下游因果**：任一点只统计位置不晚于该点的排口；下游新插入口子不改它
  上游已算出的曲线与临界点
- **顺序不敏感**：提交顺序打乱、位置负荷不变，整条曲线与所有临界点逐位不变
- 隔得够远的多个排口鼓多个峰：全部局部峰都被找到，rank=1 为全局最深，
  其余为 secondary；负荷过小、复氧始终压得住时如实报无临界点
- 分段搜索所用的「指数重组式」与闭式项直和式逐点相等（只换括号、不换公式）
- 空排口列表、负河程/负负荷、NaN/Inf/非数值、缺字段、排口数超上限等非法
  输入在计算前挡回，错误信息带具体排口下标，走既有 400/422 错误响应风格
