# Streeter-Phelps 氧垂分析计算服务

沿排污口下游计算河流溶解氧（DO）亏氧曲线、定位亏氧最大（溶解氧最低）的临界点，
支持单参数区间批量扫参与趋势汇报，以及多排污口叠加工况（各口贡献线性叠加、
全程数值搜索所有局部亏氧峰）。仅提供 HTTP JSON 接口，无页面。

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

## 多排污口叠加（/multi/profile）

真实河段常有多个口子（工业排口、市政污水口、支流汇入口）错落在不同河程位置。
亏氧方程在给定 k1、k2 下对负荷是线性的，因此总亏氧可逐项叠加：

```
D(x) = d0·e^{-k2·x/U}  +  Σ_{x_i <= x} D_closed((x - x_i)/U; D0=0, L0_i)
       本底亏氧沿途衰减      每个已经过去的口子各贡献一份平移后的单口闭式解
```

- `d0` 是河道最上游的来水本底亏氧，随水流衰减，沿程处处存在；
- 每个口子只报自己的河程位置 `x_km` 与初始碳质 BOD 负荷 `l0`，时间原点平移到
  该口所在位置，口子上游的点不受它影响（因果性）；
- 相同位置的多个口子按**负荷相加合并为一口**的显式策略处理（线性方程下严格等价）；
- 口子提交顺序不影响结果：服务内部先按位置排序、同位合并，再求和。

叠加后总亏氧是若干段起点不同的指数函数之和，`dD/dt = 0` 一般无闭式根，
临界点改用**全程数值搜索**（`app/multi_critical.py`）：按各口子过流时刻切分段，
段内扫描导数符号、凡 + → - 穿越即框住一个局部峰，再对导数穷举二分定位；
所有局部亏氧峰全部汇报，最深者标为全局临界点（`global`），其余为次级峰
（`peaks` 中 `is_global=false`）。一个上升段都找不到时如实返回 `exists=false`。
搜索上界取「最后口子时刻 + 单口临界时刻 + 5 个衰减时标」，数学上保证不漏峰。

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
| POST | `/multi/profile` | 多排污口叠加：合成沿程曲线，数值搜索全部局部亏氧峰并标出全局临界点 |

非法输入（非数值、NaN/Inf、k1/k2/U/Csat 非正、L0 为负、D0 越界等）返回 HTTP 400。
多口工况额外拦截：排污口列表为空、口子位置/负荷为负或非有限数、口子数量超过
合理上限（500）；错误信息携带出问题口子的序号。

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

curl -s localhost:8000/multi/profile -H 'content-type: application/json' -d '{
  "d0": 2.0, "k1": 0.3, "k2": 0.6, "u": 30.0, "csat": 10.0,
  "outfalls": [
    {"x_km": 5.0,   "l0": 18.0},
    {"x_km": 150.0, "l0": 25.0}
  ],
  "x_max_km": 300.0, "n_points": 601
}'
```

`/profile` 与 `/multi/profile` 的扫描窗口二选一给 `t_max_day` 或 `x_max_km`
（服务内部用 `t=x/U` 折回时间）；`/profile` 都不给时默认扫 `t = 3/k1` 天，
`/multi/profile` 都不给时默认扫到临界点搜索上界（覆盖全部峰与衰减尾部）。

## 模块划分

| 文件 | 职责 |
|---|---|
| `app/closed_form.py` | 亏氧/BOD/DO 闭式求值，时间-河程换算 |
| `app/critical.py` | 单口临界点解析定位（一般式、k2=k1 特解、无临界点判定）与数值核验 |
| `app/scanning.py` | 单口沿程扫描、网格最优点、黄金分割细化 |
| `app/sweep.py` | 区间批量扫参与临界亏氧趋势分类 |
| `app/multi_profile.py` | 多口叠加曲线合成：本底衰减 + 各口子平移闭式解叠加 |
| `app/multi_critical.py` | 多口全程数值搜索：分段扫描导数、穷举二分定位全部局部峰、标注全局临界点 |
| `app/numeric.py` | 手写 linspace / 二分求根（含浮点精度穷举二分）/ 黄金分割 |
| `app/validation.py` | 全部输入校验（含排污口列表规范化），非法即抛 `ParameterError` |
| `app/schemas.py` | FastAPI/Pydantic 请求模型 |
| `app/reference.py` | 预置参考工况 |
| `app/main.py` | HTTP 接口层 |

多口两个模块与单口闭式路径相互独立，但共享 `closed_form.py` 的同一份闭式
求值与 `t = x/U` 换算，不另立公式。

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

- 只调大 k2：临界亏氧变小、氧垂变浅（`tests/test_critical.py`）
- k2 = k1：走特解而不是一般公式（`tests/test_critical.py`）
- 流速加倍：临界时刻不变、临界距离按比例加倍（`tests/test_critical.py`）
- L0 加倍：最大亏氧升高；单调复氧时报「无临界点」；非法系数报错
- 单口退化：多口接口只放一个口子时，曲线、临界点、临界亏氧与单口接口一致
  （`tests/test_multi.py`）
- 同位合并：两个位置完全相同的口子 ≡ 一个负荷为两者之和的口子，结果逐位一致
  （`tests/test_multi.py`）
- 因果性：下游新增口子不改变它上游已算出的曲线与临界点（`tests/test_multi.py`）
- 顺序不敏感：口子提交顺序打乱，整条曲线与全部临界点逐位不变
  （`tests/test_multi.py`）
- 多峰识别：两个隔得够远的口子鼓出两个亏氧峰，最深者标为全局临界点，其余为
  次级峰；全程无上升段时如实报「无临界点」（`tests/test_multi.py`）
