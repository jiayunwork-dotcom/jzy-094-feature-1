# Streeter-Phelps 氧垂分析计算服务

沿排污口下游计算河流溶解氧（DO）亏氧曲线、定位亏氧最大（溶解氧最低）的临界点，
并支持单参数区间批量扫参与趋势汇报。仅提供 HTTP JSON 接口，无页面。

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

非法输入（非数值、NaN/Inf、k1/k2/U/Csat 非正、L0 为负、D0 越界等）返回 HTTP 400。

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
```

`/profile` 的扫描窗口二选一给 `t_max_day` 或 `x_max_km`（服务内部用 `t=x/U` 折回时间）；
都不给时默认扫 `t = 3/k1` 天。

## 模块划分

| 文件 | 职责 |
|---|---|
| `app/closed_form.py` | 亏氧/BOD/DO 闭式求值，时间-河程换算 |
| `app/critical.py` | 临界点解析定位（一般式、k2=k1 特解、无临界点判定）与数值核验 |
| `app/scanning.py` | 沿程扫描、网格最优点、黄金分割细化 |
| `app/sweep.py` | 区间批量扫参与临界亏氧趋势分类 |
| `app/numeric.py` | 手写 linspace / 二分求根 / 黄金分割 |
| `app/validation.py` | 全部输入校验，非法即抛 `ParameterError` |
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

- 只调大 k2：临界亏氧变小、氧垂变浅（`tests/test_critical.py`）
- k2 = k1：走特解而不是一般公式（`tests/test_critical.py`）
- 流速加倍：临界时刻不变、临界距离按比例加倍（`tests/test_critical.py`）
- L0 加倍：最大亏氧升高；单调复氧时报「无临界点」；非法系数报错
