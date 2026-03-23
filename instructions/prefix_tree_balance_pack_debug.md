# balancePack 调试与可测试性策略（`csrc/prefix_tree.h` / `csrc/bindings.cpp`）

## 1. 问题定位（你已描述非常清晰）

- `csrc/prefix_tree.h` 中 `PrefixTree::balancePack` 是 `private` 成员，`pack_schedule` 内部调用。
- `csrc/bindings.cpp` 暴露的是 `PrefixTree::{build_radix_tree, pack_schedule, kernel_info}`，未直接暴露 `balancePack`。
- Python 侧存在等价实现 `prefix_attn/prefix_tree.py`，但 C++ 直接调试需求最好是在 C++ 路径下完成。
- queue 关键行为差异：
  - C++ 使用 `kv_in_CTA >= 128` 触发平均计算（“更激进”）
  - Python 使用 `kv_in_CTA > 256` 触发（“更保守”）

## 2. 方案 A：`pack_schedule` 间接测试（理想方案，并最低侵入）

### 是否理想？
是的。理由：
- 无需改变 `private` 访问权限，保持 C++ API 封装不变；
- 只暴露已有的 `pack_schedule` 和 `kernel_info`，通过对比前后状态即可检测 `balancePack` 是否按预期执行；
- 依赖最少：仅 Python 标准库 + `torch` + 项目现有接口；
- 易于 CI：对少量典型case做 deterministic 测试；
- 对现有代码改动最小，仅(ps)追加绑定字段和测试脚本。

### 现状评估
- 已有 `PrefixTreeCPP.kernel_info` 读取，若 `KernelInfo` 结构已可在 pybind 中映射，现有 `schedule_test.py` 可以直接打印和对比。
- 如果此结构不可完全访问（AttributeError），仍可在 C++ 侧补充 `bindings.cpp`：
  - `def_readonly("q_tables", &KernelInfo::q_tables)`, ...
  - `def_readonly("num_split_per_seq", &KernelInfo::num_split_per_seq)`
  - 这个改动最小且可复用。

### 推荐最小改动：
1. 继续用 `pack_schedule` 对 C++ 与 Python 的行为进行一对一比对。
2. 扩展 `csrc/bindings.cpp`，使 `KernelInfo` 的关键字段（`q_tables`, `block_tables`, `num_seqs_per_CTAs`, `CTA_ranks`, `kv_in_CTAs`, `num_split_per_seq`）可读。
3. 在 `benchmark/schedule_test.py` 中编排：
   - `run_cpp_schedule` + `run_python_schedule` 结果对比；
   - 记录差异等级（C++/Python约定阈值差异也需注释）；
   - 仅使用 `numpy`, `torch` 及标准库，避免额外依赖。
4. 对于 `balancePack` 行为验证：通过 `kernel_info`（包含 `split_per_seq` 改变）间接判断；如需直接调用可选方案 B 由下面补充。

## 目标：导出/导入机制要求

1. **可回放**：能把同一批输入（`boxes`, `kvHead`, `HRatio`）写出，并在调试或CI环境直接加载，验证新旧逻辑的一致性和性能差异。
2. **可观测**：输出 `cropPack` 内容与内部关键状态（`total_blocks`, `num_splits`, `threshold_cnt` 等）可导出，定位边界行为。
3. **低侵入**：不影响生产路径性能（调试模式可开关）。
4. **可脚本化**：与现有 benchmark（如 `benchmark/run_kernel_bench.sh` + `benchmark/benchmark_kernel.py`）配合，轻量调试数据收集。

---

## 设计步骤

### 1) `PrefixTree` 增加 debug flag 或可配置回调

在 `prefix_tree.h` 的 `PrefixTree` 类声明中添加：

```cpp
struct BalancePackDebugOptions {
    bool enabled = false;
    std::string dump_input_path;
    std::string dump_output_path;
};

class PrefixTree {
public:
    BalancePackDebugOptions balancePackDebug;
    ...
};
```

### 2) 在 `balancePack` 内部增加日志/数据导出

在函数开始处：

- 记录 `boxes.size()`, `kvHead`, `HRatio`
- `total_blocks` 计算（已有）

在输出前：

- 记录 `cropPack.size()`，首尾几个 `PackedBox` 元素（`q_table.size()`, `block_table_ptr->size()`, `kv_in_CTA`）

如果 `balancePackDebug.enabled`：

- 生成 JSON/NDJSON文件，包括：
  - 输入：`boxes` 重要字段
  - 中间：`total_blocks`, `total_packs`, `avg_blocks`,  `threshold`, `threshold_cnt`, splits decisions
  - 输出：`cropPack` 重要字段

可选：使用 `torch::Tensor` 直接 dump `proxy` 数据支持，也可用简单文本格式（CSV/TSV）。

### 3) 外部输入恢复（回放）支持

新增独立函数（同文件或工具文件）：

```cpp
bool load_balance_pack_fixture(const std::string& path, std::vector<PackedBox>& boxes, int& kvHead, int& HRatio);
bool save_balance_pack_fixture(const std::string& path, const std::vector<PackedBox>& boxes, int kvHead, int HRatio);
```

- `load`：将路径里面的 JSON/NDJSON加载至 `PackedBox`，包含 `q_table`, `block_table`, `num_seqs_per_CTA`, `CTA_rank`, `kv_in_CTA`, `split_per_seq` 等。

- `save`：将 `balancePack` 输入输出 fixture 写入。

### 4) 走查实现位置：

`csrc/prefix_tree.h`:
- `balancePack` 函数开头
- `for` 机内 key radii 计算处
- `return cropPack` 前

`csrc/bindings.cpp`：若需要 Python 从 `_prefix_attn` 导出树/调度数据做对比，也可扩展接口：

- `KernelInfo` 或 `PrefixTreeCPP` 新增 `dump_balance_pack_fixtures(path)`

---

## 是否需要调整`run_kernel_bench.sh`?

**建议保留现有不改**，首选在 Python benchmark 工具链中（`benchmark/benchmark_kernel.py`）：

- 增加 CLI 参数 `--balance-pack-debug` 和 `--balance-pack-fixture`。
- 仅当启用 debug 时执行 `PrefixTreeCPP` 的回调接口（或 test-only）导出/输入。
- `run_kernel_bench.sh` 可简单传递 `--balance-pack-debug`, `--balance-pack-fixture`，不必强制改写。

如果你确实要为覆盖所有测试路径：
- 在 `run_kernel_bench.sh` 中添加 `--balance-pack-debug` 选项，使每个 TREE/HEAD_CONFIG 根据开关写出文件：`kernel_perf/fixture-${hq}-${hkv}-${tree}.json`

---

## 现有输入/输出形态是否调整

### 建议调整点

1. `PackedBox` 中 `q_table` / `block_table_ptr` 目前是 `std::vector<int>` 或 tensor，建议在中间态输出时统一为 `std::vector<int>`（便于 JSON）。
2. 目前 `balancePack` 仅返回 `cropPack`，可保留；但如果改进算法建议添加 `std::vector<PackStat>` 输出（测试引擎可验证）

```cpp
struct PackOpTrace {
  int stage; // 0=keep,1=split,2=merge
  int input_idx;
  int output_idx;
  int kv_in_CTA;
  int num_blocks;
};
```

3. `split_per_seq` 处于类成员状态，若要严谨可将其作为 `balancePack` 入参，使逻辑更纯粹，并帮助单测。

---

## 输出格式建议（JSON schema）

1. `balance_pack_fixture.json`:

```json
{
  "meta": {"kvHead": 8, "HRatio": 2, "timestamp": 168...},
  "input": {
    "split_per_seq": [0,0,...],
    "packed_boxes": [
      {"q_table": [..], "block_table": [..], "num_seqs_per_CTA": ..., "CTA_rank": ..., "kv_in_CTA": ...},
      ...
    ]
  }
}
```

2. `balance_pack_trace.json`:

```json
{
  "stats": {
    "total_blocks": 123,
    "total_packs": 10,
    "avg_blocks": 23.2,
    "threshold": 116.0,
    "threshold_cnt": 27,
    "cur_max_split": 3
  },
  "output": {
    "cropPack": [...]
  },
  "ops": [...]
}
```

---

## 测试与验证流程

1. 先写单测：`test/test_prefix_tree_balance_pack.py`（或现有单测扩展）
   - 读取硬编码/fixture JSON，调用 `balancePack`。
   - 校验 `cropPack` 在修改前后的总 `kv_in_CTA`/`q_table.size` 不变且语义合理。

2. 工具验证：执行 `python benchmark/benchmark_kernel.py --balance-pack-debug --balance-pack-fixture fixtures/xxx.json`
   - 结果以文件保存于 `bench_debug/`。
   - 采用一致的 fixture 重放可验证改进逻辑。

3. 性能对比：通过 `run_kernel_bench.sh` 不改或加 opt 参数，比较 patch 前后 `kernel_perf.json`。

---

## 小结

- `balancePack` 的调试可从“输入/输出固定化 + 可选 trace”入手，维护 `prefix_tree.h` 内部可选输出。
- 不必强制改 `run_kernel_bench.sh`，先功能设计在 benchmark 脚本里，然后逐步透传参数。
- 现有输入输出可兼容（vector->json），建议通过结构化trace组件增强透明度。

---

## 3. 新增 balancePackSota 函数开发与 C++ 内部对比策略

### 3.1 背景与目标

为了优化负载均衡性能，我们计划在 C++ 中基于现有 `balancePack` 函数开发一个新的 `balancePackSota` 函数。该函数将完全在 C++ 层面实现，避免依赖 Python 的分配机制。目标是直接对比两个 C++ 函数（`balancePack` 和 `balancePackSota`）的性能和分组结果，以便评估优化效果。

- **性能对比**：测量两个函数的执行时间、内存使用等指标。
- **分组结果对比**：比较输出 `cropPack` 的结构、CTA 分配、KV 负载分布等。
- **低侵入性**：新函数作为 `PrefixTree` 的私有成员，仅通过测试接口暴露。

### 3.2 C++ 代码修改

#### 3.2.1 在 `csrc/prefix_tree.h` 中添加新函数

在 `PrefixTree` 类中，添加 `balancePackSota` 私有成员函数。假设 `balancePackSota` 实现了一个改进的负载均衡算法，例如更智能的分组策略或更好的阈值计算。

```cpp
class PrefixTree {
private:
    // 现有 balancePack 函数
    std::vector<PackedBox> balancePack(const std::vector<PackedBox>& boxes, int kvHead, int HRatio);

    // 新增 balancePackSota 函数（示例实现，需根据具体优化逻辑调整）
    std::vector<PackedBox> balancePackSota(const std::vector<PackedBox>& boxes, int kvHead, int HRatio) {
        // 示例：改进的负载均衡逻辑
        // 1. 计算总块数和平均值（类似原函数）
        int total_blocks = 0;
        for (const auto& box : boxes) {
            total_blocks += box.block_table_ptr->size();
        }
        double avg_blocks = static_cast<double>(total_blocks) / boxes.size();

        // 2. 改进阈值计算（例如，更动态的阈值）
        double threshold = avg_blocks * 1.2;  // 示例：比原函数更保守的阈值

        // 3. 分组逻辑（简化示例，实际需实现完整算法）
        std::vector<PackedBox> cropPack;
        for (const auto& box : boxes) {
            if (box.block_table_ptr->size() > threshold) {
                // 拆分逻辑（需实现）
                // ...
            } else {
                cropPack.push_back(box);
            }
        }

        // 更新 split_per_seq 等状态
        // ...

        return cropPack;
    }

public:
    // 现有公共接口
    void pack_schedule(const std::vector<int>& MNWs, int HRatio, int kvHead);

    // 新增测试接口（仅用于对比）
    std::vector<PackedBox> test_balancePack(const std::vector<PackedBox>& boxes, int kvHead, int HRatio) {
        return balancePack(boxes, kvHead, HRatio);
    }

    std::vector<PackedBox> test_balancePackSota(const std::vector<PackedBox>& boxes, int kvHead, int HRatio) {
        return balancePackSota(boxes, kvHead, HRatio);
    }
};
```

**说明**：
- `balancePackSota` 的实现需根据具体优化策略编写。上例为简化模板，实际应包含完整的拆分、合并逻辑。
- 添加了 `test_balancePack` 和 `test_balancePackSota` 作为公共测试接口，用于从 Python 调用对比。

#### 3.2.2 在 `csrc/bindings.cpp` 中暴露测试接口

在 `pybind11` 绑定中，添加新函数的绑定，以便 Python 可以调用。

```cpp
PYBIND11_MODULE(_prefix_attn, m) {
    // 现有绑定
    py::class_<PrefixTree>(m, "PrefixTreeCPP")
        .def(py::init<int>())
        .def("build_radix_tree", &PrefixTree::build_radix_tree)
        .def("pack_schedule", &PrefixTree::pack_schedule)
        .def_readonly("kernel_info", &PrefixTree::kernel_info)
        // 新增测试接口绑定
        .def("test_balancePack", &PrefixTree::test_balancePack)
        .def("test_balancePackSota", &PrefixTree::test_balancePackSota);
}
```

**说明**：
- 这允许 Python 通过 `PrefixTreeCPP` 实例直接调用 `test_balancePack` 和 `test_balancePackSota`。

### 3.3 项目配置修改

#### 3.3.1 编译配置

确保 C++ 代码编译时包含新函数。修改 `setup.py` 或相应的构建脚本（如果使用 CMake），添加必要的编译标志。

- 如果使用 `setup.py`，确保 `csrc/` 中的文件被编译。
- 运行 `python setup.py build_ext --inplace` 或 `pip install -e .` 来重新编译扩展。

#### 3.3.2 依赖检查

- 确保 CUDA 和 CUTLASS 已安装（如果新函数依赖 GPU 计算）。
- 运行测试前，验证 `_prefix_attn` 扩展可以加载。

### 3.4 修改 `schedule_test.py` 以支持 C++ 内部对比

#### 3.4.1 新增 C++ 函数对比函数

在 `schedule_test.py` 中，添加函数来运行和对比两个 C++ 函数。

```python
# 在 SECTION 2 中添加

def run_cpp_balancePack(tree: PrefixTreeCPP, boxes: List[PackedBox], kvHead: int, HRatio: int) -> List[PackedBox]:
    """运行 C++ balancePack 函数。"""
    return tree.test_balancePack(boxes, kvHead, HRatio)

def run_cpp_balancePackSota(tree: PrefixTreeCPP, boxes: List[PackedBox], kvHead: int, HRatio: int) -> List[PackedBox]:
    """运行 C++ balancePackSota 函数。"""
    return tree.test_balancePackSota(boxes, kvHead, HRatio)

# 在 SECTION 4 中添加对比函数

def compare_cpp_functions(cropPack1: List[PackedBox], cropPack2: List[PackedBox], label1: str, label2: str) -> bool:
    """对比两个 C++ 函数的输出 cropPack。"""
    # 简化比较：检查长度和基本字段
    if len(cropPack1) != len(cropPack2):
        print(f"❌ DIFF  cropPack size: {label1}={len(cropPack1)}, {label2}={len(cropPack2)}")
        return False

    for i, (box1, box2) in enumerate(zip(cropPack1, cropPack2)):
        if box1.kv_in_CTA != box2.kv_in_CTA or len(box1.q_table) != len(box2.q_table):
            print(f"❌ DIFF  Box[{i}]: kv_in_CTA {label1}={box1.kv_in_CTA} vs {label2}={box2.kv_in_CTA}")
            return False

    print(f"✅ PASS  {label1} vs {label2}")
    return True

# 在 SECTION 7 中修改主测试函数

def run_single_test_cpp_compare(name: str, seq_lens: List[int], block_table: List[List[int]], block_size: int, HRatio: int, kvHead: int) -> bool:
    """运行单个测试用例：对比两个 C++ 函数。"""
    print(f"\n{'#'*70}")
    print(f"  TEST CASE: {name} (C++ Internal Compare)")
    print(f"  batch_size={len(seq_lens)}, block_size={block_size}")
    print(f"  HRatio={HRatio}, kvHead={kvHead}")

    # 构建树和初始 boxes（假设有获取 boxes 的方法）
    tree = PrefixTreeCPP(block_size)
    padded_tensor = pad_block_table(block_table)
    tree.build_radix_tree(seq_lens, padded_tensor)
    # 注意：需要从 tree 中提取初始 boxes，可能需添加接口
    boxes = tree.get_initial_boxes()  # 假设添加此接口

    # 运行两个函数
    cropPack_old = run_cpp_balancePack(tree, boxes, kvHead, HRatio)
    cropPack_new = run_cpp_balancePackSota(tree, boxes, kvHead, HRatio)

    # 打印结果（简化）
    print(f"  Old balancePack: {len(cropPack_old)} boxes")
    print(f"  New balancePackSota: {len(cropPack_new)} boxes")

    # 对比
    comp = compare_cpp_functions(cropPack_old, cropPack_new, "balancePack", "balancePackSota")
    print(f"  test {name} result: {comp}")

    return comp

# 在主函数中调用
if __name__ == "__main__":
    # 现有代码
    all_cases = MANUAL_CASES
    run_compare_cases(all_cases)

    # 新增 C++ 内部对比
    print("\n=== C++ Internal Compare ===")
    for case in all_cases:
        run_single_test_cpp_compare(**case)
```

**说明**：
- 假设 `PackedBox` 在 Python 中可访问（需在 `bindings.cpp` 中绑定）。
- 添加了 `get_initial_boxes` 接口（需在 C++ 中实现），用于获取 `pack_schedule` 前的 boxes。
- 性能测量可使用 `time.time()` 包装函数调用。

#### 3.4.2 性能测量

在对比函数中添加时间测量：

```python
import time

def measure_performance(func, *args):
    start = time.time()
    result = func(*args)
    end = time.time()
    return result, end - start

# 在 run_single_test_cpp_compare 中
cropPack_old, time_old = measure_performance(run_cpp_balancePack, tree, boxes, kvHead, HRatio)
cropPack_new, time_new = measure_performance(run_cpp_balancePackSota, tree, boxes, kvHead, HRatio)

print(f"  Old time: {time_old:.4f}s")
print(f"  New time: {time_new:.4f}s")
```

### 3.5 测试与验证流程

1. **编译与安装**：重新编译 C++ 扩展，确保新函数可用。
2. **运行测试**：执行修改后的 `schedule_test.py`，观察对比结果。
3. **迭代优化**：根据对比结果调整 `balancePackSota` 的实现。
4. **扩展测试**：添加更多测试用例，覆盖边界情况。

### 3.6 小结

- 通过在 C++ 中添加 `balancePackSota` 和测试接口，实现纯 C++ 层面的对比。
- 修改 `schedule_test.py` 以调用和比较两个函数的输出和性能。
- 此方案保持了项目的封装性，同时提供了直接的性能评估途径。
