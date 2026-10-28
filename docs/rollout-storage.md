# Rollout 与 Artifact 存储

`RuntimeState` 决定业务事务与引用，`RolloutStore` 拥有 SQLite 连接及 blob I/O。每个 Store 使用单一 owner 线程；不同实例和进程通过同一 SQLite 数据库的 `BEGIN IMMEDIATE` 写锁协调。支持共享数据库及其同名 `.artifacts` 目录，不支持多个独立数据库指向同一个 blob 目录。

## 输入版本是执行契约

普通模型请求和摘要请求都在捕获 Step 时保存正整数 `input_revision`，准备请求时校验，派发和执行前与当前用户输入版本比较。恢复使用保存的版本，不重新读取时钟、推算请求顺序或填入默认版本。

缺少或非法版本的旧请求不再支持继续执行。Session 与导出的 TurnExecutor 恢复入口在准备模型、恢复 owner 或修改持久状态前校验，并提示新建 Turn；可安全取消的旧 Turn 仍可 abort。拒绝不删除数据库，也不重写旧请求。只读历史仍展示保留的事实，不推算缺失版本、不授予执行权限；有效请求继续按显式版本过滤已被新输入取代的答复。更新后的当前格式请求继续支持崩溃恢复、补充输入和工具 reconciliation。

## Blob 发布与引用提交

`state.commit_artifact(content=...)` 接受 `bytes` 或 `pathlib.Path`。文件由调用方持有，在操作完成前须保持不变；Store 复制文件，不移动、硬链接或接管源文件。调用方已有内容摘要时，可传 `expected_sha256`，阻止路径替换后将不同内容误认为原证据。工具输出持久化使用文件路径及预先计算的摘要。

```python
artifact = await state.run_async(
    state.commit_artifact,
    turn_id=turn_id,
    content=source_path,
    media_type="application/octet-stream",
    name="result.bin",
    expected_sha256=known_digest,  # 没有预先计算的摘要时可省略
)
```

同一写事务内依次执行：

1. 校验运行中的 Turn，取得与 GC 相同的写锁。
2. 以 1 MiB 块复制到独立临时文件，同时计算 SHA-256 和实际字节数。
3. fsync 临时内容；校验预期摘要；复用经校验的同哈希 blob，或原子 rename 并 fsync blob 目录。
4. 追加 `artifact_committed`，更新投影并 COMMIT。通知只在 COMMIT 后发布。

新建目录的父目录项也在引用提交前 fsync。低层 `install_blob` 必须位于写事务中，避免调用方绕开 GC 互斥边界。文件输入在打开的描述符上检查常规文件类型，拒绝目录、管道和设备，不因误传 FIFO 阻塞 owner；复制前后检查源文件变化。产物元数据使用实际复制结果，不用源文件 stat 代替大小与摘要。文件输入的持久化缓冲区有界；`bytes` 输入本身、大型工具结果的内存表示和 JSON 序列化仍占内存，`read_artifact()` 也仍返回完整 bytes。

| 中断位置 | 数据库引用 | 文件结果 |
| --- | --- | --- |
| 临时文件写入期间 | 不提交 | 异常清理临时文件；进程崩溃可留下临时孤儿 |
| rename 后、COMMIT 前 | 回滚 | 可留下完整孤儿 blob，等待 GC |
| COMMIT 后 | 已提交 | 必须存在完整、匹配 hash/size 的 blob |

不会通过回滚文件操作伪装 SQLite 与文件系统具有统一原子性。源读取失败、摘要不符或已存在 blob 损坏时，不提交新引用。调用方取消会等待 owner 的真实提交或回滚结果。

## GC 是可部分完成的维护操作

`gc_unreferenced_artifacts()` 返回 `ArtifactGCReport(removed, errors)`。它在同一 SQLite 写锁内读取已提交引用，再删除超过宽限期的孤儿 blob 和临时文件。引用中的 blob 永远跳过，包括 mtime 已很老的复用内容。默认一小时宽限期是保留策略，不承担并发互斥职责；宽限期必须是有限正数。

单个文件的删除或检查失败会记录错误，并继续处理其他候选；实际删除项保留在 `removed`。目录 fsync 失败也记录为删除持久性错误。重跑可继续清理；不能宣称失败前的删除已回滚。SQLite 获取锁、查询或事务失败仍抛异常，调用方不得把异常理解为文件恢复。

```python
report = await state.run_async(state.gc_unreferenced_artifacts, min_age_seconds=3600)
for relative_path in report.removed:
    print("removed", relative_path)
for error in report.errors:
    print("cleanup error", error)
```

大文件复制和 GC 扫描会占用当前 owner 与 SQLite 写锁；其他实例最多等待配置的 busy timeout，超时明确失败。本实现选择单一锁协议保证正确性，不承诺大文件期间的并行写吞吐。所有写入者与清理者须遵守同一协议；外部直接删除文件、混用不遵守协议的旧程序和不可靠文件系统不在该保证内。

## 验证

回归覆盖 bytes/文件的异常与独立进程硬退出边界、真实独立进程 GC 与提交/回滚竞争、源摘要不符、分块读取、GC 部分失败与重跑，以及新请求拒绝非法输入版本和 Session 恢复前拒绝旧格式。CLI/SDK 的补充输入、摘要、取消和工具结果重开恢复继续走同一持久化实现。
