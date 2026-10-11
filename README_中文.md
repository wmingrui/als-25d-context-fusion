# 代码库使用说明

这是论文代码的私有整理版。论文标题为 *Evaluating observation support and restoration pretraining in 2.5D context fusion for aerial point-cloud segmentation*。

## 内容

- `legacy/`：论文使用的训练、模型、预处理和评估代码，保留原目录关系。
- `configs/archive/`：历史实验配置，包括 DALES 新增的 45、46 号种子。
- `provenance/`：源码血统、清理前后哈希、协议锁和环境记录。
- `third_party/`：第三方来源和许可证。
- `tools/`、`tests/`：建立独立运行目录的工具及测试。

普通过程注释已清理。栅格代码中有 3 条注释被程序作为源码截取标记，不能直接删除；编码声明和必要的第三方归属也予以保留。

## 运行顺序

建议在 Linux 服务器的对应 Python 环境中使用。已有环境的版本记录在 `provenance/environments.json`。

1. 克隆仓库到新目录。
2. 运行 `python -m unittest discover -v` 检查独立目录准备工具。
3. 运行 `python tools/materialize_workspace.py --workspace /绝对路径/新运行目录`。
4. 按 `docs/REPRODUCIBILITY.md` 放置或挂载已经处理的数据；PointNeXt 还需要额外依赖和编译算子。
5. 先运行对应测试，再按历史配置启动新实验。

准备工具不会复制数据或训练权重，不会自动启动训练。原始配置留作实验记录；新运行目录里的机器路径和源码哈希会重新记录，不冒充原训练时的逐字节源码。

数据、完整权重和预测缓存没有上传。代码库目前私有，论文中仍不能写“代码已公开”。原始代码的公开发布许可证、作者信息和正式引用信息留待作者确认。完整训练和所有表格结果没有在本次整理中重跑，实际核验范围见 `docs/VERIFICATION.md`。
