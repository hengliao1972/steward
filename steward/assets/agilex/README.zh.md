# AgileX 资产

[English](README.md) | 中文

[robot.usd](robot.usd) 保留原始模型。[runtime/template.json](runtime/template.json) 描述连杆、关节及物理参数，`runtime/*.msh` 提供几何形状，网格路径相对于模板解析。[默认配置](../../config/agilex-usd.toml) 直接加载这些已提交资产，无需进行 USD 转换。[runtime/report.json](runtime/report.json) 记录转换参数、工具版本及源文件和产物的 SHA-256 校验值。模板、网格和报告应一起保留。
