# AgileX assets

English | [中文](README.zh.md)

[robot.usd](robot.usd) is the retained source model. [runtime/template.json](runtime/template.json) describes its links, joints and physics; `runtime/*.msh` supplies the geometry, referenced relative to the template. The [default profile](../../config/agilex-usd.toml) loads these committed assets without USD conversion. [runtime/report.json](runtime/report.json) records conversion options, tool versions and source/output SHA-256 checksums. Keep the template, meshes and report together.
