# Carrier Attach Protocol

Cyber Force 把“我”定义成一个主权运行包，carrier 只是装载和执行接口。carrier 不能定义人格，也不能越权执行。

## 目标

- 同一个人格包可以被 CLI、HTTP、本地 App、未来可穿戴或机器人装载。
- 所有载体都只通过统一输入 envelope 和统一 session contract 接入。
- 载体只能声明能力和权限，不能改写原则、身份、风格、模式或记忆门禁。

## Manifest 必填字段

- `protocol_version`
- `carrier_id`
- `name`
- `version`
- `transport`
- `capabilities`
- `permissions`
- `identity_claim`
- `session_defaults`

可选字段：

- `adapter`
- `bundle_mount`
- `integrity`
- `device`
- `metadata`

## 握手步骤

1. Carrier 发送 `CarrierManifest`。
2. Runtime 校验协议版本、能力列表、权限声明。
3. Runtime 生成 `SessionContract`：
   - 哪些 capability 被接受
   - 哪些 capability 被拒绝
   - 本轮哪些动作必须确认
   - 当前 policy version
4. 只有握手成功后，carrier 才能发送 `InputEnvelope`。

## 运行时保证

- 所有输入都归一到 `InputEnvelope`。
- 所有裁决都返回结构化 `EngineResponse`。
- `refuse / challenge / ask_clarifying / await_confirmation` 是主权级结果，carrier 必须尊重。
- 即使 carrier 有执行能力，未被 session contract 允许时也只能提议，不能擅自 act。

## 当前 MVP 边界

- 重点是装载协议、治理裁决和记忆门禁。
- 不做多平台深度接入。
- 不做复杂插件市场。
- 不默认执行高后果动作。
