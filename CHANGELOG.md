# Changelog

## 0.1.0 (2026-09-29)


### Features

* add audit-log ([#49](https://github.com/indykite/intent-agent/issues/49)) ([6e7f46c](https://github.com/indykite/intent-agent/commit/6e7f46cd90d9f3fd1ccffe50b53b2818744d531f))
* add context filters and qualifiers ([#60](https://github.com/indykite/intent-agent/issues/60)) ([c1aba4d](https://github.com/indykite/intent-agent/commit/c1aba4d9203c336c15ffe77f056f5babbc6b55b6))
* add evaluation suite ([#75](https://github.com/indykite/intent-agent/issues/75)) ([ed64633](https://github.com/indykite/intent-agent/commit/ed64633f63a95f9db08595ec1baa342403bae30c))
* add evaluation suite latency and feedback ([#77](https://github.com/indykite/intent-agent/issues/77)) ([9f66706](https://github.com/indykite/intent-agent/commit/9f66706325370963f06b08b574bf1a1e33fc1612))
* add metrics ([#17](https://github.com/indykite/intent-agent/issues/17)) ([5430d2f](https://github.com/indykite/intent-agent/commit/5430d2fa2e7791d4dc42082d3c0734f99c5539ac))
* add optional in-cluster redis to helm chart ([#40](https://github.com/indykite/intent-agent/issues/40)) ([bcda921](https://github.com/indykite/intent-agent/commit/bcda92166e2ec956651508c9df443195c6c8626a))
* add policy-based action derivation ([#68](https://github.com/indykite/intent-agent/issues/68)) ([eb9fda8](https://github.com/indykite/intent-agent/commit/eb9fda8adcd75831fa3c7c620eee0978296d0b79))
* add structured logging ([#19](https://github.com/indykite/intent-agent/issues/19)) ([d9cf8fb](https://github.com/indykite/intent-agent/commit/d9cf8fbd16c9404c2c76b02e9478694fe222a2b7))
* add token error management ([#65](https://github.com/indykite/intent-agent/issues/65)) ([2b0616f](https://github.com/indykite/intent-agent/commit/2b0616f64ca58f9e57b16f0bdce1dc3d9eee2b28))
* add validation loop with retry ([#45](https://github.com/indykite/intent-agent/issues/45)) ([5051e1f](https://github.com/indykite/intent-agent/commit/5051e1f41b36b94dadd537d87777a0d832e404cf))
* add validation loop with retry ([#45](https://github.com/indykite/intent-agent/issues/45)) ([#46](https://github.com/indykite/intent-agent/issues/46)) ([9e4b55e](https://github.com/indykite/intent-agent/commit/9e4b55e8d643fca90426f5eff374e7426e1324ac))
* constrain Gemini output with schema-derived structured output ([#43](https://github.com/indykite/intent-agent/issues/43)) ([e9070a2](https://github.com/indykite/intent-agent/commit/e9070a2355d4a8c7fd38c06a377cbdc58790cfa4))
* deterministic subject binding with schema-derived type (ENG-9051) ([#18](https://github.com/indykite/intent-agent/issues/18)) ([ff66a0f](https://github.com/indykite/intent-agent/commit/ff66a0f5dc1cc70313d5ac44b1b385097120b616))
* enforce cel in authzen ([#62](https://github.com/indykite/intent-agent/issues/62)) ([db5a420](https://github.com/indykite/intent-agent/commit/db5a420c617477fb4d3dccb70e228bbab7d556b6))
* fix _bind_subject overrides in search ([#64](https://github.com/indykite/intent-agent/issues/64)) ([3eb4d6b](https://github.com/indykite/intent-agent/commit/3eb4d6b0525e3e54af8d4176a1b77951855a57a7))
* reject access tokens scoped to a different agent ([#74](https://github.com/indykite/intent-agent/issues/74)) ([d3d8206](https://github.com/indykite/intent-agent/commit/d3d8206758da1730c60ac218569ee92c4623d83e))
* retry schema fetch per-request, validate SUBJECT_TYPE per-request, lower schema TTL to 5m ([#35](https://github.com/indykite/intent-agent/issues/35)) ([8f4be77](https://github.com/indykite/intent-agent/commit/8f4be7709cf6892b128cd142eb328d46ed6ce237))
* schema TTL + background refresh with pluggable cache (ENG-9052) ([#27](https://github.com/indykite/intent-agent/issues/27)) ([d95cfe9](https://github.com/indykite/intent-agent/commit/d95cfe9a5b894f5df737b629f994eeec6aca60f0))
* subject type/id from caller params ([#59](https://github.com/indykite/intent-agent/issues/59)) ([d5b74d1](https://github.com/indykite/intent-agent/commit/d5b74d1cfd8380bffb0f2a31f0c2b396109112c5))
* validate access_token via RFC 7662 introspection ([#9](https://github.com/indykite/intent-agent/issues/9)) ([032f024](https://github.com/indykite/intent-agent/commit/032f02487af4d9e57c4759bc1b33f38fe857f37b))


### Bug Fixes

* allow any schema node type as SUBJECT_TYPE, not just DigitalTwin… ([#39](https://github.com/indykite/intent-agent/issues/39)) ([e7999c2](https://github.com/indykite/intent-agent/commit/e7999c2d762554419887bd9c8ebc7966b63155e5))
* distinguish empty graph from schema unavailable ([#44](https://github.com/indykite/intent-agent/issues/44)) ([97ced62](https://github.com/indykite/intent-agent/commit/97ced6203e5e0b9333366d7f0812f790271225ab))
* fix helm parsing issue ([#53](https://github.com/indykite/intent-agent/issues/53)) ([3be72e6](https://github.com/indykite/intent-agent/commit/3be72e6b1e28495149e37256230eef80a65e9ae3))
* lite version of gemini model is faster, latest version is gemini… ([#34](https://github.com/indykite/intent-agent/issues/34)) ([00a75c5](https://github.com/indykite/intent-agent/commit/00a75c5cf4b2101969d4077225e79562c2b77a83))
* remove unnecessary act/aud delegation check from access_token validation ([#36](https://github.com/indykite/intent-agent/issues/36)) ([2a3cb4b](https://github.com/indykite/intent-agent/commit/2a3cb4bfb62c10f1906b1f638d52c9b8b8779ad3))
* require subject_id, not just subject_type ([#72](https://github.com/indykite/intent-agent/issues/72)) ([ccd5636](https://github.com/indykite/intent-agent/commit/ccd5636f6956b9da861b0f0588913481d2c338ed))
* schema debug logs, warn on empty model ([#61](https://github.com/indykite/intent-agent/issues/61)) ([11aa389](https://github.com/indykite/intent-agent/commit/11aa3899a0c0a251853c4120e27877c40634f895))
