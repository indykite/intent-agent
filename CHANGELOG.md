# Changelog

## [0.4.2](https://github.com/indykite/intent-agent/compare/v0.4.1...v0.4.2) (2026-10-20)


### Bug Fixes

* implement a2a compliant errors ([#109](https://github.com/indykite/intent-agent/issues/109)) ([0685c0d](https://github.com/indykite/intent-agent/commit/0685c0d8167dfb2df1b60cd7860320b9285aad78))
* mention auth requirement in agent-card ([#113](https://github.com/indykite/intent-agent/issues/113)) ([4b8499a](https://github.com/indykite/intent-agent/commit/4b8499ac10e01088ae4834499625e2bc1a717e31))

## [0.4.1](https://github.com/indykite/intent-agent/compare/v0.4.0...v0.4.1) (2026-10-20)


### Documentation

* clarify docker tags ([#110](https://github.com/indykite/intent-agent/issues/110)) ([b49cceb](https://github.com/indykite/intent-agent/commit/b49cceb08c6e5eb90c2203c878011e4743615b27))

## [0.4.0](https://github.com/indykite/intent-agent/compare/v0.3.0...v0.4.0) (2026-10-19)


### Features

* adjust skill for eval workflow ([#83](https://github.com/indykite/intent-agent/issues/83)) ([30f1d4f](https://github.com/indykite/intent-agent/commit/30f1d4f8b1119a4f1955e4b55ac345e632569037))
* derive intent only from access policies ([#94](https://github.com/indykite/intent-agent/issues/94)) ([a008e1a](https://github.com/indykite/intent-agent/commit/a008e1ac2344b3f7ff5f8e5c8e1df869b6872609))
* implement auth for every JSON-RPC method ([#102](https://github.com/indykite/intent-agent/issues/102)) ([184b2c3](https://github.com/indykite/intent-agent/commit/184b2c3d635385c82d8124cd8a45a9f2be55308a))
* tighten intent validation and rework evals ([#100](https://github.com/indykite/intent-agent/issues/100)) ([3879824](https://github.com/indykite/intent-agent/commit/38798246bc5937fc4935c360e00e79fa35f03f70))


### Documentation

* document commit-type release versioning ([#92](https://github.com/indykite/intent-agent/issues/92)) ([9adfc72](https://github.com/indykite/intent-agent/commit/9adfc727c4a908ed0e17ad9b7d6ace28086e9367))
* minor clarifications ([#105](https://github.com/indykite/intent-agent/issues/105)) ([343440f](https://github.com/indykite/intent-agent/commit/343440f4ce9433c112a5655bc4a5f8c8f6b737c4))
* minor clarifications ([#107](https://github.com/indykite/intent-agent/issues/107)) ([8f9c35d](https://github.com/indykite/intent-agent/commit/8f9c35dfb8a99ef74b471df103f3272fc19d82bc))
* minor clarifications ([#96](https://github.com/indykite/intent-agent/issues/96)) ([ce76cb8](https://github.com/indykite/intent-agent/commit/ce76cb8c703ea3aa09b6a5a4b4e4db32cd09a92b))

## 0.3.0 (2026-09-29)


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
