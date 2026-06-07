# Changelog

## [0.6.1](https://github.com/guywaldman/rxgraph/compare/v0.6.0...v0.6.1) (2026-06-07)


### Bug Fixes

* publish single Rust crate with Py03 bindings behind a FF ([dc39173](https://github.com/guywaldman/rxgraph/commit/dc391732adcb85753c682304995d82a0a8265d05))

## [0.6.0](https://github.com/guywaldman/rxgraph/compare/v0.5.0...v0.6.0) (2026-06-06)


### Features

* improve ergonomics and setup of custom Rust kernels ([#27](https://github.com/guywaldman/rxgraph/issues/27)) ([dcfb52c](https://github.com/guywaldman/rxgraph/commit/dcfb52c16276958825ba9047c239cb465bce703f))

## [0.5.0](https://github.com/guywaldman/rxgraph/compare/v0.4.0...v0.5.0) (2026-06-06)


### Features

* add support for writing Rust search kernel plugin ([#25](https://github.com/guywaldman/rxgraph/issues/25)) ([dba4e19](https://github.com/guywaldman/rxgraph/commit/dba4e1997bb817e50a74bb895f1f8367fb82f880))

## [0.4.0](https://github.com/guywaldman/rxgraph/compare/v0.3.1...v0.4.0) (2026-06-05)


### Features

* add support for list argmin/argmax ops ([8841fae](https://github.com/guywaldman/rxgraph/commit/8841fae50e2cab41833d4d6f26d72bf0c203d835))

## [0.3.1](https://github.com/guywaldman/rxgraph/compare/v0.3.0...v0.3.1) (2026-06-04)


### Performance Improvements

* split topology storage, reduce allocations, short-circuit boolean ops, optimize u64 queries ([#22](https://github.com/guywaldman/rxgraph/issues/22)) ([f50dcfa](https://github.com/guywaldman/rxgraph/commit/f50dcfa6eaa5bc83fbef634a2c9c3d985a300d36))

## [0.3.0](https://github.com/guywaldman/rxgraph/compare/v0.2.0...v0.3.0) (2026-06-02)


### Features

* heavily optimize/cap RSS & introduce `from_lazy` ([#19](https://github.com/guywaldman/rxgraph/issues/19)) ([9ae9536](https://github.com/guywaldman/rxgraph/commit/9ae95362d1b6024ac47ed79c737619d354f861bc))
* introduce support for optionally reporting on progress ([#21](https://github.com/guywaldman/rxgraph/issues/21)) ([9dbd5f3](https://github.com/guywaldman/rxgraph/commit/9dbd5f3f6e92bd1b43be663f3c3265fcfa339079))

## [0.2.0](https://github.com/guywaldman/rxgraph/compare/v0.1.0...v0.2.0) (2026-06-01)


### Features

* add support for list and struct operations ([#13](https://github.com/guywaldman/rxgraph/issues/13)) ([5b33577](https://github.com/guywaldman/rxgraph/commit/5b335772054d3aae777bedc37efd809209d5e714))
* extend DSL support ([#17](https://github.com/guywaldman/rxgraph/issues/17)) ([07ecdbe](https://github.com/guywaldman/rxgraph/commit/07ecdbe4dbeedd87cbac6c87f1edfadf8da21461))


### Bug Fixes

* **ci:** actually migrate crates.io publishing off TP ([dea883f](https://github.com/guywaldman/rxgraph/commit/dea883f8f73c021e4a75c7011425235f823ef068))
* **ci:** allow release-please to inherit secrets ([02ec449](https://github.com/guywaldman/rxgraph/commit/02ec4498d597fe36669aa4f1427b77ddec4aec0c))
* **ci:** attempt to resolve publishing & secret retrieval ([fb12050](https://github.com/guywaldman/rxgraph/commit/fb12050bc676917c00a894bb2997c4eab6996443))
* **ci:** explicitly set token in Cargo publish job ([999c281](https://github.com/guywaldman/rxgraph/commit/999c28163555e5d9e1216deb3e044f832a67c876))
* **ci:** fix release pipeline trigger ([85c3978](https://github.com/guywaldman/rxgraph/commit/85c397824cab0980f4a41485059e99bef342c3dd))
* **ci:** fix release-please ([d3c9253](https://github.com/guywaldman/rxgraph/commit/d3c9253961e44daef16b92fc8e5fd08c9b44edba))
* **ci:** migrate off PT for PyPI publishing ([5a80963](https://github.com/guywaldman/rxgraph/commit/5a809636040740625d2f1b0d58ff4b2ffba0a718))
* **ci:** publish Rust crate with token ([3432494](https://github.com/guywaldman/rxgraph/commit/3432494b8231a966bfd77f5bfd26000b57f6b41b))
* **ci:** release from repository root ([#3](https://github.com/guywaldman/rxgraph/issues/3)) ([01b14c0](https://github.com/guywaldman/rxgraph/commit/01b14c093b36e37b3efbe5c7257b7d5e61370463))

## [0.1.0](https://github.com/guywaldman/rxgraph/compare/v0.0.14...v0.1.0) (2026-06-01)


### Features

* add support for list and struct operations ([#13](https://github.com/guywaldman/rxgraph/issues/13)) ([5b33577](https://github.com/guywaldman/rxgraph/commit/5b335772054d3aae777bedc37efd809209d5e714))

## [0.0.14](https://github.com/guywaldman/rxgraph/compare/v0.0.13...v0.0.14) (2026-05-31)


### Bug Fixes

* **ci:** allow release-please to inherit secrets ([02ec449](https://github.com/guywaldman/rxgraph/commit/02ec4498d597fe36669aa4f1427b77ddec4aec0c))

## [0.0.13](https://github.com/guywaldman/rxgraph/compare/v0.0.12...v0.0.13) (2026-05-31)


### Bug Fixes

* **ci:** explicitly set token in Cargo publish job ([999c281](https://github.com/guywaldman/rxgraph/commit/999c28163555e5d9e1216deb3e044f832a67c876))

## [0.0.12](https://github.com/guywaldman/rxgraph/compare/v0.0.11...v0.0.12) (2026-05-31)


### Bug Fixes

* **ci:** attempt to resolve publishing & secret retrieval ([fb12050](https://github.com/guywaldman/rxgraph/commit/fb12050bc676917c00a894bb2997c4eab6996443))

## [0.0.11](https://github.com/guywaldman/rxgraph/compare/v0.0.10...v0.0.11) (2026-05-31)


### Bug Fixes

* **ci:** migrate off PT for PyPI publishing ([5a80963](https://github.com/guywaldman/rxgraph/commit/5a809636040740625d2f1b0d58ff4b2ffba0a718))

## [0.0.10](https://github.com/guywaldman/rxgraph/compare/v0.0.9...v0.0.10) (2026-05-31)


### Bug Fixes

* **ci:** actually migrate crates.io publishing off TP ([dea883f](https://github.com/guywaldman/rxgraph/commit/dea883f8f73c021e4a75c7011425235f823ef068))

## [0.0.9](https://github.com/guywaldman/rxgraph/compare/v0.0.8...v0.0.9) (2026-05-31)


### Bug Fixes

* **ci:** publish Rust crate with token ([3432494](https://github.com/guywaldman/rxgraph/commit/3432494b8231a966bfd77f5bfd26000b57f6b41b))

## [0.0.8](https://github.com/guywaldman/rxgraph/compare/v0.0.7...v0.0.8) (2026-05-31)


### Bug Fixes

* **ci:** fix release pipeline trigger ([85c3978](https://github.com/guywaldman/rxgraph/commit/85c397824cab0980f4a41485059e99bef342c3dd))

## [0.0.7](https://github.com/guywaldman/rxgraph/compare/v0.0.6...v0.0.7) (2026-05-31)


### Bug Fixes

* **ci:** fix release-please ([d3c9253](https://github.com/guywaldman/rxgraph/commit/d3c9253961e44daef16b92fc8e5fd08c9b44edba))
* **ci:** release from repository root ([#3](https://github.com/guywaldman/rxgraph/issues/3)) ([01b14c0](https://github.com/guywaldman/rxgraph/commit/01b14c093b36e37b3efbe5c7257b7d5e61370463))

## Changelog

Release notes are managed by
[release-please](https://github.com/googleapis/release-please).
