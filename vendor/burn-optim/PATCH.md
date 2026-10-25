# Local Compatibility Patch

Upstream: Burn 0.22.0, https://github.com/tracel-ai/burn/tree/v0.22.0/crates/burn-optim

Adds explicit-device optimizer restore APIs. The upstream loader materializes
state on Device::default(), which may initialize a GPU even during explicit CPU
training when multiple backend features are compiled. The application calls
load_on_device with its selected base device. Existing APIs retain their behavior.
No optimizer mathematics, checkpoint schema, or other code is changed.
