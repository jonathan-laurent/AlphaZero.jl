# Miscellaneous

We gather on this page the documentation of internal utility functions that are
omitted from the manual for brievety.

```@meta
CurrentModule = AlphaZero
```

## Batchifying oracles

```@docs
Batchifier
Batchifier.launch_server
Batchifier.client_done!
Batchifier.BatchedOracle
```

## KnetLib and FluxLib

```@docs
#KnetLib
#KnetLib.KNetwork
#KnetLib.TwoHeadNetwork
FluxLib
FluxLib.FluxNetwork
FluxLib.TwoHeadNetwork
```

## Utilities

```@autodocs
Modules = [AlphaZero.Util]
```