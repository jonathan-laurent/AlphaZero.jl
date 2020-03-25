# [User Interface](@id ui)

```@meta
CurrentModule = AlphaZero
```

```@docs
UserInterface
```

```@meta
CurrentModule = AlphaZero.UserInterface
```

## [Session](@id session)

```@docs
Session
```

![Session CLI (first iteration)](../assets/img/ui-first-iter.png)

```@docs
Session(::Env) # Strangely, this includes all constructors...
resume!
save
play_interactive_game
start_explorer(::Session)
SessionReport
```

## [Explorer](@id explorer)

```@docs
Explorer
```

![Explorer](../assets/img/explorer.png)

```@docs
start_explorer(::Explorer)
```
