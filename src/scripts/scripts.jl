module Scripts

  using ..AlphaZero
  using ..AlphaZero.UserInterface: Session, resume!

  include("dummy_run.jl")
  export dummy_run

  include("test_game.jl")
  export test_game

  train(e::Experiment; args...) = resume!(Session(e, args...))

  train(s::String, args...) = train(Examples.experiments[s])

end