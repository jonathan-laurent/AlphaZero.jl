module Scripts

  using ..AlphaZero
  using ..AlphaZero.UserInterface: Session, resume!

  include("dummy_run.jl")
  export dummy_run

  include("test_game.jl")
  export test_game

  test_game(e::Experiment; args...) = test_game(e.gspec; args...)

  test_game(s::String; args...) = test_game(Examples.experiments[s]; args...)

  train(e::Experiment; args...) = UserInterface.resume!(Session(e, args...))

  train(s::String, args...) = train(Examples.experiments[s]; args...)

  explore(e::Experiment, args...) = UserInterface.explore(Session(e, args...))

  explore(s::String, args...) = explore(Examples.experiments[s]; args...)

  function play(e::Experiment, args...)
    session = Session(e, args...)
    interactive!(session.env.gspec, AlphaZeroPlayer(session), Human())
  end

  play(s::String, args...) = play(Examples.experiments[s]; args...)

end