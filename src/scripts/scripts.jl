module Scripts

  using ..AlphaZero
  using ..AlphaZero.UserInterface: Session, resume!

  include("dummy_run.jl")
  export dummy_run

  dummy_run(s::String; args...) = dummy_run(Examples.experiments[s]; args...)

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
    if GI.two_players(e.gspec)
      interactive!(session.env.gspec, AlphaZeroPlayer(session), Human())
    else
      interactive!(session.env.gspec, Human())
    end
  end

  play(s::String, args...) = play(Examples.experiments[s]; args...)

end