import OpenSpiel
using ..AlphaZero

spiel_game = OpenSpiel.load_game("tic_tac_toe")

function vectorize_state(spec, state)
  if !OpenSpiel.is_terminal(state)
    obsten = OpenSpiel.observation_tensor(state)
    empt =  obsten[1:9]
    black = obsten[10:18]
    white = obsten[19:27]
    # in AlphaZero, oracle always see game from current player's point of view
    if OpenSpiel.current_player(state) == 1
      white, black = black, white
    end
    ans = convert(Array{Float32}, vcat(empt, white, black))
    
    return reshape(ans, 3,3,3)
  else
    return cat(
      zeros(Float32, 3,3),
      ones(Float32,3,3),
      ones(Float32,3,3);
      dims=3
    )
  end
end

#####
##### Interaction API (from AlphaZero's tictactoe)
#####

function action_string(spec, a)
  string(Char(Int('A') + a))
end

function parse_action(spec, str)
  length(str) == 1 || (return nothing)
  x = Int(uppercase(str[1])) - Int('A')
  (0 <= x < 9) ? x : nothing
end


GameSpec() = AlphaZero.OpenSpielWrapper.Spec(
  spiel_game;
  vectorize_state,
  action_string,
  parse_action,
  suppress_warnings=true
)