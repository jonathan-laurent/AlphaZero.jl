################################################################################
# Game Interface
################################################################################

"""
A generic interface for zero-sum, symmetric games.

  + `white_playing(::State) :: Bool`
  + `white_reward(::State) :: Union{Nothing, Float64}`
  + `board(::State)`
  + `board_symmetric(::State)`
  + `available_actions(::State) :: Vector{Action}`
  + `play!(::State, ::Action)`
  + `undo!(::State, ::Action)`

Actions must be symmetric in the sense that they do not depend on the current
player (they are expressed in relative terms).
We expect the following to hold:
  available_actions(s) =
    available_actions(State(board_symmetric(s), player=symmetric(s.curplayer)))
"""

module GameInterface

  function Board end
  function Action end

  # Game functions
  function white_playing end
  function white_reward end
  function board end
  function board_symmetric end
  function available_actions end
  function play! end
  function undo! end

  # Interface with neural networks
  function board_dim end
  function vectorize_board end
  function num_actions end
  function action end
  function action_id end

  function actions_mask(G, available_actions) # Derived function
    nactions = num_actions(G)
    mask = falses(nactions)
    for a in available_actions
      mask[action_id(G, a)] = true
    end
    return mask
  end

  function canonical_board(state)
    white_playing(state) ? board(state) : board_symmetric(state)
  end

end

################################################################################
