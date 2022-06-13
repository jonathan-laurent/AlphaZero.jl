@reexport module Shortcuts

using RLZero

export uniform_mcts_policy

function uniform_mcts_policy(; n=100)
    return Policy(;
        num_simulations=n,
        num_considered_actions=9,
        value_scale=0.1,
        max_visit_init=50,
        oracle=uniform_oracle,
    )
end

end
