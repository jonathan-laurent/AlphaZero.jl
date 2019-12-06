#####
##### Validate against rollouts
#####

struct ValidationReport
  z :: Float64
  games :: Vector{Float64}
  time :: Float64
end

abstract type Validation end

@kwdef struct RolloutsValidation <: Validation
  num_games :: Int
  reset_mcts_every :: Int
  baseline :: MctsParams
  contender :: MctsParams
end

Base.length(v::RolloutsValidation) = v.num_games

function validation_score(env::Env{G}, v::RolloutsValidation, progress) where G
  baseline = MctsPlayer(MCTS.RolloutOracle{G}(), v.baseline)
  contender = MctsPlayer(env.bestnn, v.contender)
  let games = Vector{Float64}(undef, v.num_games)
    avg, time = @timed begin
      pit(baseline, contender, v.num_games, v.reset_mcts_every) do i, z
        games[i] = z
        next!(progress)
      end
    end
    return ValidationReport(avg, games, time)
  end
end
