#####
##### Validate against rollouts
#####

@kwdef struct RolloutsValidation
  num_games :: Int
  baseline :: MctsParams
  contender :: MctsParams
end

struct ValidationReport
  z :: Float64
  games :: Vector{Float64}
end

function validation_score(env::Env{G}, v::RolloutsValidation, progress) where G
  baseline = MctsPlayer(MCTS.RolloutOracle{G}(), v.baseline)
  contender = MctsPlayer(env.bestnn, v.contender)
  let games = Vector{Float64}(undef, v.num_games)
    avg = pit(baseline, contender, v.num_games) do i, z
      games[i] = z
      next!(progress)
    end
    return ValidationReport(avg, games)
  end
end
