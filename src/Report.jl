#####
##### Analytical reports for debugging and hyperparameters tuning
#####

module Report

struct Loss
  L :: Float64
  Lp :: Float64
  Lv :: Float64
  Lreg :: Float64
  Hp :: Float64
end

struct Network
  maxw  :: Float64
  meanw :: Float64
  pbiases :: Vector{Float64}
  vbias :: Float64
end

struct Game
  reward :: Float64
  length :: Int
end

struct Evaluation
  games :: Vector{Game}
  average_reward :: Float64 # redundant
end

struct Checkpoint
  epoch_id :: Int
  time_eval :: Float64
  eval :: Evaluation
end

struct Epoch
  time_train :: Float64
  time_loss :: Float64
  loss_after :: Loss
  network_after :: Network
end

struct Learning
  time_convert :: Float64
  init_loss :: Loss
  epochs :: Vector{Epoch}
  checkpoints :: Vector{Checkpoint}
  nn_replaced :: Bool
end

struct SelfPlay
  games :: Vector{Game}
end

struct Iteration
  time_self_play :: Float64
  time_learning :: Float64
  self_play :: SelfPlay
  learning :: Learning
end

struct Training
  num_nn_params :: Int
  iterations :: Vector{Iteration}
end

end
