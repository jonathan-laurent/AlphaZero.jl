#####
##### JSON Serialization
#####

import JSON3

for T in [
    # Reports
    Report.Loss, Report.LearningStatus, Report.Checkpoint, Report.Epoch,
    Report.Learning, Report.Samples, Report.StageSamples, Report.Memory,
    Report.SelfPlay, Report.Perfs, Report.Iteration, Report.Initial,
    # Benchmarks
    Benchmark.DuelOutcome,
    # Network Hyperparameters
    SimpleNetHP, ResNetHP,
    # Parameters
    Params, SelfPlayParams, LearningParams, ArenaParams,
    MctsParams, MemAnalysisParams,
    # Optimisers
    CyclicNesterov, Adam,
    # Schedules
    PLSchedule, StepSchedule
  ]
  @eval JSON3.StructType(::Type{<:$T}) = JSON3.Struct()
end

# Abstract types

JSON3.StructType(::Type{OptimiserSpec}) = JSON3.AbstractType()
JSON3.subtypekey(::Type{OptimiserSpec}) = :type
JSON3.subtypes(::Type{OptimiserSpec}) =
  (nesterov=Nesterov, cyclic_nesterov=CyclicNesterov)

JSON3.StructType(::Type{<:AbstractSchedule}) = JSON3.AbstractType()
JSON3.subtypekey(::Type{<:AbstractSchedule}) = :type
JSON3.subtypes(::Type{<:AbstractSchedule}) =
  (piecewise_linear=PLSchedule, step=StepSchedule)

# TODO: for the subtypes of abstract types above, the `subtypekey` field
# is missing so deserialization won't work.
