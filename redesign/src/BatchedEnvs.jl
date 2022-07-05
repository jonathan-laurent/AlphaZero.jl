"""
Interface for batchable environements that can be run on the GPU.
"""
module BatchedEnvs

export num_actions, valid_actions, act, terminated

function num_actions end

function valid_actions end

function act end

function terminated end

end
