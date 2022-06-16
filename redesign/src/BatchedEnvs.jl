"""
Interface for batchable environements that can be run on the GPU.
"""
module BatchedEnvs

function num_actions end

function valid_action end

function act end

function terminated end

end
