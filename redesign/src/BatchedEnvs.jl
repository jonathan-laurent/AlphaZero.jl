"""
Interface for batchable environements that can be run on the GPU.
"""
module BatchedEnvs

export num_actions, valid_action, act, terminated, make_image

function num_actions end

function valid_action end

function act end

function terminated end

function make_image end

end
