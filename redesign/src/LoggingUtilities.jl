"""
    LoggingUtilities

This module provides utilities for logging various kinds of data, including training and
evaluation metrics, to different output formats such as files and TensorBoard.

## Initialization Functions

The following functions can be used to initialize file and tensorboard loggers:

- [`init_file_logger(logfile; overwrite=false, min_level=Logging.Info)`](@ref):
    Initializes a `FileLogger`.
- [`init_tb_logger(tb_logdir; overwrite_tb_logdir=false)`](@ref):
    Initializes a TensorBoard logger.

During training, we will most likely use multiple loggers at the same time. The following
function can be used to initialize multiple loggers at once and return them in a Dictionary:

- [`init_loggers(config; overwrite_logfiles=false, overwrite_tb_logdir=false)`](@ref):
    Initializes multiple loggers based on a given configuration.

## Logging Functions

- [`log_losses(tb_logger, value_loss, policy_loss, loss)`](@ref):
    Logs loss values using a TensorBoard logger.
- [`log_msg(file_logger, msg)`](@ref): Logs a message to a file.
- [`write_msg(file_logger, msg)`](@ref): Writes a message directly to a log file.

## Example

```julia
# Initialize loggers
config = Dict(
    "train_logfile" => "train.log",
    "eval_logfile" => "eval.log",
    "tb_logdir" => "tensorboard-logs/"
)
loggers = init_loggers(config)

# Log losses to TensorBoard
log_losses(loggers["tb"], 0.2, 0.1, 0.3)

# Log a message to a file
log_msg(loggers["train"], "Training started.")
```
"""
module LoggingUtilities

using Logging
using TensorBoardLogger

export init_loggers, log_losses, log_msg, write_msg


struct FileLogger <: AbstractLogger
    logfile::String
    min_level::Logging.LogLevel
end

function init_file_logger(logfile; overwrite=false, min_level=Logging.Info)
    if !overwrite && isfile(logfile)
        # increment name with 2 digits until a non-existing filename is found
        number = 1
        logfile = logfile[1:end-4] * "-" * lpad(number, 2, "0") * logfile[end-3:end]
        while number < 99 && isfile(logfile)
            number += 1
            logfile = logfile[1:end-7] * lpad(number, 2, "0") * logfile[end-3:end]
        end
        (number == 99) && error("Too many logfiles with the same name")
    end

    # empty file
    open(logfile, "w") do f
        write(f, "")
    end

    return FileLogger(logfile, min_level)
end

function init_tb_logger(tb_logdir; overwrite_tb_logdir=false)
    overwrite = overwrite_tb_logdir ? tb_overwrite : tb_increment
    return TBLogger(tb_logdir, overwrite, min_level=Logging.Info)
end

function init_loggers(config; overwrite_logfiles=false, overwrite_tb_logdir=false)
    loggers = Dict()

    train_logfile, eval_logfile = config.train_logfile, config.eval_logfile
    tb_logdir = config.tb_logdir

    overwrite = overwrite_logfiles
    (train_logfile != "") && (loggers["train"] = init_file_logger(train_logfile; overwrite))
    (eval_logfile != "") && (loggers["eval"] = init_file_logger(eval_logfile; overwrite))
    (tb_logdir != "") && (loggers["tb"] = init_tb_logger(tb_logdir; overwrite_tb_logdir))

    return loggers
end

function log_losses(tb_logger, value_loss, policy_loss, loss)
    with_logger(tb_logger) do
        @info "train" minibatch_value_loss=value_loss log_step_increment=0
        @info "train" minibatch_policy_loss=policy_loss log_step_increment=0
        @info "train" minibatch_loss=loss
    end
end

function log_msg(file_logger, msg)
    io = open(file_logger.logfile, "a")
    simple_logger = SimpleLogger(io, file_logger.min_level)
    with_logger(simple_logger) do
        @info msg
    end
    flush(io)
    close(io)
end

function write_msg(file_logger, msg)
    io = open(file_logger.logfile, "a")
    write(io, msg)
    flush(io)
    close(io)
end

end
