#####
## Utilities for profiling
#####

# Chrome tracing reference:
# - https://www.gamedeveloper.com/programming/in-depth-using-chrome-tracing-to-view-your-inline-profiling-data
# - https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview

"""
Export some profiling info to be visualized with chrome://tracing.
"""
module ProfUtils

import JSON3
using LoggingExtras: FormatLogger, EarlyFilteredLogger, TeeLogger, global_logger

const LOG_GROUP = :chrome_tracing

function event(; name::String, cat::String, typ::String, pid::Int, tid::Int)
  @assert typ in ["B", "E", "i"]
  ts = Int(time_ns()) ÷ 1000
  msg = name
  typ == "B" && (msg *= " (started)")
  typ == "E" && (msg *= " (ended)")
  args = (; name, cat, ph=typ, ts, pid, tid)
  @debug msg _group=LOG_GROUP args...
end

event_start(; name::String, cat::String, pid::Int, tid::Int) =
  event(; name, cat, typ="B", pid, tid)

event_end(; name::String, cat::String, pid::Int, tid::Int) =
  event(; name, cat, typ="E", pid, tid)

instant_event(; name::String, cat::String, pid::Int, tid::Int) =
  event(; name, cat, typ="i", pid, tid)

function log_event(f; name::String, cat::String, pid::Int, tid::Int)
  event_start(; name, cat, pid, tid)
  res = f()
  event_end(; name, cat, pid, tid)
  return res
end

function chrome_tracing_logger(file::String; always_flush=false)
  stream = open(file, "w")
  println(stream, "[")
  formatter = FormatLogger(stream; always_flush) do stream, log
    args = values(log.kwargs)
    dict = (; args.name, args.cat, args.ph, args.ts, args.pid, args.tid)
    JSON3.write(stream, dict)
    println(stream, ",")
  end
  return EarlyFilteredLogger(formatter) do log
    return log.group == LOG_GROUP
  end
end

function set_chrome_tracing(file::String; always_flush=false)
  tracing_logger = chrome_tracing_logger(file; always_flush)
  logger = TeeLogger(global_logger(), tracing_logger)
  global_logger(logger)
  return
end

end
