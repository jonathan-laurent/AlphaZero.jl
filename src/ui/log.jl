#####
##### Logging utilities
##### Support nested logging, tables...
#####

module Log

export Logger

using Crayons
using Formatting: fmt
import ProgressMeter

const INDENT_STEP = 2
const HEADER_COLORS = [crayon"bold yellow", crayon"bold", crayon"underline"]

const TABLE_HEADER_STYLE = crayon"negative"
const TABLE_COMMENTS_STYLE = crayon"italics cyan"
const TABLE_COMMENTS_MARGIN = 2
const TABLE_COL_SEP = 2
const TABLE_COMMENTS_SEP = " / "

# To be used externally
const NO_STYLE = crayon""
const BOLD = crayon"bold"
const RED = crayon"red"

#####
##### Basic nested logging
#####

mutable struct Logger
  console :: IO
  logfile :: IO
  indent_level :: Int
  style :: Crayon
  lastsep :: Bool
  lastrow :: Bool
  console_only :: Bool
  function Logger(console=stdout; logfile=devnull)
    new(console, logfile, 0, crayon"", false, false, false)
  end
end

indent!(l::Logger) = l.indent_level += 1

deindent(l::Logger) = l.indent_level -= 1

offset(l::Logger) = INDENT_STEP * l.indent_level

console_only!(l::Logger, v=true) = l.console_only = v

logfile(l::Logger) = l.logfile

set_logfile!(l::Logger, file) = l.logfile = file

function console_only(f, l::Logger)
  v = l.console_only
  console_only!(l, true)
  f()
  console_only!(l, v)
end

function print(l::Logger, args...)
  args = [repeat(" ", offset(l)), l.style, args..., crayon"reset"]
  Base.println(l.console, args...)
  if !l.console_only
    args_nostyle = filter(args) do x
      !isa(x, Crayon)
    end
    Base.println(l.logfile, args_nostyle...)
  end
  l.lastsep = false
  l.lastrow = false
end

function sep(l::Logger; force=false)
  (!l.lastsep || force) && print(l, "")
  l.lastsep = true
end

function section(logger::Logger, level, args...)
  logger.indent_level = level - 1
  sep(logger)
  print(logger, HEADER_COLORS[level], args...)
  sep(logger)
  indent!(logger)
end

function reset!(logger::Logger)
  logger.indent_level = 0
  logger.style = crayon""
  logger.lastsep = false
  logger.lastrow = false
end

function Progress(logger::Logger, nsteps)
  indent = repeat(" ", offset(logger))
  desc = indent * "Progress: "
  return ProgressMeter.Progress(nsteps, desc=desc, output=logger.console)
end

#####
##### Tables
#####

struct ColType
  width :: Union{Int, Nothing}
  format :: Function
end

struct Column
  name :: String
  type :: ColType
  content :: Function
end

struct Table
  columns :: Vector{Column}
  header_style :: Crayon
  comments_style :: Crayon
end

function Table(cols;
    header_style=TABLE_HEADER_STYLE,
    comments_style=TABLE_COMMENTS_STYLE)
  cols = [Column(c...) for c in cols]
  Table(cols, header_style, comments_style)
end

set_columns(tab, cols) = Table(cols, tab.header_style, tab.comments_style)

fixed_width(str, width) = fmt(">$(width)s", first(str, width))

intersperse(sep, words) = reduce((x, y) -> x * sep * y, words)

function table_legend(l::Logger, tab::Table)
  labels = map(enumerate(tab.columns)) do (i, col)
    w = col.type.width
    i > 1 && (w += TABLE_COL_SEP)
    fixed_width(col.name, w)
  end
  print(l, tab.header_style, labels...)
  return
end

function table_row(l::Logger, tab::Table, obj, comments=[]; style=nothing)
  l.lastrow || table_legend(l, tab)
  args = map(enumerate(tab.columns)) do (i, col)
    v = col.content(obj)
    vstr = col.type.format(v)
    @assert !isnothing(col.type.width) "Column widths must be specified"
    w = col.type.width
    i > 1 && (w += TABLE_COL_SEP)
    fixed_width(vstr, w)
  end
  if isempty(comments)
    commargs = ()
  else
    comments_str = intersperse(TABLE_COMMENTS_SEP, comments)
    margin = repeat(" ", TABLE_COMMENTS_MARGIN)
    commargs = (margin, tab.comments_style, comments_str)
  end
  isnothing(style) || (args = (style, args..., crayon"reset"))
  print(l, args..., commargs...)
  l.lastrow = true
  return
end

# Add two features: automatic tuning of column width and
# handling of nothing values
function table(l::Logger, tab::Table, objs; comments=nothing, styles=nothing)
  cols = map(tab.columns) do col
    f = x -> isnothing(x) ? "" : col.type.format(x)
    width = maximum(length(f(col.content(x))) for x in objs)
    width = max(width, length(col.name))
    coltype = ColType(width, f)
    Column(col.name, coltype, col.content)
  end
  cols = filter(cols) do col
    any(!isnothing(col.content(x)) for x in objs)
  end
  tab = set_columns(tab, cols)
  for (i, x) in enumerate(objs)
    c = isnothing(comments) ? [] : comments[i]
    s = isnothing(styles) ? nothing : styles[i]
    table_row(l, tab, x, c, style=s)
  end
end

end
