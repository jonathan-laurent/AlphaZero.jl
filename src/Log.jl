#####
##### Logging utilities
##### Support nested logging, tables...
#####

module Log

export Logger

using Crayons
using Formatting

const INDENT_STEP = 2
const HEADER_COLORS = [crayon"bold yellow", crayon"bold"]

const TABLE_HEADER_STYLE = crayon"negative"
const TABLE_COMMENTS_STYLE = crayon"italics cyan"
const TABLE_COMMENTS_MARGIN_LENGTH = 2
const TABLE_DEFAULT_COL_WIDTH = 6
const TABLE_COL_SEP = 2
const TABLE_COMMENTS_SEP = " / "

#####
##### Basic nested logging
#####

mutable struct Logger
  indent_level :: Int
  style :: Crayon
  lastsep :: Bool
  lastrow :: Bool
  Logger() = new(0, crayon"", false)
end

indent!(l::Logger) = l.indent_level += 1

deindent(l::Logger) = l.indent_level -= 1

function print(l::Logger, args...)
  indent = INDENT_STEP * l.indent_level
  println(repeat(" ", indent), l.style, args..., crayon"reset")
  l.lastsep = false
  l.lastrow = false
end

function sep(l::Logger)
  l.lastsep || print(l, "")
  l.lastsep = true
end

function section(logger::Logger, level, args...)
  logger.indent_level = level - 1
  sep(logger)
  print(logger, HEADER_COLORS[level], args...)
  sep(logger)
  indent!(logger)
end

#####
##### Tables
#####

struct Table
  fields :: Vector{Pair{String, Function}}
  col_width :: Int
  header_style :: Crayon
  comments_style :: Crayon
  comments_margin :: String
  comments_sep :: String
  function Table(fields; col_width=TABLE_DEFAULT_COL_WIDTH)
    longest_label = maximum(length(f[1]) for f in fields)
    col_width = TABLE_COL_SEP + max(col_width, longest_label)
    comments_margin = repeat(" ", TABLE_COMMENTS_MARGIN_LENGTH)
    header_style = TABLE_HEADER_STYLE
    comments_style = TABLE_COMMENTS_STYLE
    comments_sep = TABLE_COMMENTS_SEP
    new(collect(fields), col_width, header_style,
        comments_style, comments_margin, comments_sep)
  end
end

fixed_width(str, width) = fmt(">$(width)s", first(str, width))

intersperse(sep, words) = reduce((x, y) -> x * sep * y, words)

function table_legend(l::Logger, tab::Table)
  labels = map(tab.fields) do f
    fixed_width(f[1], tab.col_width)
  end
  sep(l)
  print(l, tab.header_style, labels...)
  return
end

function table_row(l::Logger, tab::Table, fields, comments=[])
  l.lastrow || table_legend(l, tab)
  D = Dict(fields...)
  args = map(tab.fields) do f
    v = D[f[1]]
    vstr = f[2](v)
    fixed_width(vstr, tab.col_width)
  end
  if isempty(comments)
    commargs = ()
  else
    comments_str = intersperse(tab.comments_sep, comments)
    commargs = (tab.comments_margin, tab.comments_style, comments_str)
  end
  print(l, args..., commargs...)
  l.lastrow = true
  return
end

end
