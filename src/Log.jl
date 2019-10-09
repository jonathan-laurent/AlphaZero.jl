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
const TABLE_COMMENTS_MARGIN = 4
const TABLE_COL_SEP = 2
const TABLE_COMMENTS_SEP = " / "

# To be used externally
const BOLD = crayon"bold"

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

struct ColType
  width :: Int
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
  function Table(cols...;
      header_style=TABLE_HEADER_STYLE,
      comments_style=TABLE_COMMENTS_STYLE)
    cols = [Column(c...) for c in cols]
    new(cols, header_style, comments_style)
  end
end

fixed_width(str, width) = fmt(">$(width)s", first(str, width))

intersperse(sep, words) = reduce((x, y) -> x * sep * y, words)

function table_legend(l::Logger, tab::Table)
  labels = map(tab.columns) do col
    fixed_width(col.name, col.type.width + TABLE_COL_SEP)
  end
  sep(l)
  print(l, tab.header_style, labels...)
  return
end

function table_row(l::Logger, tab::Table, obj, comments=[]; style=nothing)
  l.lastrow || table_legend(l, tab)
  args = map(tab.columns) do col
    v = col.content(obj)
    vstr = col.type.format(v)
    w = col.type.width + TABLE_COL_SEP
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

end
