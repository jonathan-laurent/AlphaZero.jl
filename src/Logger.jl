#####
##### Logger
#####

using Crayons
using Formatting

const INDENT_STEP = 2
const HEADER_COLORS = [crayon"bold yellow", crayon"bold"]
const TAB_HEADER_STYLE = crayon"negative"
const TAB_COMMENT_STYLE = crayon"italics cyan"
const COMMENT_SEP_LENGTH = 4

mutable struct Logger
  indent_level :: Int
  style :: Crayon
  lastsep :: Bool
  lastrow :: Bool
  Logger() = new(0, crayon"", false)
end

indent!(l::Logger) = l.indent_level += 1

deindent(l::Logger) = l.indent_level -= 1

function Base.print(l::Logger, args...)
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

struct Table
  fields :: Vector{Pair{String, String}}
  col_width :: Int
  header_style :: Crayon
  comment_style :: Crayon
  comment_sep :: String
  function Table(fields; col_width=8)
    col_width = max(col_width, 1 + maximum(length(f[1]) for f in fields))
    header_style = TAB_HEADER_STYLE
    comment_style = TAB_COMMENT_STYLE
    comment_sep = repeat(" ", COMMENT_SEP_LENGTH)
    new(collect(fields), col_width, header_style, comment_style, comment_sep)
  end
end

fixed_width(str, width) = fmt(">$(width)s", first(str, width))

function print_table_legend(l::Logger, tab::Table)
  labels = map(tab.fields) do f
    fixed_width(f[1], tab.col_width)
  end
  sep(l)
  print(l, tab.header_style, labels...)
  return
end

function print_table_row(l::Logger, tab::Table, fields, comment=nothing)
  l.lastrow || print_table_legend(l, tab)
  D = Dict(fields...)
  args = map(tab.fields) do f
    v = D[f[1]]
    vstr = fmt(f[2], v)
    fixed_width(vstr, tab.col_width)
  end
  if isnothing(comment)
    commargs = ()
  else
    commargs = (tab.comment_sep, tab.comment_style, comment)
  end
  print(l, args..., commargs...)
  l.lastrow = true
  return
end

#=
table = Table("Loss" => ".4f", "Lp" => ".4f")
logger = Logger()
print_table_row(logger, table, ("Loss" => 0.34, "Lp" => 0.23))
print_table_row(logger, table, ("Loss" => 0.12, "Lp" => 0.00), "Hello")
print_table_row(logger, table, ("Loss" => 0.12, "Lp" => 0.00))
print_table_row(logger, table, ("Loss" => 0.12, "Lp" => 0.00), "Hello")
=#
