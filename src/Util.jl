################################################################################
# Generic utilities
################################################################################

module Util

################################################################################

module Records

"""
    generate_named_constructors(T)

generates constructors for record type `T` that are reminiscent of the
record constructors in functional programming languages such as Ocaml or
Haskell. For example, given the following structure definition,

    struct Point
      x :: Float64
      y :: Float64
    end

then

    generate_named_constructors(Point)

generates the following code:

    Point(;x,y) = Point(x,y)
    Point(p; x=p.x, y=p.y) = Point(x,y)

"""
function generate_named_constructors(T)
  fields = fieldnames(T)
  Tname = Symbol(split(string(T), ".")[end])
  base = :_old_
  @assert base ∉ fields
  fields_withdef = [Expr(:kw, f, :($base.$f)) for f in fields]
  quote
    $Tname(;$(fields...)) = $Tname($(fields...))
    #$Tname($base::$Tname; $(fields_withdef...)) = $Tname($(fields...))
  end
end

end

################################################################################

end

################################################################################
