module UtilTests

using ...Util

export run_util_tests

function run_util_tests()
    Util.StaticBitArrays.run_tests()
    return nothing
end

end
