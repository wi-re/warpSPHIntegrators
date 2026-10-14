# Wave 6: extract every ConstantCache field at Float64 so the Python side
# can embed exact doubles with zero transcription risk.
using OrdinaryDiffEqLowOrderRK
using OrdinaryDiffEqTsit5

LO = OrdinaryDiffEqLowOrderRK
TS = OrdinaryDiffEqTsit5

low_order_names = [
    "OwrenZen3ConstantCache",
    "OwrenZen4ConstantCache",
    "OwrenZen5ConstantCache",
    "BS5ConstantCache",
    "Anas5ConstantCache",
    "PSRK4p7q6ConstantCache",
    "PSRK3p6q5ConstantCache",
    "PSRK3p5q4ConstantCache",
    "MSRK5ConstantCache",
    "MSRK6ConstantCache",
    "Stepanov5ConstantCache",
    "SIR54ConstantCache",
    "Alshina2ConstantCache",
    "Alshina3ConstantCache",
    "Alshina6ConstantCache",
    "Ralston4ConstantCache",
    "RKO65ConstantCache",
    "FRK65ConstantCache",
    "RKMConstantCache",
]
tsit_names = ["Tsit5ConstantCacheActual"]

function dump_cache(mod, name)
    S = getfield(mod, Symbol(name))
    cc = S(Float64, Float64)
    println("### $name")
    for f in fieldnames(S)
        v = getfield(cc, f)
        if v isa AbstractArray
            vals = join(repr(x) for x in v)
            println("$f = [$vals]")
        else
            println("$f = $(repr(v))")
        end
    end
end

for n in low_order_names
    try
        dump_cache(LO, n)
    catch e
        println("### $n ERROR: $(sprint(showerror, e))")
    end
end
for n in tsit_names
    try
        dump_cache(TS, n)
    catch e
        println("### $n ERROR: $(sprint(showerror, e))")
    end
end
println("EXTRACTION DONE")
