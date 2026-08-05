#!/usr/bin/env julia
# Process raw borehole geodatabase data into GeoJSON files.
# Usage: julia --project=. scripts/process_data.jl

using GeothermalViz

gdb_path = joinpath(@__DIR__, "..", "data", "Grunnvannsborehull.gdb")
output_dir = joinpath(@__DIR__, "..", "processed_data")

println("Processing geodatabase: $gdb_path")
println("Output directory: $output_dir")
println()

output_files = process_geodatabase(gdb_path; output_dir=output_dir)

println()
println("Done! Processed files:")
for (name, path) in sort(collect(output_files))
    size_kb = round(filesize(path) / 1024; digits=1)
    println("  $name → $path ($size_kb KB)")
end
