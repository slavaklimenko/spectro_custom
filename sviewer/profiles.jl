using DataStructures
using Interpolations
using ImageFiltering
using LsqFit
using PeriodicTable
using Polynomials
using Roots
using SpecialFunctions
using DSP

function argclose2(x, value)
    return argmin(abs.(x .- value))
end

function binsearch(x, item; type="close")

    if item <= x[1]
        return 1
    elseif item >= x[end]
        return size(x)[1]
    else
        first = 1
        last = length(x)

        while first < last - 1
            middle = fld(first+last, 2)
            if item > x[middle]
                first = middle
            else
                last = middle
            end
        end

        if type == "min"
            return first
        elseif type == "max"
            return last
        elseif type == "close"
            if item - x[first] < x[last] - item
                return first
            else
                return last
            end
        end
    end
end

function Dawson(x)
    y = x .* x
    return x .* (1 .+ y .* (0.1107817784 .+ y .* (0.0437734184 .+ y .* (0.0049750952 .+ y .* 0.0015481656)))) / (1 .+ y .* (0.7783701713 .+ y .* (0.2924513912 .+ y .* (0.0756152146 .+ y .* (0.0084730365 .+ 2 .*  0.0015481656 .* y)))))
end

function Voigt(a, x)
    return exp.(-1 .* x .* x) .* (1 .+ a^2 .* (1 .- 2 .* x .* x)) - 2 / sqrt(π) .* (1 .- 2 .* x .* Dawson.(x))
end

function voigt_range_old(a, level)
    f(x) = real(SpecialFunctions.erfcx(a - im * x)) - level
    return find_zero(f, (min(sqrt(a / level / sqrt(π)), sqrt(max(0, -log(level)))) / 2, 3*max(sqrt(a / level / sqrt(π)), sqrt(max(0, -log(level))))), tol=0.001)
end

function voigt_range(a, level)
    if level < 1
        α = 1.3
        if exp(sqrt(a / sqrt(π) / level)) ^ α == Inf
            return sqrt(a / sqrt(π) / level)
        else
            return log((exp(sqrt(-log(level))) ^ α + exp(sqrt(a / sqrt(π) / level)) ^ α - 1) ^ (1 / α))
        end
    elseif level == 1
        return 0
    else
        return 0
    end
end

function voigt_deriv(x, a, tau_0)
    w = SpecialFunctions.erfcx.(a .- im .* x)
    return exp.( - tau_0 .* real(w)) .* tau_0 .* 2 .* (imag(w) .* a .- real(w) .* x)
end

function voigt_step(a, tau_0; level=0.002, step=0.03)
    x_0 = voigt_max_deriv(a, tau_0)
    x = [-abs(x_0)]
    w = real(SpecialFunctions.erfcx.(a .- im .* x[1])) * tau_0
    der = Vector{Float64}()
    while x[end] < 0
        append!(der, voigt_deriv(x[end], a, tau_0))
        append!(x, x[end] + step / der[end])
    end
    deleteat!(x, size(x))
    t = real(SpecialFunctions.erfcx.(a .- im .* x[1])) * tau_0
    while t > level
        pushfirst!(x, x[1] - step / der[1])
        w = SpecialFunctions.erfcx.(a .- im .* x[1])
        t = tau_0 .* real(w)
        pushfirst!(der, exp.( - t) .* tau_0 .* 2 .* (imag(w) .* a .- real(w) .* x[1]))
    end
    append!(x, -x[end:-1:1])
    append!(der, der[end:-1:1])
    return x[2:end-1], der
end

function df(x, a, tau_0)
    v = SpecialFunctions.erfcx(a - im * x)
    return tau_0 * (imag(v) * a - real(v) * x)^2 + real(v) * (a^2 - x^2 + 0.5) + 2 * imag(v) * a * x - a / sqrt(π)
end

function voigt_max_deriv(a, tau_0)
    f = (x -> df(x, a, tau_0))
    level = tau_0 > 0.3 ? 0.1 / tau_0 : 1 / 3
    try
        r = find_zero(f, (0, abs(voigt_range(a, level))))
        return r
    catch
        r = find_zero(f, abs(voigt_range(a, level) / 2))
        return r
    end
end

function voigt_grid(l, a, tau_0; step=0.03)
    x, r = voigt_step(a, tau_0)
    k_min, k_max = binsearch(l, x[1], type="min"), binsearch(l, x[end], type="max")-1
    g = Vector{Float64}()
    for k in k_min:k_max
        i_min, i_max = binsearch(x, l[k]), binsearch(x, l[k+1])
        #append!(g, maximum(r[i_min:i_max]))
        append!(g, Int(floor((l[k+1] - l[k]) / (step / maximum(r[i_min:i_max]))))+1)
    end
    return k_min:k_max, g
end

function z_to_v(;z=nothing, v=nothing, z_ref=0)
    c = 299792.458
    if v == nothing
        return c * (z - z_ref) / (1 + z_ref)
    elseif z == nothing
        return z_ref + v / c * (1 + z_ref)
    end
end
##############################################################################
##############################################################################
##############################################################################

mutable struct par
    name::String
    val::Float64
    min::Float64
    max::Float64
    step::Float64
    vary::Bool
    addinfo::String
    tied::String
    ref
end

function make_pars(p_pars; tieds=Dict(), z_ref=nothing)
    pars = OrderedDict{String, par}()
    for p in p_pars
        if occursin("z_", p.__str__()) * (z_ref == true)
            pars[p.__str__()] = par(p.__str__(), 0.0, z_to_v(z=p.min, z_ref=p.val), z_to_v(z=p.max, z_ref=p.val), p.step, p.fit * p.vary, p.addinfo, "", p.val)
        else
            pars[p.__str__()] = par(p.__str__(), p.val, p.min, p.max, p.step, p.fit * p.vary, p.addinfo, "", nothing)
        end
        if occursin("cf", p.__str__())
            pars[p.__str__()].min, pars[p.__str__()].max = 0, 1
        end

    end
    for (k, v) in tieds
        pars[k].vary = false
        pars[k].tied = v
    end
    return pars
end

function get_element_name(name)
    st = name
    if occursin("j", st)
        st = st[1:findfirst("j", st)[1]-1]
    end
    for s in ["I", "V", "X", "*"]
        st = replace(st, s => "")
    end
    return st
end

function doppler(name, turb, kin)
    name = get_element_name(name)
    if name == "D"
        mass = 2
    else
        for e in elements
            if e.symbol == name
                mass = e.atomic_mass.val
            end
        end
        #mass = element(get_element_name(name)).atomic_mass.val
    end
    #println(mass)
    #println(element(get_element_name(name)).atomic_mass.val)
    return (turb ^ 2 + 0.0164 * kin / mass) ^ .5
end

function abundance(name, logN_ref, me)
    d = Dict([
    ("H", [12, 0, 0]),
    ("D", [7.4, 0, 0]), # this is not in Asplund
    ("He", [10.93, 0.01, 0.01]), #be carefull see Asplund 2009
    ("Li", [1.05, 0.10, 0.10]),
    ("Be", [1.38, 0.09, 0.09]),
    ("B", [2.70, 0.20, 0.20]),
    ("C", [8.43, 0.05, 0.05]),
    ("N", [7.83, 0.05, 0.05]),
    ("O", [8.69, 0.05, 0.05]),
    ("F", [4.56, 0.30, 0.30]),
    ("Ne", [7.93, 0.10, 0.10]),  #be carefull see Asplund 2009
    ("Na", [6.24, 0.04, 0.04]),
    ("Mg", [7.60, 0.04, 0.04]),
    ("Al", [6.45, 0.03, 0.03]),
    ("Si", [7.51, 0.03, 0.03]),
    ("P", [5.41, 0.03, 0.03]),
    ("S", [7.12, 0.03, 0.03]),
    ("Cl", [5.50, 0.30, 0.30]),
    ("Ar", [6.40, 0.13, 0.13]),  #be carefull see Asplund 2009
    ("K", [5.03, 0.09, 0.09]),
    ("Ca", [6.34, 0.04, 0.04]),
    ("Sc", [3.15, 0.04, 0.04]),
    ("Ti", [4.95, 0.05, 0.05]),
    ("V", [3.93, 0.08, 0.08]),
    ("Cr", [5.64, 0.04, 0.04]),
    ("Mn", [5.43, 0.04, 0.04]),
    ("Fe", [7.50, 0.04, 0.04]),
    ("Co", [4.99, 0.07, 0.07]),
    ("Ni", [6.22, 0.04, 0.04]),
    ("Cu", [4.19, 0.04, 0.04]),
    ("Zn", [4.56, 0.05, 0.05]),
    ("Ga", [3.04, 0.09, 0.09]),
    ("Ge", [3.65, 0.10, 0.10]),
    ])
    return logN_ref - (12 - d[get_element_name(name)][1]) + me
end

function update_pars(pars, spec, add)
    for (k, v) in pars
        if v.tied != ""
            pars[k].val = pars[v.tied].val
        end
        if occursin("res", pars[k].name)
            #println(pars[k].name, " ", pars[k].val, " ", parse(Int, pars[k].addinfo[5:end]))
            spec[parse(Int, pars[k].addinfo[5:end]) + 1].resolution = pars[k].val
        end
        if occursin("disps", pars[k].name)
            spec[parse(Int, split(pars[k].name, "_")[2]) + 1].disps = pars[k].val
        end
        if occursin("dispz", pars[k].name)
            spec[parse(Int, split(pars[k].name, "_")[2]) + 1].dispz = pars[k].val
        end
        if occursin("Ntot", pars[k].name)
            ind = split(pars[k].name, "_")[2]
            pr = add["pyratio"][parse(Int, ind) + 1]
            x = pyratio_predict(pr, pars)
            col = [v.val - log10(sum(x)) + log10(x[i]) for i in 1:pr.num]
            for (k1, v1) in pars
                if startswith(k1, "N_" * ind * "_" * pr.species) * occursin("Ntot", v1.addinfo)
                    i = occursin(pr.species * "j", k1) ? parse(Int, replace(k1, "N_" * ind * "_" * pr.species * "j" => "")) + 1 : 1
                    pars[k1].val = col[i]
                end
            end
        end
        if occursin("dtoh", pars[k].name)
            for (k1, v1) in pars
                if occursin("N_", k1) * occursin("DI", k1) * occursin("DtoH", v1.addinfo)
                    pars[k1].val = pars[replace(k1, "DI" => "HI")].val + pars[k].val
                end
            end
        end
        if occursin("me", pars[k].name)
            for (k1, v1) in pars
                if occursin(pars[k].name, v1.addinfo)
                    pars[k1].val = abundance(split(k1, "_")[3], pars[replace(k1, split(k1, "_")[3] => "HI")].val, pars[k].val)
                end
            end
        end
    end
end

mutable struct prior
    name::String
    val::Float64
    plus::Float64
    minus::Float64
end

function make_priors(p_priors)
    priors = OrderedDict{String, prior}()
    for (k, p) in p_priors
        priors[k] = prior(k, p.val, p.plus, p.minus)
    end
    return priors
end

function use_prior(prior, val)
    a = val - prior.val
    if a > 0
        return 0.5 * (a / prior.plus) ^ 2
    else
        return 0.5 * (a / prior.minus) ^ 2
    end
end

##############################################################################

mutable struct line
    name::String
    sys::Int64
    lam::Float64
    f::Float64
    g::Float64
    b::Float64
    logN::Float64
    z::Float64
    l::Float64
    tau0::Float64
    a::Float64
    ld::Float64
    dx::Float64
    cf::Int64
    stack::Int64
    stv::Dict{}
end

function update_lines(lines, pars; ind=0)
    mask = Vector{Bool}(undef, 0)
    for line in lines
        if pars["b_" * string(line.sys) * "_" * line.name].addinfo == "consist"
            line.b = doppler(line.name, pars["turb_" * string(line.sys)].val, pars["kin_" * string(line.sys)].val)
        elseif pars["b_" * string(line.sys) * "_" * line.name].addinfo != ""
            line.b = pars["b_" * string(line.sys) * "_" * pars["b_" * string(line.sys) * "_" * line.name].addinfo].val
        else
            line.b = pars["b_" * string(line.sys) * "_" * line.name].val
        end
        line.logN = pars["N_" * string(line.sys) * "_" * line.name].val
        if pars["z_" * string(line.sys)].ref == nothing
            line.z = pars["z_" * string(line.sys)].val
        else
            line.z = z_to_v(v=pars["z_" * string(line.sys)].val, z_ref=pars["z_" * string(line.sys)].ref)
        end
        line.l = line.lam * (1 + line.z)
        line.tau0 = sqrt(π) * 0.008447972556327578 * (line.lam * 1e-8) * line.f * 10 ^ line.logN / (line.b * 1e5)
        line.a = line.g / 4 / π / line.b / 1e5 * line.lam * 1e-8
        line.ld = line.lam * line.b / 299794.26 * (1 + line.z)
        if line.stack > -1
            line.stv = Dict([s => pars[s * "_" * string(line.stack)].val for s in ["sts", "stNl", "stNu"]])
        end
        append!(mask, (ind == 0) || (line.sys == ind-1))
    end
    return mask
end

function prepare_lines(lines)
    fit_lines = Vector{line}(undef, size(lines)[1])
    for (i, l) in enumerate(lines)
        fit_lines[i] = line(l.name, l.sys, l.l(), l.f(), l.g(), l.b, l.logN, l.z, l.l()*(1+l.z), 0, 0, 0, 0, l.cf, l.stack, Dict())
    end
    return fit_lines
end

function prepare_cheb(pars, ind)
    cont = []
    d = [[parse(Int, split(k, "_")[2]), parse(Int, split(k, "_")[3]), parse(Int, split(v.addinfo, "_")[3]), v] for (k, v) in pars if occursin("cont_", k)]
    if length(d) > 0
        d = permutedims(reshape(hcat(d...), (length(d[1]), length(d))))
        for k in unique(d[:, 1][d[:, 3] .== ind - 1])
            append!(cont, [cheb([], 0, 0, 0)])
            for i in sort(d[:, 2][(d[:, 1] .== k) .& (d[:, 3] .== ind - 1)])
                p = d[:, 4][(d[:, 1] .== k) .& (d[:, 2] .== i) .& (d[:, 3] .== ind - 1)]
                if i == 0
                    cont[end].left, cont[end].right, cont[end].disp = parse(Float64, split(split(p[1].addinfo, "_")[1], "..")[1]), parse(Float64, split(split(p[1].addinfo, "_")[1], "..")[2]), parse(Float64, split(p[1].addinfo, "_")[4])
                end
                append!(cont[end].c, [p[1].name])
            end
        end
    end
    return cont
end

function prepare_coll(pr, s)
    #println(keys(pr.species[s].coll))
    c = Dict()
    for sp in keys(pr.species[s].coll)
        #println(sp)
        c[sp] = Array{coll}(undef, pr.species[s].num, pr.species[s].num)
        #println(pr.species[s].num, " ", pr.species[s].coll[sp].c[1].rates)
        for i in 1:pr.species[s].num
            for j in 1:pr.species[s].num
                if i != j
                    if pr.species[s].coll[sp].c[1].rates == nothing
                         coll(i, j, LinearInterpolation([0, 6], [0, 0], extrapolation_bc=Flat()))
                    else
                        #println(pr.species[s].coll[sp].rate(i-1, j-1, 2), " ", pr.species[s].coll[sp].rate(j-1, i-1, 2))
                        #println(pr.species[s].coll[sp].c[1].rates[1,:], " ", pr.species[s].coll[sp].rate(i-1, j-1, pr.species[s].coll[sp].c[1].rates[1,:]))
                        c[sp][i,j] = coll(i, j, LinearInterpolation(pr.species[s].coll[sp].c[1].rates[1,:], pr.species[s].coll[sp].rate(i-1, j-1, pr.species[s].coll[sp].c[1].rates[1,:]), extrapolation_bc=Flat()))
                    end
                end
            end
        end
        #for col in pr.species[s].coll[sp].c
        #    println(col.rate(1, 0, 2), " ", col.rate(0, 1, 2))
        #end
    end
    return c
end

function prepare_add(fit, pars)
    add = Dict()
    if any(occursin("Ntot", k) for k in keys(pars))
        add["pyratio"] = Dict()
    end
    for (k, v) in pars
        if occursin("Ntot", pars[k].name)
            ind = split(pars[k].name, "_")[2]
            #add["pyratio"][parse(Int, split(pars[k].name, "_")[2]) + 1] = fit.sys[parse(Int, split(pars[k].name, "_")[2])+1].pr
            pr = fit.sys[parse(Int, ind)+1].pr
            s = collect(keys(pr.species))[1]
            #println(prepare_coll(pr, s))
            #println(ind)
            #println(collect(keys(pr.pars)))
            #println(keys(pr.species))
            #println(pr.balance(debug="A"))
            #println(pr.balance(debug="C"))
            #println(pr.balance(debug="IR"))
            #println(pr.balance(debug="UV"))
            add["pyratio"][parse(Int, ind) + 1] = pyratio(ind, collect(keys(pr.pars)), s, pr.species[s].num, pr.species[s].Eij, pr.species[s].Aij, pr.species[s].Bij, pr.balance(debug="C"), pr.species[s].rad_rate, pr.species[s].pump_rate, prepare_coll(pr, s))
        end
    end
    #println(add)
    return add
end

##############################################################################
function pyratio_predict(pr, pars)
    #pars["logT_" * ind].val
    W = pr.Aij

    update_coll_rate(pr, pars)
    if any(s in pr.pars for s in ["n", "e"])
        W = W .+ pr.coll_rate
    end

    if "CMB" in pr.pars
        TCMB = pars["CMB_" * pr.ind].val
    else
        TCMB = 2.726 * (pars["z_" * pr.ind].val + 1)
    end
    W = W .+ cmb_rate(pr.Bij, pr.Eij, TCMB)

    if "rad" in pr.pars
        W = W .+ pr.rad_rate .* 10 ^ pars["rad_" * pr.ind].val
    end
    if "rad" in pr.pars
        W = W .+ pr.pump_rate .* 10 ^ pars["rad_" * pr.ind].val
    end
    #println(W)
    K = copy(transpose(W))
    for i = 1:size(W)[1]
        K[i, i] -= sum(W, dims=2)[i]
    end
    #println(K)
    #println(K[2:end, 2:end])
    #println(K[2:end, 1])
    #println(K[2:end, 2:end] \ (-1 .* K[2:end, 1]))
    return insert!(abs.(K[2:end, 2:end] \ (-1 .* K[2:end, 1])), 1, 1)
end

function cmb_rate(Bij, Eij, TCMB)
    return Bij .* 8 .* π .* 6.62607015e-27 .* Eij .^ 3 ./ (exp.(Eij .* 6.62607015e-27 ./ 1.380649e-16 .* 2.99792458e10 ./ TCMB) .- 1 .+ 1.66533e-16)
end

function update_coll_rate(pr, pars)
    f_He = 0 #0.08
    for i in 1:pr.num
        for j in 1:pr.num
            pr.coll_rate[i, j] = 0
            if i != j
                for p in pr.pars
                    if p in ["e", "H"]
                        pr.coll_rate[i, j] += 10 ^ (pars["logn_" * pr.ind].val) * pr.coll[p][i, j].rate(pars["logT_" * pr.ind].val)
                    elseif p in ["n"]
                        m_fr = "f" in pr.pars ? 10 ^ pars["logf_" * pr.ind].val : mol_fr
                        f_HI, f_H2 = (1 - m_fr) / (f_He + 1 - m_fr / 2), m_fr / 2 / (f_He + 1 - m_fr / 2)
                        pr.coll_rate[i, j] += 10 ^ (pars["logn_" * pr.ind].val) * (pr.coll["H"][i, j].rate(pars["logT_" * pr.ind].val)) * f_HI
                        otop = 9 * exp(-170.6 / 10 ^ pars["logT_" * pr.ind].val)
                        #println(m_fr, " ", f_HI, " ", f_H2, " ", otop)
                        pr.coll_rate[i, j] += 10 ^ (pars["logn_" * pr.ind].val) * pr.coll["pH2"][i, j].rate(pars["logT_" * pr.ind].val) * f_H2 / (1 + otop)
                        pr.coll_rate[i, j] += 10 ^ (pars["logn_" * pr.ind].val) * pr.coll["oH2"][i, j].rate(pars["logT_" * pr.ind].val) * f_H2 * otop / (1 + otop)
                        if f_He != 0
                            pr.coll_rate[i, j] += 10 ^ (pars["logn_" * pr.ind].val) * pr.coll["He4"][i, j].rate(pars["logT_" * pr.ind].val) * f_He / (f_He + 1 - m_fr / 2)
                        end
                    end
                end
            end
        end
    end
    #println(pr.coll_rate)
end

mutable struct coll
    i::Int64
    j::Int64
    rate::Interpolations.Extrapolation
end

mutable struct pyratio
    ind::String
    pars::Array{String}
    species::String
    num::Int64
    Eij::Array{Float64, 2}
    Aij::Array{Float64, 2}
    Bij::Array{Float64, 2}
    coll_rate::Array{Float64, 2}
    rad_rate::Array{Float64, 2}
    pump_rate::Array{Float64, 2}
    coll::Dict{}
end

##############################################################################

mutable struct hst_instr_f
    y::Vector{Float64}
    step::Float64
end


function read_hst_instr()
    list = [] #Vector{Float64}
    step = Float64
    open("data/hst/hst-kernel-lp3-cen1220-lw-1260.dat") do f
        for (i, line) in enumerate(eachline(f))
           #println(i)
           if i == 1
               #println(i,line)
               step = parse(Float64,line)
               #print(eltype(step),step)
               #print(" ")
           else
               #println(step)
               #println(i,line)
               push!(list,parse(Float64,line))
           end
        end
        #println(step)
        #println(list)
    end
    hst_instr = hst_instr_f(list,step)
    return hst_instr
end



mutable struct cheb
    c::Vector{String}
    left::Float64
    right::Float64
    disp::Float64
end

mutable struct spectrum
    x::Vector{Float64}
    y::Vector{Float64}
    unc::Vector{Float64}
    unclow::Vector{Float64}
    mask::BitArray
    resolution::Float64
    lines::Vector{line}
    disps::Float64
    dispz::Float64
    cont::Vector{Any}
end



function prepare(s, pars, add)
    #c = Vector{Any}
    #println(append!(c,  [cheb([], 0, 0, 0)]))
    spec = Vector(undef, size(s)[1])
    for (i, si) in enumerate(s)
        spec[i] = spectrum(si.spec.norm.x, si.spec.norm.y, si.spec.norm.err, si.spec.norm.errm, si.mask.norm.x .== 1, si.resolution, prepare_lines(si.fit_lines), 0, 0, prepare_cheb(pars, i))
    end
    update_pars(pars, spec, add)
    return spec
end


function correct_continuum(conts, pars, x)
    c = ones(size(x))
    for cont in conts
         m = (x .> cont.left) .& (x .< cont.right)
         cheb = ChebyshevT([pars[name].val for name in cont.c])
         c[m] = cheb.((x[m] .- cont.left) .* 2 ./ (cont.right - cont.left) .- 1)
         #if any([occursin("hcont", p.first) for p in pars])
         #   c[m] *= (1 + parse(Float64, split(pars[cont.c[1]].addinfo, "_")[4]) * randn(1)[1] * pars["hcont"].val)
         #   #println(parse(Float64, split(pars[cont.c[1]].addinfo, "_")[4]), " ", randn(1)[1], " ", pars["hcont"].val)
         #end
    end
    return c
end

function line_profile(line, x; toll=1e-6)
    if line.stack == -1
        return exp.( - line.tau0 .* real.(SpecialFunctions.erfcx.(line.a .- im .* (x .- line.l) ./ line.ld)))
    else
        tau = line.tau0 .* real.(SpecialFunctions.erfcx.(line.a .- im .* (x .- line.l) ./ line.ld))
        tau_low =  tau .* 10 ^ (line.stv["stNl"] - line.logN)
        tau_up = tau .* 10 ^ (line.stv["stNu"] - line.logN)
        p = (exp.(- tau_low) .- exp.(- tau_up) .* 10 ^ ((line.stv["stNu"] - line.stv["stNl"]) * (line.stv["sts"] + 1)) .- (gamma.(line.stv["sts"] + 2, tau_low) .- gamma.(line.stv["sts"] + 2, tau_up)) ./ tau_low .^ (line.stv["sts"] + 1)) ./ (1 - 10 ^ ((line.stv["stNu"]-line.stv["stNl"]) * (line.stv["sts"] + 1)))
        p[p .< toll] .= toll
        return p
    end
end

function calc_spectrum(spec, pars; ind=0, regular=-1, regions="fit", out="all")
    println("spec_res: ", spec.resolution)
    timeit = 1
    if timeit == 1
        start = time()
        println("start ", spec.resolution)
    end

    line_mask = update_lines(spec.lines, pars, ind=ind)

    x_instr = 1.0 / spec.resolution / 2.355
    x_grid = -1 .* ones(Int8, size(spec.x)[1])
    x_grid[spec.mask] = zeros(sum(spec.mask))
    for i in findall(!=(0), spec.mask[2:end])
        x_grid[i] = max(x_grid[i], round(Int, (spec.x[i] - spec.x[i-1]) / spec.x[i] / x_instr * 2))
    end
    for i in findall(!=(0), spec.mask[1:end-1] - spec.mask[2:end])
        for k in binsearch(spec.x, spec.x[i] * (1 - 6 * x_instr), type="min"):binsearch(spec.x, spec.x[i] * (1 + 6 * x_instr), type="max")
            x_grid[k] = max(x_grid[k], round(Int, (spec.x[i] - spec.x[i-1]) / spec.x[i] / x_instr * 2))
        end
    end

    for line in spec.lines[line_mask]
        line.dx = voigt_range(line.a, 0.001 / line.tau0)
        x, r = voigt_step(line.a, line.tau0)
        x = line.l .+ x * line.ld
        if size(x)[1] > 0
            i_min, i_max = binsearch(spec.x, x[1], type="min"), binsearch(spec.x, x[end], type="max") - 1
            if i_max - i_min > 0 && i_min > 0
                for i in i_min:i_max
                    k_min, k_max = binsearch(x, spec.x[i]), binsearch(x, spec.x[i+1])
                    x_grid[i] = max(x_grid[i], Int(floor((spec.x[i+1] - spec.x[i]) / (0.1 / maximum(r[k_min:k_max]) * line.ld)))+1)
                end
            end
            i_min, i_max = binsearch(spec.x, x[1] * (1 - 3 * x_instr), type="min"), binsearch(spec.x, x[end] * (1 + 3 * x_instr), type="max")
            if i_max - i_min > 1 && i_min > 1
                for i in i_min:i_max
                    x_grid[i] = max(x_grid[i], round(Int, (spec.x[i] - spec.x[i-1]) / line.l / x_instr * 3))
                end
            end
        end

    end

    if timeit == 1
        println("update ", time() - start)
    end

    x_grid[x_grid .>= 0] = round.(imfilter(x_grid[x_grid .>= 0], ImageFiltering.Kernel.gaussian((2,))))

    if timeit == 1
        println("grid conv ", time() - start)
    end

    if regular == 0
        x = spec.x[x_grid .> -1]
        x_mask = ~isinf(x)
    else
        x = [0.0]
        x_mask = Vector{Int64}(undef, 0)
        k = 1
        if regular == -1
            for i in 1:size(x_grid)[1]-1
                if spec.mask[i] > 0
                    append!(x_mask, k)
                end
                if x_grid[i] == 0
                    splice!(x, k, [spec.x[i], spec.x[i]])
                    k += 1
                elseif x_grid[i] > 0
                    step = (spec.x[i+1] - spec.x[i]) / (x_grid[i] + 1)
                    splice!(x, k, range(spec.x[i], length=x_grid[i]+2, step=step))
                    k += x_grid[i]+1
                end

            end
        elseif regular > 0
            for i in 1:size(x_grid)[1]-1
                if spec.mask[i] > 0
                    append!(x_mask, k)
                end
                if x_grid[i] > -1 && x_grid[i+1] > -1
                    step = (spec.x[i+1] - spec.x[i]) / (regular + 1)
                    splice!(x, k, range(spec.x[i], stop=spec.x[i+1], length=regular+2))
                    k += kind + 1
                end
            end
        end
    end

    if timeit == 1
        println("make grid ", time() - start)
    end

    if ~any([occursin("cf", p.first) for p in pars])
        y = ones(size(x))
        for line in spec.lines[line_mask]
            i_min, i_max = binsearch(x, line.l - line.dx * line.ld, type="min"), binsearch(x, line.l + line.dx * line.ld, type="max")
            t = line_profile(line, x[i_min:i_max])
            @. @views y[i_min:i_max] = y[i_min:i_max] .* t
        end
    else
        y = zeros(size(x))
        cfs, inds = [], []
        for (i, line) in enumerate(spec.lines[line_mask])
            append!(cfs, line.cf)
            append!(inds, i)
        end
        for l in unique(cfs)
            if l > -1
                cf = pars["cf_" * string(l)].val
            else
                cf = 1
            end
            profile = zeros(size(x))
            for (i, c) in zip(inds, cfs)
                if c == l
                    line = spec.lines[line_mask][i]
                    i_min, i_max = binsearch(x, line.l - line.dx * line.ld, type="min"), binsearch(x, line.l + line.dx * line.ld, type="max")
                    t = line_profile(line, x[i_min:i_max])
                    @. @views profile[i_min:i_max] += log.(t)
                end
            end
            #y += log.(exp.(profile) .* cf .+ (1 .- cf))
            y += real.(log.(exp.(profile) .* cf .+ (1 .- cf) .+ 0im))
        end
        y = exp.(y)
    end

    if timeit == 1
        println("calc lines ", time() - start)
        #println(size(x))
    end

    #if any([occursin("disp", p.first) for p in pars])
    #    n = Int(sum([occursin("disp", p.first) for p in pars]) / 2)
    #    for i in 0:n-1
    #        println(i)
    #        for p in pars
    #            #println(p.first, " ", occursin("disp", p.first), " ", parse(Int, split(p.first, "_")[2]) == i, " ", occursin("disp", p.first) & (parse(Int, split(p.first, "_")[2]) == i))
    #            if occursin("disp", p.first) & (parse(Int, split(p.first, "_")[2]) == i)
    #                println(p.first, " ", p.second.addinfo)
    #            end
    #        end
    #    end
    #end
    #println(spec.dispz, " ", spec.disps)
    if (spec.dispz != 0) & (spec.disps != 0)
        inter = LinearInterpolation(x, y, extrapolation_bc=Flat())
        y = inter(x .+ (x .- spec.dispz) .* spec.disps)
    end
    println("spec_res(2): ", spec.resolution)
    if spec.resolution != 0
        #start = time()
        y = 1 .- y
        y_c = Vector{Float64}(undef, size(y)[1])
        if 1<0
            for (i, xi) in enumerate(x)
                sigma_r = xi / spec.resolution / 1.66511
                k_min, k_max = binsearch(x, xi - 3 * sigma_r), binsearch(x, xi + 3 * sigma_r)
                #println(k_min, "  ", k_max)
                instr = exp.( -1 .* ((view(x, k_min:k_max) .- xi) ./ sigma_r ) .^ 2)
                s = 0
                @inbounds for k = k_min+1:k_max
                    s = s + (y[k] * instr[k-k_min+1] + y[k-1] * instr[k-k_min]) * (x[k] - x[k-1])
                end
                y_c[i] = s / 2 / sqrt(π) / sigma_r  + y[k_min] * (1 - SpecialFunctions.erf((xi - x[k_min]) / sigma_r)) / 2 + y[k_max] * (1 - SpecialFunctions.erf((x[k_max] - xi) / sigma_r)) / 2
                #sleep(5)
            end
        else
            finstr = read_hst_instr()
            println("read file: ", time() - start)

            sigma_instr = Float64
            kernel_size = Int64
            kernel_size =length(finstr.y)
            hst_kernel = zeros(kernel_size)
            hst_kernel_x = zeros(kernel_size)
            step = finstr.step

            sigma_instr = step*length(finstr.y)/2
            #println("sigma_instr",sigma_instr)
            hst_kernel = finstr.y./step
            hst_kernel_x = collect(range(0,kernel_size-1,step=1))*step


            kernel_interp = LinearInterpolation(hst_kernel_x, hst_kernel, extrapolation_bc=Flat())  # builds up interpolated function from input spectrum
            #println("interp", time() - start)
            print("sigma_instr ", sigma_instr)
            for (i, xi) in enumerate(x)
                k_min, k_max = binsearch(x, xi - sigma_instr/1), binsearch(x, xi + sigma_instr/1)
                s = 0
                println(xi, "  ", k_min, "  ", k_max, " ", x[k_min], " ", x[k_max], " ", y[k_min], " ", y[k_max])
                @inbounds for k = k_min+1:k_max-1
                    if y[k] != 0
                        if y[k-1] != 0
                            s = s + (y[k] * kernel_interp(x[k] - (xi - sigma_instr)) + y[k-1] * kernel_interp(x[k-1] - (xi - sigma_instr)))/2 * (x[k] - x[k-1])
                            println(s, " ", y[k], " ",  kernel_interp(x[k] - (xi - sigma_instr)))
                        end
                    end
                end
                println("res: ", s)
                y_c[i] = s
            end
        end
        #println("convolve hst", time() - start)
        if timeit == 1
            println("convolve ", time() - start)
        end

        if size(spec.cont)[1] > 0
            y_c = (1 .-y_c) .* correct_continuum(spec.cont, pars, x)
        else
            y_c = 1 .- y_c
        end

        if out == "all"
            return x, y_c
        elseif out == "init"
            #println("all done ", sum(y_c[x_mask]))
            return y_c[x_mask]
        end
    else
        if out == "all"
            return x, y
        elseif out == "init"
            return y[x_mask]
        end
    end

end


function fitLM(spec, p_pars, add; tieds=Dict())

    function cost(p)
        i = 1
        #println(p)
        for (k, v) in pars
            if v.vary == 1
                pars[k].val = p[i]
                i += 1
            end
        end

        update_pars(pars, spec, add)

        res = Vector{Float64}()
        for s in spec
            if sum(s.mask) > 0
                append!(res, (calc_spectrum(s, pars, out="init") .- s.y[s.mask]) ./ s.unc[s.mask])
            end
        end
        return res
    end

    pars = make_pars(p_pars, tieds=tieds, z_ref=true)

    #println("fitLM ", pars)
    params = [p.val for (k, p) in pars if p.vary == true]
    lower = [p.min for (k, p) in pars if p.vary == true]
    upper = [p.max for (k, p) in pars if p.vary == true]

    #println(params, " ", lower, " ", upper)
    fit = LsqFit.lmfit(cost, params, Float64[]; maxIter=100, lower=lower, upper=upper, show_trace=true, x_tol=1e-3)
    param, sigma, covar = copy(fit.param), stderror(fit), estimate_covar(fit)

    #println(dof(fit))
    i = 1
    for (k, p) in pars
        if p.vary == true
            if occursin("z", p.name)
                param[i] = z_to_v(v=param[i], z_ref=p.ref)
                sigma[i] = z_to_v(v=sigma[i], z_ref=p.ref) - p.ref
            end
            println(k, ": ", param[i], " ± ", sigma[i])
            i += 1
        end
    end
    #println(covar)

    return dof(fit), param, sigma
end