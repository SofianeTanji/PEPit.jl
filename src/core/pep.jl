mutable struct PEP
    list_of_functions::Vector{AbstractFunction}
    list_of_points::Vector{Point}
    list_of_conditions::Vector{Constraint}
    list_of_performance_metrics::Vector{Expression}
    list_of_psd::Vector{PSDMatrix}
    residual::Union{Matrix{Float64},Nothing}

    function PEP()
        Point_counter[] = 0
        Expression_counter[] = 0
        Function_counter[] = 0
        Global_Constraint_counter[] = 0
        PSDMatrix_counter[] = 0
        NEXT_ID[] = 0
        
        empty!(GLOBAL_LEAF_POINTS)
        empty!(GLOBAL_LEAF_EXPRESSIONS)
        
        return new([], [], [], [], [], nothing)
    end
end


function declare_function!(pep::PEP, func_class, param; reuse_gradient=nothing)
    f = reuse_gradient === nothing ?
        func_class(param; is_leaf=true) :
        func_class(param; is_leaf=true, reuse_gradient=reuse_gradient)
    push!(pep.list_of_functions, f)
    return f
end


add_constraint!(pep::PEP, constraint::Constraint) = push!(pep.list_of_conditions, constraint)

set_initial_point!(pep::PEP) = (x = Point(); push!(pep.list_of_points, x); x)

set_initial_condition!(pep::PEP, condition::Constraint) = add_constraint!(pep, condition)

set_performance_metric!(pep::PEP, expression::Expression) = push!(pep.list_of_performance_metrics, expression)


function add_psd_matrix!(pep::PEP, matrix_of_expressions)
    push!(pep.list_of_psd, PSDMatrix(matrix_of_expressions))
    return pep.list_of_psd[end]
end



function _expression_to_jump(expr::Expression, F, G)

    pc = size(G, 1)
    ec = length(F)

    F_coeffs = zeros(Float64, ec)
    G_coeffs = zeros(Float64, pc, pc)
    const_val = 0.0

    if get_is_leaf(expr)
        @assert 0 <= expr.counter < ec "Expression counter out of bounds"
        F_coeffs[expr.counter+1] = 1.0
    else
        for (key, weight) in expr.decomposition_dict
            if key isa Expression
                @assert get_is_leaf(key)
                @assert 0 <= key.counter < ec "Expression counter out of bounds"
                F_coeffs[key.counter+1] += weight

            elseif key isa Tuple{Point,Point}
                p1, p2 = key
                @assert get_is_leaf(p1) && get_is_leaf(p2)
                @assert 0 <= p1.counter < pc && 0 <= p2.counter < pc "Point counter out of bounds"
                i, j = p1.counter + 1, p2.counter + 1
                G_coeffs[i, j] += weight

            elseif key == 1
                const_val += weight

            else
                error("Unsupported key in expression decomposition: $(typeof(key))")
            end
        end
    end

    G_coeffs .= 0.5 .* (G_coeffs .+ G_coeffs')

    sum_G = sum(G[i, j] * G_coeffs[i, j] for i in 1:pc, j in 1:pc)

    return const_val + dot(F_coeffs, F) + sum_G

end


function _eval_points_and_function_values!(pep::PEP, F_val::Vector{Float64}, G_val::Matrix{Float64}, verbose::Bool)
    ev = eigen(Symmetric(G_val))
    eig_val = ev.values
    eig_vec = ev.vectors
    if minimum(eig_val) < 0
        verbose && println("💻 PEPit:  Postprocessing: solver's output is not entirely feasible (smallest eigenvalue: $(minimum(eig_val)) < 0). Projecting Gram matrix.")
        eig_val = max.(eig_val, 0)
    end

    sqrt_diag = Diagonal(sqrt.(eig_val))
    points_values = qr(sqrt_diag * eig_vec').R

    for point_any in GLOBAL_LEAF_POINTS
        point = point_any::Point
        point._value = points_values[:, point.counter+1]
    end

    for expr_any in GLOBAL_LEAF_EXPRESSIONS
        expr = expr_any::Expression
        expr._value = F_val[expr.counter+1]
    end
end


function _apply_psd_duals!(packs)
    for (mat, psd_ref, eq_refs, sz) in packs
        mat._dual_variable_value = dual(psd_ref)
        n, m = sz
        @assert n == m
        entries = [dual(cref) for cref in eq_refs]
        mat.entries_dual_variable_value = reshape(collect(entries), n, m)
    end
end



function _eval_constraint_dual_values!(pep::PEP;
    perf_refs::Vector,
    init_refs::Vector,
    class_refs::Vector,
    main_psd_ref=nothing,
    global_psd_refs=Vector{Tuple}(),
    class_psd_refs=Vector{Tuple}())

    perf_duals = [dual(cr) for cr in perf_refs]
    pos_min = findmax(perf_duals)[2]

    if main_psd_ref !== nothing
        pep.residual = dual(main_psd_ref)
    else
        pep.residual = nothing
    end

    for (cond, cref) in zip(pep.list_of_conditions, init_refs)
        d = dual(cref)
        cond._dual_variable_value =
            cond.equality_or_inequality == "inequality" ? -d : d
    end

    idx = 1
    for f in pep.list_of_functions
        internal = _get_pep_func(f)
        for c in internal.list_of_constraints
            d = dual(class_refs[idx])
            c._dual_variable_value = c.equality_or_inequality == "inequality" ? -d : d
            idx += 1
        end
    end

    _apply_psd_duals!(global_psd_refs)
    _apply_psd_duals!(class_psd_refs)

    return pos_min
end




function _get_nb_eigs_and_corrected(M::AbstractMatrix{<:Real})
    S = 0.5 .* (M .+ M')
    ev = eigen(Symmetric(S))
    λ = ev.values
    V = ev.vectors
    maxpos = maximum(λ)
    maxneg = -minimum(λ)
    eig_threshold = max(maxpos / 1e3, 2 * maxneg)
    nonzero = λ .>= eig_threshold
    nb = count(nonzero)
    λcorr = nonzero .* λ
    Scorr = V * Diagonal(λcorr) * V'
    t = nb < length(λ) ? max(maximum(λ[.!nonzero]), 0.0) : 0.0
    return nb, t, Scorr
end



function _logdet_dimension_reduction!(model::JuMP.Model, G, objective, wc_value::Float64;
    niter::Int, eig_regularization::Float64,
    tol::Float64, verbose::Bool)

    pc = size(G, 1)

    Gval = value.(G)
    _, _, Gcorr = _get_nb_eigs_and_corrected(Gval)

    @constraint(model, objective >= wc_value - tol)

    for k in 1:niter
        W = inv(Symmetric(Gcorr + eig_regularization * I(pc)))

        @objective(model, Min, sum(W[i, j] * G[i, j] for i in 1:pc, j in 1:pc))
        verbose && println(" 💻 PEPit:  Calling SDP solver (logdet step $k)")
        optimize!(model)

        wc_value = value(objective)

        Gval = value.(G)
        nb2, thr2, Gcorr = _get_nb_eigs_and_corrected(Gval)

        if verbose
            println(" 💻 PEPit:  Solver status: $(termination_status(model)); objective value: $(wc_value)")
            println(" 💻 PEPit:  Postprocessing: $nb2 eigenvalue(s) > $thr2 after $k logdet step(s)")
        end
    end

    return wc_value
    
end


function solve!(pep::PEP;
    solver=Clarabel.Optimizer,
    # Options for solver: Clarabel.Optimizer, Mosek.Optimizer
    verbose::Bool=true,
    tracetrick::Bool=false,
    logdetiters::Int=0,
    eig_regularization::Float64=1e-3,
    tol_dimension_reduction::Float64=1e-5,
    return_full_model::Bool=false,
)


    for func in pep.list_of_functions
        add_class_constraints!(func)
    end


    model = Model(solver)
    if !verbose
        set_silent(model)
    end

    pc, ec = Point_counter[], Expression_counter[]
    verbose && println(" 💻 PEPit:  Setting up the problem: size of the main PSD matrix: $(pc)x$(pc)")

    @variable(model, objective)

    @variable(model, F[1:ec])

    @variable(model, G[1:pc, 1:pc], Symmetric)
    main_psd_ref = @constraint(model, G in PSDCone())


    verbose && println(" 💻 PEPit:  Setting up the problem: performance measure is minimum of $(length(pep.list_of_performance_metrics)) element(s)")
    perf_con_refs = Vector{Any}()
    for metric in pep.list_of_performance_metrics
        con = @constraint(model, objective <= _expression_to_jump(metric, F, G))
        push!(perf_con_refs, con)
    end


    verbose && println(" 💻 PEPit:  Setting up the problem: Adding initial conditions and general constraints ...")
    initial_con_refs = Vector{Any}()
    for cond in pep.list_of_conditions
        expr_jump = _expression_to_jump(cond.expression, F, G)
        cref = cond.equality_or_inequality == "inequality" ?
               @constraint(model, expr_jump <= 0) :
               @constraint(model, expr_jump == 0)
        push!(initial_con_refs, cref)
    end
    verbose && println(" 💻 PEPit:  Setting up the problem: initial conditions and general constraints ($(length(pep.list_of_conditions)) constraint(s) added)")


    global_psd_refs = Vector{Tuple}()
    if !isempty(pep.list_of_psd)
        verbose && println(" 💻 PEPit:  Setting up the problem: $(length(pep.list_of_psd)) lmi constraint(s) added")
        for (k, psd_matrix) in enumerate(pep.list_of_psd)
            n = psd_matrix.shape[1]
            @variable(model, M[1:n, 1:n], Symmetric)
            psd_ref = @constraint(model, M in PSDCone())
            eq_refs = Vector{Any}()
            for i in 1:n, j in 1:n
                push!(eq_refs, @constraint(model, M[i, j] == _expression_to_jump(psd_matrix[i, j], F, G)))
            end
            push!(global_psd_refs, (psd_matrix, psd_ref, eq_refs, (n, n)))
            verbose && println("\t\t Size of PSD matrix $(k): $(n)x$(n)")
        end
    end


    verbose && println(" 💻 PEPit:  Setting up the problem: interpolation conditions for $(length(pep.list_of_functions)) function(s)")
    class_con_refs = Vector{Any}()
    class_psd_refs = Vector{Tuple}()
    for (i, f) in enumerate(pep.list_of_functions)
        internal = _get_pep_func(f)

        added = 0
        for c in internal.list_of_constraints
            expr_jump = _expression_to_jump(c.expression, F, G)
            cref = c.equality_or_inequality == "inequality" ?
                   @constraint(model, expr_jump <= 0) :
                   @constraint(model, expr_jump == 0)
            push!(class_con_refs, cref)
            added += 1
        end
        verbose && println("\t\t function $i : $added scalar constraint(s) added")

        if !isempty(internal.list_of_class_psd)
            verbose && println("\t\t function $i : Adding $(length(internal.list_of_class_psd)) lmi constraint(s) ...")
            for (k, psd_matrix) in enumerate(internal.list_of_class_psd)
                n = psd_matrix.shape[1]
                @variable(model, Mc[1:n, 1:n], Symmetric)
                psd_ref = @constraint(model, Mc in PSDCone())
                eq_refs = Vector{Any}()
                for ii in 1:n, jj in 1:n
                    push!(eq_refs, @constraint(model, Mc[ii, jj] == _expression_to_jump(psd_matrix[ii, jj], F, G)))
                end
                push!(class_psd_refs, (psd_matrix, psd_ref, eq_refs, (n, n)))
                verbose && println("\t\t function $i : size of PSD matrix $(k): $(n)x$(n)")
            end
            verbose && println("\t\t function $i : $(length(internal.list_of_class_psd)) lmi constraint(s) added")
        end

        if !isempty(internal.list_of_psd)
            verbose && println("\t\t function $i : Adding $(length(internal.list_of_psd)) lmi constraint(s) ...")
            for (k, psd_matrix) in enumerate(internal.list_of_psd)
                n = psd_matrix.shape[1]
                @variable(model, Mf[1:n, 1:n], Symmetric)
                psd_ref = @constraint(model, Mf in PSDCone())
                eq_refs = Vector{Any}()
                for ii in 1:n, jj in 1:n
                    push!(eq_refs, @constraint(model, Mf[ii, jj] == _expression_to_jump(psd_matrix[ii, jj], F, G)))
                end
                push!(class_psd_refs, (psd_matrix, psd_ref, eq_refs, (n, n)))
                verbose && println("\t\t function $i : size of PSD matrix $(k): $(n)x$(n)")
            end
            verbose && println("\t\t function $i : $(length(internal.list_of_psd)) lmi constraint(s) added")
        end
    end


    verbose && println(" 💻 PEPit:  Compiling SDP")
    @objective(model, Max, objective)
    verbose && println(" 💻 PEPit:  Calling SDP solver")
    optimize!(model)
    if verbose
        println(" 💻 PEPit:  Solver status: $(termination_status(model)); optimal value: $(objective |> value)")
    end
    wc_value = value(objective)


    if tracetrick
        tol = tol_dimension_reduction
        @constraint(model, objective >= wc_value - tol)
        @objective(model, Min, sum(G[i, i] for i in 1:pc))
        verbose && println(" 💻 PEPit:  Calling SDP solver (trace heuristic)")
        optimize!(model)
        wc_value = value(objective)
        if verbose
            println(" 💻 PEPit:  Solver status: $(termination_status(model)); objective value: $(wc_value)")
        end
    end


    if logdetiters > 0
        nb, thr, _ = _get_nb_eigs_and_corrected(value.(G))
        if verbose
            println(" 💻 PEPit:  Postprocessing: $nb eigenvalue(s) > $thr before dimension reduction")
        end
        wc_value = _logdet_dimension_reduction!(model, G, objective, wc_value;
            niter=logdetiters, eig_regularization=eig_regularization,
            tol=tol_dimension_reduction, verbose=verbose)
    end



    F_val = value.(F)
    G_val = value.(G)
    _eval_points_and_function_values!(pep, F_val, G_val, verbose)


    pos_min_metric = _eval_constraint_dual_values!(pep;
        perf_refs=perf_con_refs,
        init_refs=initial_con_refs,
        class_refs=class_con_refs,
        main_psd_ref=main_psd_ref,
        global_psd_refs=global_psd_refs,
        class_psd_refs=class_psd_refs)


    if return_full_model
        return (wc_value=wc_value,
            model=model,
            variables=(objective=objective, F=F, G=G),
            constraints=(performance=perf_con_refs,
                initial=initial_con_refs,
                class=class_con_refs,
                main_psd=main_psd_ref,
                global_psd=global_psd_refs,
                class_psd=class_psd_refs),
            position_of_min_metric=pos_min_metric,
            residual=pep.residual)
    else
        return wc_value
    end
end


