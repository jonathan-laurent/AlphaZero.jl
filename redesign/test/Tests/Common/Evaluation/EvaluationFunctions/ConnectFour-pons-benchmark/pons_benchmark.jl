# A benchmark to evaluate connect-four agents is available at:
#   http://blog.gamesolver.org/solving-connect-four/02-test-protocol/
# It features 6000 positions along with their expected scores.

# Let `ne` the number of elapsed moves. A game stage is either
# `begin` (ne <= 14), `middle` (14 < ne <= 28) or `end` (ne > 28)
const STAGES = [:beginning, :middle, :end]

# Let `nr` the number of remaining moves. A difficulty level is either
# `easy` (nr < 14), `medium` (14 <= nr < 28) or `hard` (nr > 28).
const DIFFICULTIES = [:easy, :medium, :hard]

const BENCHMARKS_DIR = joinpath(@__DIR__, "benchmark")


function _parse_test_filename(name)
    m = match(r"^Test_L(\d)_R(\d)$", name)
    L = parse(Int, m.captures[1])
    R = parse(Int, m.captures[2])
    return (STAGES[L], DIFFICULTIES[R])
end

struct Bench
    stage::Symbol
    difficulty::Symbol
    entries::Vector{Tuple{BitwiseConnectFourEnv, Array, Vector{Int}}}
end

function _optimal_on(process, get_move_fn, env, acts)
    na = BatchedEnvs.num_actions(BitwiseConnectFourEnv)
    valid_actions = [BatchedEnvs.valid_action(env, a) for a in 1:na]
    action = get_move_fn(env)
    qs = [(valid_actions[a] ? _get_solver_score(process, acts, a) : -Inf) for a in 1:na]
    println(qs)
    println(action)
    return sign(qs[action]) == maximum(sign, qs)
end


function _load_benchmarks(dir, process)
    @assert isdir(dir)
    benchmarks = Bench[]
    files = readdir(dir)
    for bf in files
        meta = _parse_test_filename(bf)
        f = joinpath(dir, bf)
        (isnothing(meta) || !isfile(f)) && continue

        stage, difficulty = meta
        entries = []
        for L in readlines(f)
            L = split(L)
            action_seq = L[1]
            action_seq_list = [parse(Int, a) for a in action_seq]
            env = _create_pos_env(action_seq_list)
            state = BatchedEnvs.vectorize_state(env)
            optimal_actions = _get_optimal_actions(process, env, action_seq)
            push!(entries, (env, state, optimal_actions))
        end
        push!(benchmarks, Bench(stage, difficulty, entries))
    end
    rank(b) = (
        findfirst(==(b.difficulty), DIFFICULTIES),
        findfirst(==(b.stage), STAGES))
    sort!(benchmarks, by=rank)

    return benchmarks
end

function get_pons_benchmark_fn(global_times, global_errs, kwargs)
    @warn "Disable other evaluation-functions/benchmarks to get accurate results on the " *
          "pons benchmark."

    cmd = pipeline(Cmd(`./c4solver`, dir="connect4"), stderr=devnull)
    process = open(cmd, "r+")
    benchmarks = _load_benchmarks(BENCHMARKS_DIR, process)
    close(process)

    start_time = time()

    function az_pons_benchmark(loggers, nn, _, _)
        bench_start_time = time()

        errs = []
        for bench in benchmarks
            envs = DeviceArray(kwargs["device"])([entry[1] for entry in bench.entries])
            device_nn = (kwargs["device"] == GPU()) ? Flux.gpu(nn) : Flux.cpu(nn)
            mcts_config = init_mcts_config(kwargs["device"], device_nn, kwargs["config"])

            tree = MCTS.explore(mcts_config, envs)
            policy = Array(MCTS.evaluation_policy(tree, mcts_config))

            num_errors = 0
            for i in eachindex(envs)
                action = policy[i]
                (action ∉ bench.entries[i][3]) && (num_errors += 1)
            end
            push!(errs, num_errors / length(envs))
        end

        with_logger(loggers["tb"]) do
            @info "eval" az_posbench_beggining_easy=errs[1] log_step_increment=0
            @info "eval" az_posbench_middle_easy=errs[2] log_step_increment=0
            @info "eval" az_posbench_end_easy=errs[3] log_step_increment=0
            @info "eval" az_posbench_beggining_medium=errs[4] log_step_increment=0
            @info "eval" az_posbench_middle_medium=errs[5] log_step_increment=0
            @info "eval" az_posbench_beginning_hard=errs[6] log_step_increment=0
        end

        bench_end_time = time()

        # increment start time by the time it took to evaluate the benchmarks
        start_time += bench_end_time - bench_start_time

        eval_time = time() - start_time
        push!(global_times["az"], eval_time)
        push!(global_errs["az"], errs)
    end

    function nn_pons_benchmark(loggers, nn, _, _)
        bench_start_time = time()

        cpu_nn = Flux.cpu(nn)
        errs = []
        for bench in benchmarks
            envs = [entry[1] for entry in bench.entries]
            valid_actions = get_valid_actions(envs)

            states = hcat([entry[2] for entry in bench.entries]...)
            _, logits = forward(cpu_nn, states, false)

            num_errors = 0
            for i in eachindex(envs)
                action_mask = typemin(Float32) .* .!valid_actions[:, i]
                masked_env_logits = logits[:, i] .+ action_mask
                action = argmax(masked_env_logits)
                (action ∉ bench.entries[i][3]) && (num_errors += 1)
            end
            push!(errs, num_errors / length(envs))
        end

        with_logger(loggers["tb"]) do
            @info "eval" nn_posbench_beggining_easy=errs[1] log_step_increment=0
            @info "eval" nn_posbench_middle_easy=errs[2] log_step_increment=0
            @info "eval" nn_posbench_end_easy=errs[3] log_step_increment=0
            @info "eval" nn_posbench_beggining_medium=errs[4] log_step_increment=0
            @info "eval" nn_posbench_middle_medium=errs[5] log_step_increment=0
            @info "eval" nn_posbench_beginning_hard=errs[6] log_step_increment=0

        end

        bench_end_time = time()

        # increment start time by the time it took to evaluate the benchmarks
        start_time += bench_end_time - bench_start_time

        eval_time = time() - start_time
        push!(global_times["nn"], eval_time)
        push!(global_errs["nn"], errs)
    end

    return az_pons_benchmark, nn_pons_benchmark
end
