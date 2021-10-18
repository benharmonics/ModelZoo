using Flux
using Flux: onehot, chunk, batchseq, throttle, logitcrossentropy
using StatsBase: wsample
using Base.Iterators: partition
using Dates: today
import Downloads
import BSON

# hyperparameters
Base.@kwdef struct Args
    η::Float32 = 1e-2           # learning rate
    seqlen::Integer = 50        # length of batch sequences
    nbatch::Integer = 50        # number of batches
    throttle::Integer = 30      # throttle timeout
    savepath = "runs/"          # runs saved to this directory
end

function get_data(args)
    # Download the data in this directory as "input.txt"
    url = "https://cs.stanford.edu/people/karpathy/char-rnn/shakespeare_input.txt"
    isfile("input.txt") || Downloads.download(url, "input.txt")

    # Collect all the characters in the text
    characters = collect(String(read("input.txt")))

    # Getting an alphabet of all unique characters + the underscore
    alphabet = [unique(characters)..., '_']

    text = [onehot(ch, alphabet) for ch in characters]
    stop = onehot('_', alphabet)

    N = length(alphabet)

    # partitioning the data as a sequence of batches,
    # which are collected into an array of batches.
    Xs = collect(partition(batchseq(chunk(text, args.nbatch), stop), args.seqlen))
    Ys = collect(partition(batchseq(chunk(text[2:end], args.nbatch), stop), args.seqlen))

    Xs, Ys, N, alphabet
end

function model(N::Integer)
    Chain(
        LSTM(N, 128),
        LSTM(128, 128),
        Dense(128, N)
    )
end

function train(; kws...)
    # initialize hyperparameters
    args = Args(; kws...)

    # get data
    Xs, Ys, N, alphabet = get_data(args)

    # construct model
    m = model(N)

    # loss and optimization functions
    loss(xs, ys) = logitcrossentropy.([m(x) for x ∈ xs], ys) |> sum
    opt = ADAM(args.η)

    evalcb = () -> @show loss(Xs[5], Ys[5])

    # Training
    @info "Training begun."
    Flux.train!(loss, params(m), zip(Xs, Ys), opt, cb=throttle(evalcb, args.throttle))

    # Saving model
    !ispath(args.savepath) && mkpath(args.savepath)
    modelpath = joinpath(args.savepath, "model$(today()).bson")
    let model = cpu(m)
        BSON.@save modelpath model
    end

    m, alphabet
end

function sample(m, alphabet, len; seed="")
    m = cpu(m)
    Flux.reset!(m)
    buf = IOBuffer()
    seed == "" && (seed = string(rand(alphabet)))
    write(buf, seed)
    c = wsample(alphabet, softmax([m(onehot(c, alphabet)) for c ∈ collect(seed)][end]))
    for i ∈ 1:len
        write(buf, c)
        c = wsample(alphabet, softmax(m(onehot(c, alphabet))))
    end
    String(take!(buf))
end

if abspath(PROGRAM_FILE) == @__FILE__
    cd(@__DIR__)
    m, alphabet = @time train()
    sample(m, alphabet, 1000) |> println
end
