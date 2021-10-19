using Flux
using Flux: onehotbatch, onecold, logitcrossentropy
using Flux: DataLoader
using MLDatasets, CUDA
import Dates
import BSON

Base.@kwdef struct Args
    η::Float32 = 3e-4
    batchsize::Integer = 128
    epochs::Integer = 20
    infotime::Integer = 2
    usecuda::Bool = true
    savemodel::Bool = true
    savepath::String = "runs/"
end

function dataloaders(args)
    ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

    xtrain, ytrain = MNIST.traindata(Float32)
    xtest, ytest = MNIST.testdata(Float32)

    xtrain, xtest = flatten(xtrain), flatten(xtest)

    ytrain = onehotbatch(ytrain, sort(unique(ytrain)))
    ytest = onehotbatch(ytest, sort(unique(ytest)))

    trainloader = DataLoader((xtrain, ytrain); batchsize=args.batchsize, shuffle=true)
    testloader = DataLoader((xtest, ytest); batchsize=args.batchsize)

    trainloader, testloader
end

function eval_loss_accuracy(loader, model, device)
    acc = Float32(0)
    ls = Float32(0)
    ntot = UInt64(0)

    for (x′, y′) ∈ loader
        x, y = device(x′), device(y′)
        ŷ = model(x)
        ls += logitcrossentropy(ŷ, y, agg=sum)
        acc += sum(onecold(ŷ) .== onecold(y))
        ntot += size(x, ndims(x))
    end

    round(ls/ntot, digits=4), round(100acc/ntot, digits=4)
end

function train(; kws...)
    args = Args(; kws...)

    device = (args.usecuda && CUDA.functional()) ? gpu : cpu

    model = Chain(Dense(28*28, 32, relu), Dense(32, 10), softmax) |> device

    ps = params(model)

    trainloader, testloader = dataloaders(args)

    opt = ADAM(args.η)

    function report(epoch)
        train = eval_loss_accuracy(trainloader, model, device)
        test = eval_loss_accuracy(testloader, model, device)
        @info "Epoch $epoch\n\t• Train loss/acc: $train\n\t• Test loss/acc: $test"
    end

    @info "$(args.epochs) training epochs beginning now."
    report(0)
    for epoch ∈ 1:args.epochs
        for (x′, y′) ∈ trainloader
            x, y = device(x′), device(y′)
            gs = gradient(ps) do
                logitcrossentropy(model(x), y)
            end
            Flux.Optimise.update!(opt, ps, gs)
        end
        epoch % args.infotime == 0 && report(epoch)
    end

    # Saving model
    if args.savemodel
        !ispath(args.savepath) && mkpath(args.savepath)
        modelpath = joinpath(args.savepath, "model_$(Dates.today()).bson")
        let m = cpu(model)
            BSON.@save modelpath m
        end
        @info "Model saved in $modelpath."
    end
    return cpu(model)
end

if abspath(PROGRAM_FILE) == @__FILE__
    cd(@__DIR__)
    model = @time train()
end
