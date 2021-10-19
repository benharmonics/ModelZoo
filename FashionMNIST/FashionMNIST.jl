using Flux, MLDatasets, CUDA
using Flux: onehotbatch, onecold, logitcrossentropy
using Flux: DataLoader
using StatsBase: mean
import Dates
import BSON

# hyperparameters stored in a convenience struct
Base.@kwdef struct Args
    η::Float32         = 3e-4       # learning rate
    batchsize::Integer = 128
    usecuda::Bool      = true       # attempt to use GPU?
    savemodel::Bool    = true
    epochs::Integer    = 10
    infotime::Integer  = epochs÷5
    savepath::String   = "runs/"
end

function get_data(args)
    """Downloading & manipulating data"""
    ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

    (xtrain, ytrain) = FashionMNIST.traindata(Float32)
    (xtest, ytest) = FashionMNIST.testdata(Float32)

    # adding color channel dimension for Conv function (though images are B&W)
    xtrain = reshape(xtrain, size(xtrain)[1:2]..., 1, size(xtrain, ndims(xtrain)))
    xtest = reshape(xtest, size(xtest)[1:2]..., 1, size(xtest, ndims(xtest)))

    # encoding labels with onehot schema
    ytrain = onehotbatch(ytrain, sort(unique(ytrain)))
    ytest = onehotbatch(ytest, sort(unique(ytest)))

    # loading data into mini-batches
    trainloader = DataLoader((xtrain, ytrain); batchsize=args.batchsize, shuffle=true)
    testloader = DataLoader((xtest, ytest); batchsize=args.batchsize)

    trainloader, testloader
end

# Model is a CNN based on the LeNet5 architecture (Yann LeCun et al., 1998)
model = Chain(
    # First convolution step on an initially 28×28×1 image
    Conv((5, 5), 1 => 6, relu; pad=SamePad()),
    MaxPool((2,2)),

    # Second convolution step; image is 14×14×6 from the first step
    Conv((5, 5), 6 => 16, relu; pad=SamePad()),
    MaxPool((2, 2)),

    # After convolutions & maxpools, our dims are 7×7×16.
    # Flattening our set of images to (7*7*16)×N:
    flatten,

    # Two fully connected layers followed by softmax for nice probabilities:
    Dense(7*7*16, 84, relu),
    Dense(84, 10),
    softmax
)

# loss of a single batch
loss(ŷ, y) = logitcrossentropy(ŷ, y)

function loss_accuracy(loader, model, device)
    """Loss & Accuracy across all batches"""
    # Loss: mean logitcrossentropy
    # Accuracy: mean(onecold(ŷ) .== onecold(y))
    losstot = Float32(0)
    acc = Float32(0)
    ntot = UInt64(0)

    for (x′, y′) ∈ loader
        x, y = device(x′), device(y′)
        ŷ = model(x)
        losstot += loss(ŷ, y) * size(x, ndims(x))
        acc += sum(onecold(ŷ |> cpu) .== onecold(y |> cpu))
        ntot += size(x, ndims(x))
    end

    lossval = round(losstot/ntot, digits=4)
    accval = round(100acc/ntot, digits=4)

    lossval, accval
end

function train(; kws...)
    """Training the model"""
    args = Args(; kws...)

    device = (args.usecuda && CUDA.functional()) ? gpu : cpu

    m = device(model)

    ps = params(m)

    trainloader, testloader = get_data(args)

    opt = ADAM(args.η)

    # Report loss & accuracy mostly so we know the training is working
    function report(epoch)
        train = loss_accuracy(trainloader, m, device)
        test = loss_accuracy(testloader, m, device)
        @info "Epoch $epoch\n\t• Train Loss/Acc: $train\n\t• Test Loss/Acc: $test"
    end

    @info "Training started."
    report(0)
    for epoch ∈ 1:args.epochs
        # Each batch in the loader
        for (x′, y′) ∈ trainloader
            x, y = device(x′), device(y′)
            gs = gradient(ps) do
                loss(m(x), y)
            end
            Flux.Optimise.update!(opt, ps, gs)
        end
        # Logging
        epoch % args.infotime == 0 && report(epoch)
    end

    # Saving model
    if args.savemodel
        !ispath(args.savepath) && mkpath(args.savepath)
        modelpath = joinpath(args.savepath, "model_$(Dates.today()).bson")
        let model = cpu(m)
            BSON.@save modelpath model
        end
        @info "Model saved in $modelpath."
    end
    return cpu(m)
end

if abspath(PROGRAM_FILE) == @__FILE__
    cd(@__DIR__)
    m = @time train()
end
