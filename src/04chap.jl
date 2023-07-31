using DataFrames
using CSV
using Statistics
using Flux

using MLDatasets
using Plots; unicodeplots()

# load training set
train_x, train_y = MNIST(split=:train)[:]
# load test set
test_x,  test_y  =MNIST(split=:test)[:]


# Select just one picture
heatmap(train_x[:, :, 3])

heatmap(reshape(train_x[:, :, 3],  ))

train_x[1:28, 1:28, 1]

train_y

pixel_avg = zeros(Float64, 28,28)

for i in 1:size(train_x)[1]
		for j in 1:size(train_x)[1]
		pixel_avg[i,j] = mean(train_x[i,j,findall(x -> x ==5 , train_y)])
		end
end

heatmap(pixel_avg)

mean(train_x[10,10,findall(x -> x ==5 , train_y)])

function matrix_avg(y_value)
		
    pixel_avg = zeros(Float64, 28,28)

    for i in 1:size(train_x)[1]
		for j in 1:size(train_x)[1]
				pixel_avg[i,j] = mean(train_x[i,j,findall(x -> x ==y_value , train_y)])
		end
    end
    return  pixel_avg
end


mean(abs.(a - b))

test_x[:,:,1] 

mean(train_x[10,10,findall(x -> x ==5 , train_y)])

function absolute_error_avg(real_matrix , model_matrix)
		return mean(abs.(real_matrix - model_matrix))
end

function prediction()
	Nothing
end
