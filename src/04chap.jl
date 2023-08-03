using Plots: text_box_width
using DataFrames
using CSV
using Statistics
using Flux
using BenchmarkTools

using MLDatasets
using Plots; unicodeplots()

# load training set
train_x, train_y = MNIST(split=:train)[:]
# load test set
test_x,  test_y  =MNIST(split=:test)[:]


# Select just one picture
heatmap(train_x[:, :, 3])

pixel_avg = zeros(Float64, 28,28)

for i in 1:size(train_x)[1]
		for j in 1:size(train_x)[1]
		pixel_avg[i,j] = mean(train_x[i,j,findall(x -> x ==5 , train_y)])
		end
end

heatmap(pixel_avg)

#average every pixel
mean(train_x[10,10,findall(x -> x ==5 , train_y)])

# Obtener los 9 moldes
# ir iterando en cada matriz del testeo
# vamos a sacar el error absoluto para los 10 caso
# nos quedamos con el menor error


function matrix_avg(y_value)
		
    pixel_avg = zeros(Float64, 28,28)

    for i in 1:size(train_x)[1]
		for j in 1:size(train_x)[1]
				pixel_avg[i,j] = mean(train_x[i,j,findall(x -> x ==y_value , train_y)])
		end
    end
    return  pixel_avg
end


test_x[:,:,1] 

mean(train_x[10,10,findall(x -> x ==5 , train_y)])

function absolute_error_avg(real_matrix , model_matrix)
	return mean(abs.(real_matrix - model_matrix))
end

function prediction()
	model_avg_list = []
	list_predictions = []
	for i in 0:9
		push!(model_avg_list, matrix_avg(i))
	end

	for target in 1:size(test_x, 3)
		min_error = Inf
		prediction =  Inf
		for (idx, model_avg) in enumerate(model_avg_list)
			if absolute_error_avg(test_x[:,:,target], model_avg) < min_error
				min_error = absolute_error_avg(test_x[:,:,target], model_avg)
				prediction = idx -1 
			end
		end
		push!(list_predictions, prediction)
	end
	return list_predictions
end

list_results = prediction()

function accuracy(y_real, y_predict)
	sum(y_real.== y_predict) / length(y_predict)
end

accuracy(test_y, list_results)


