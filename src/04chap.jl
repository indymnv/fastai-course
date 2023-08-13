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
heatmap(train_x[:, :, 8])


# Select a picture
pixel_avg = zeros(Float64, 28,28)

for i in 1:size(train_x)[1]
		for j in 1:size(train_x)[1]
		pixel_avg[i,j] = mean(train_x[i,j,findall(x -> x ==3 , train_y)])
		end
end

#average every pixel
heatmap(pixel_avg)

#Create function
function matrix_avg(y_value)
		
    pixel_avg = zeros(Float64, 28,28)

    for i in 1:size(train_x)[1]
		for j in 1:size(train_x)[1]
				pixel_avg[i,j] = mean(train_x[i,j,findall(x -> x ==y_value , train_y)])
		end
    end
    return  pixel_avg
end



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



# Select only 3 and 7 in the dataset

# Create a filter function to select only 3 and 7
filter_fn(y) = y == 3 || y == 7

# Filter the training set
train_filter = filter_fn.(train_y)

train_x_filtered = train_x[:, :, train_filter]
train_y_filtered = train_y[train_filter]

# Filter the test set
test_filter = filter_fn.(test_y)
test_x_filtered = test_x[:, :, test_filter]
test_y_filtered = test_y[test_filter]
		

function prediction_fastai(test_x)
	matrix_avg_3 = matrix_avg(3)
	matrix_avg_7 = matrix_avg(7)

	list_predictions = []
	for target in 1:size(test_x, 3)
		if absolute_error_avg(test_x[:,:,target], matrix_avg_3) < absolute_error_avg(test_x[:,:,target], matrix_avg_7) 
			prediction = 3
		else
			prediction = 7
		end
	push!(list_predictions, prediction)
	end

	return list_predictions
end

list_results = prediction_fastai(test_x_filtered)

accuracy(test_y_filtered, list_results)

####################################################################
# NOW ITS TIME TO GO WITH FLUX
####################################################################

# Take the data and flat every row

train_x_flat = Matrix{Float32}[]
for i in 1:size(train_x)[3]
	push!(train_x_flat, reshape(train_x[:,:,i], 1, :))
end


# Initialize a random weight for every pixel ax + b require a and b
# Calculate the prediction for thee first row of your training
# Do for all the pictures
# 
#
#
#


















