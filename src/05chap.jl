using JSON: end_array
using CSV: columnname
using DataFrames
using CSV
using Statistics
using Flux
using BenchmarkTools
using JSON
using Plots; unicodeplots()
using JSONTables
using JSON3

df = DataFrame(CSV.File("./data/primary_data.csv"))

df3 = DataFrame(CSV.File("./data/jsonformatter.txt"))

size(df)  #115 mil

histogram(df.age) #esperdo condensado entre los 62 a mas de 100 anos, se reduce a partir de los 70 al parecer

# Hay que estabkecer una funcion para parsear el diccionario en el dataframe

columns_iter = ["sex", "race","hispanic" , "alcohol" ,"drug" ,"body_part" ,"body_part_2","diagnosis","diagnosis_2","disposition" , "location","fire_involvement" ,"product_1" ,"product_2" ,"product_3" ]


function dictionary_transform(dictionary, var, value)
	if string(value) != "missing"
		#transform_var = convert(Int64,value)
		equivalent_value = dictionary[!, var * "." * string(value)]
		return first(equivalent_value)
	end
	return "missing"
end


for i in columns_iter
	println(i)
	if  eltype(df[!,i])== Union{Missing, Float64} 
		df[!,i] = convert.(Union{Missing,Int64}, df[!,i])
	end
	df[!,i] = string.(df[!,i])
	println("pasa la trans")
	for j in 1:size(df)[1]
		df[j, i] = dictionary_transform(df3, i, df[j, i])
	end
end
df[j, i] = dictionary_transform(df3, i, df[j, i])

CSV.write("./data/primary_data_with_labels.csv", df)
