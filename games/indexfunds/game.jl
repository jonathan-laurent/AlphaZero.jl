import AlphaZero.GI
using StaticArrays
using CSV, DataFrames, Dates
using Statistics, Random

const SEQ_LEN = 20  # how long of a preceeding sequence to collect for RNN
const FUTURE_PERIOD_PREDICT = 1  # how far into the future are we trying to predict?
const indices = ["SNP", "STOXX", "IBEX", "DAX"]
const NUM_INDICES = 4


function pct_change(input::AbstractVector{<:Number})
	[i == 1 ? missing : (input[i]-input[i-1])/input[i-1] for i in eachindex(input)]
end

function normalise(x::AbstractArray)
	ϵ=1e-5
	μ = mean(x, dims=1)
	σ = std(x, dims=1, mean=μ, corrected=false) # use this when #478 gets merged
	# σ = std(x, dims=dims, corrected=false)
	return (x .- μ) ./ (σ .+ ϵ)
end

function data_loader()
	data_array=[]
	for index in indices
		df = CSV.read("./indices_data/$(index)_daily.csv",types=Dict(:"Adj Close"=>Float64))
		df[!,:Date]=[Dates.date2epochdays(i) for i in df[!,:Date]]
		select!(df,[1,6])
		rename!(df,[Symbol(string(index,"_",i)) for i in names(df)])
		rename!(df,Symbol(string(index,"_Date"))=>:Date)
		if isempty(data_array)
			data_array=df
		else
			data_array=innerjoin(data_array,df,on=:Date)
		end
	end
	sort!(data_array,:Date)
	select!(data_array,2:5)
	data_array.=ifelse.(isnan.(data_array), missing, data_array)
	data_array.=ifelse.(isnothing.(data_array), missing, data_array)
	data_array=dropmissing(data_array)
	select!(data_array, names(data_array) .=> pct_change .=> names(data_array))
	data_array=dropmissing(data_array)
	select!(data_array, names(data_array) .=> normalise .=> names(data_array))
	return data_array
end

function sequencer()
	data=Matrix(data_loader())
	sequences=Array{Float64}(undef,SEQ_LEN,4,size(data)[1]-SEQ_LEN-1)
	futures=Array{Float64}(undef,1,4,size(data)[1]-SEQ_LEN-1)
	sequences_futures=[]
	for i in 1:size(data)[1]-SEQ_LEN-1
		sequences[:,:,i]=data[i:i+SEQ_LEN-1,:]
		futures[:,:,i]=data[i+SEQ_LEN,:]
		push!(sequences_futures,(sequences[:,:,i],futures[:,:,i]))
	end

	len=size(sequences_futures)[1]
	sequences_futures=sequences_futures[shuffle(1:len)]
	return sequences_futures
end


const NUM_TIMESTEPS = SEQ_LEN
const Transaction = Float64
const M_HANNA = true
const Cell = Union{Nothing, Transaction}
const Board = SMatrix{NUM_TIMESTEPS, 4,  Cell}
const Predictions=SMatrix{1, 4,  Cell}
const Futures=SMatrix{1, 4,  Transaction}
# const INITIAL_Board = Board(repeat([nothing], NUM_TIMESTEPS,4))
const PREDICTIONS = Predictions(repeat([nothing], 1,4))
# const Futures = Futures(repeat([nothing], 1,4))
const INITIAL_State = (board=Board,predictions=PREDICTIONS,futures=Futures)
const SEQUENCE = sequencer()
const LEN_SEQ = collect(1:length(SEQUENCE))

function INITIAL_STATE(SEQ)
	INITIAL_BOARD = Board(SEQUENCE[SEQ][1])
	FUTURES = Futures(SEQUENCE[SEQ][2])
	return (board=INITIAL_BOARD,predictions=PREDICTIONS,futures=FUTURES)
end

mutable struct Game <: GI.AbstractGame
	board :: Board
	predictions:: Predictions
	futures:: Futures
	function Game(SEQ)
		state=INITIAL_STATE(SEQ)
		return Game(state.board,state.predictions,state.futures)
	end
end

GI.State(::Type{Game}) = typeof(INITIAL_State)

GI.Action(::Type{Game}) = Float64

GI.two_players(::Type{Game}) = false

#####
##### Defining winning conditions
#####

function has_won(g::Game)
	if !isnothing.(g.predictions)
		return sum(g.predictions.*sign.(g.futures))>0
	else
		return false
	end
end

#####
##### Game API
#####

const ACTIONS = collect(1:2^NUM_INDICES)

function actions_list()
	array=[]
	for i=[1,-1],j=[1,-1],k=[1,-1],l=[1,-1]
		push!(array,[i,j,k,l])
	end
	return array
end

const Actions=actions_list()

GI.actions(::Type{Game}) = Actions

GI.actions_mask(g::Game) = map(isnothing, g.predictions)

GI.current_state(g::Game) = Game(rand(LEN_SEQ))

GI.white_playing(::Type{Game}) = true

function terminal_white_reward(g::Game)
	has_won(g) && return 1.
	has_won(g) || return -1.
end

GI.game_terminated(g::Game) = !isnothing(terminal_white_reward(g))

function GI.white_reward(g::Game)
	return terminal_white_reward(g)
end

function GI.play!(g::Game, action)
	g.predictions = Predictions(Actions(action))
end

#####
##### Simple heuristic for minmax
#####

# function alignment_value_for(g::Game, player, alignment)
# 	γ = 0.3
# 	N = 0
# 	for pos in alignment
# 		mark = g.board[pos]
# 		if mark == player
# 			N += 1
# 		elseif !isnothing(mark)
# 			return 0.
# 		end
# 	end
# 	return γ ^ (BOARD_SIDE - 1 - N)
# end
#
# function heuristic_value_for(g::Game, player)
# 	return sum(alignment_value_for(g, player, al) for al in ALIGNMENTS)
# end
#
# function GI.heuristic_value(g::Game)
# 	mine = heuristic_value_for(g, g.curplayer)
# 	yours = heuristic_value_for(g, !g.curplayer)
# 	return mine - yours
# end

#####
##### Machine Learning API
#####

# function flip_colors(board)
# 	flip(cell) = isnothing(cell) ? nothing : !cell
# 	# Inference fails when using `map`
# 	return @SVector Cell[flip(board[i]) for i in 1:NUM_TIMESTEPS]
# end

# Vectorized representation: 3x3x3 array
# Channels: free, white, black
# The board is represented from the perspective of white
# (as if white were to play next)
function GI.vectorize_state(::Type{Game}, state)
	board = GI.white_playing(Game, state) ? state.board : flip_colors(state.board)
	return Float32[
	board[pos_of_xy((x, y))] == c
	for x in 1:BOARD_SIDE,
		y in 1:BOARD_SIDE,
		c in [nothing, M_HANNA, BLACK]]
end


#####
##### Symmetries
#####


# function GI.symmetries(::Type{Game}, s)
# 	return Vector[]
# 	end
#
#####
##### Interaction API
#####

# function GI.action_string(::Type{Game}, action)
# 	string(action)
# end
#
# function GI.parse_action(g::Game, str)
# 	length(str) == 1 || (return nothing)
# 	x = Int(uppercase(str[1])) - Int('A')
# 	(0 <= x < NUM_TIMESTEPS) ? x + 1 : nothing
# end
#
# function read_board(::Type{Game})
# 	n = BOARD_SIDE
# 	str = reduce(*, ((readline() * "   ")[1:n] for i in 1:n))
# 	white = ['w', 'r', 'o']
# 	black = ['b', 'b', 'x']
# 	function cell(i)
# 		if (str[i] ∈ white) M_HANNA
# 		elseif (str[i] ∈ black) BLACK
# 		else nothing end
# 	end
# 	@SVector [cell(i) for i in 1:NUM_TIMESTEPS]
# end
#
# function GI.read_state(::Type{Game})
# 	b = read_board(Game)
# 	nw = count(==(M_HANNA), b)
# 	nb = count(==(BLACK), b)
# 	if nw == nb
# 		return (board=b, curplayer=M_HANNA)
# 	elseif nw == nb + 1
# 		return (board=b, curplayer=BLACK)
# 	else
# 		return nothing
# 	end
# end
#
# using Crayons
#
# player_color(p) = p == M_HANNA ? crayon"light_red" : crayon"light_blue"
# player_name(p)  = p == M_HANNA ? "Red" : "Blue"
# player_mark(p)  = p == M_HANNA ? "o" : "x"
#
# function GI.render(g::Game; with_position_names=true, botmargin=true)
# 	pname = player_name(g.curplayer)
# 	pcol = player_color(g.curplayer)
# 	print(pcol, pname, " plays:", crayon"reset", "\n\n")
# 	for y in 1:BOARD_SIDE
# 		for x in 1:BOARD_SIDE
# 			pos = pos_of_xy((x, y))
# 			c = g.board[pos]
# 			if isnothing(c)
# 				print(" ")
# 			else
# 				print(player_color(c), player_mark(c), crayon"reset")
# 			end
# 			print(" ")
# 		end
# 		if with_position_names
# 			print(" | ")
# 			for x in 1:BOARD_SIDE
# 				print(GI.action_string(Game, pos_of_xy((x, y))), " ")
# 			end
# 		end
# 		print("\n")
# 	end
# 	botmargin && print("\n")
# end
